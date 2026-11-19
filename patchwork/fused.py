"""The fused backend: a tile made in one compiled pass over each input, on as many threads as it is given.

The other backends interpolate an input with a compiled kernel (or numpy),
then weight and add it with numpy, an array of the whole region at a time.
That half of the work runs on one core, and holds the GIL. Here all of it is
done by ``patchwork.fused_kernels``, pixel by pixel, and Python only finds
where each input falls on the tile and reads it.

With more than one thread, a tile is made by three things at once:

- threads of this module read the inputs a few ahead of the one being added,
  which is mostly waiting for the file system,
- numba's threads add each input in turn, sharing the pixels of the tile,
- and they then turn the sums into the tile.

This suits a dask worker that runs one task at a time and has many cores, as
a cluster set up for a multithreaded imager has: each task then uses all of
them. See ``tile_threads`` for how many threads a tile is given.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from patchwork import fused_kernels as kernels
from patchwork.combine import TileSums
from patchwork.coordinates import east_north, map_to_input, tile_coordinates
from patchwork.frame import tile_region
from patchwork.io import read_window
from patchwork.logging import logger
from patchwork.polarisation import STOKES_CODES
from patchwork.resample import input_region, kernel_width, tile_threads
from patchwork.tile import (
    MosaicInput,
    TileResult,
    _add_input,
    group_members,
    weight_groups,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray

    from patchwork.coordinates import TileCoordinates
    from patchwork.frame import Frame
    from patchwork.io import Slices
    from patchwork.options import MosaicOptions
    from patchwork.primary_beam import BeamModel

READ_AHEAD_BYTES = 256 * 2**20
"""The memory each thread of a tile may hold in inputs read ahead of the one being added"""

KERNEL_CODES = {"nearest": 0, "linear": 1, "cubic": 2}
"""The code of each interpolation kernel in the compiled kernels, where any other is a Lanczos kernel"""
COMBINE_CODES = {
    "weighted": kernels.WEIGHTED,
    "average": kernels.AVERAGE,
    "sum": kernels.SUM,
}
WEIGHT_TYPE_CODES = {
    "primary_beam": kernels.BY_PRIMARY_BEAM,
    "weight_image": kernels.BY_WEIGHT_IMAGE,
    "combined": kernels.BY_BOTH,
}
WEIGHT_STATE_CODES = {
    "corrected": kernels.CORRECTED,
    "inherent": kernels.INHERENT,
    "weighted": kernels.ALREADY_WEIGHTED,
}


@functools.cache
def _warn_fewer_threads(requested: int, most: int) -> None:
    """Say, once, that numba has fewer threads than were asked for."""
    logger.warning(
        f"Making tiles with {most} threads, not the {requested} asked for, as numba has no more. "
        "It starts as many as the CPUs the process may run on, or NUMBA_NUM_THREADS"
    )


def kernel_threads(requested: int) -> int:
    """The threads the kernels make a tile with: ``patchwork.resample.tile_threads``, and no more than numba has."""
    threads = tile_threads(requested)
    most = kernels.numba_threads()
    if requested > most:
        _warn_fewer_threads(requested, most)
    return min(threads, most)


KERNEL_SHARE = 5 / 8
"""The share of a tile's threads that run the kernels by default, the rest reading.

On a 16-core node with the inputs on a striped Lustre file system the kernels
took 2.8 s a strip on 12 threads while 4 readers took 6 s to read and map the
inputs onto it; the two balance at about 10 kernel threads of 16.
"""


def split_threads(threads: int, requested: int) -> tuple[int, int]:
    """How many of a tile's ``threads`` run the kernels, and how many read the inputs.

    ``requested`` is ``ResampleOptions.kernel_threads``: a count below
    ``threads`` is used as given, and 0 picks ``KERNEL_SHARE`` of the threads
    for the kernels when there are at least four. With fewer, or ``requested``
    at least ``threads``, every thread does both.
    """
    if 0 < requested < threads:
        return requested, threads - requested
    if requested == 0 and threads >= 4:
        in_kernels = max(1, round(threads * KERNEL_SHARE))
        return in_kernels, threads - in_kernels
    return threads, threads


class _Holography(NamedTuple):
    """The holography beams of some channels of an input, as the kernels take them"""

    pattern: NDArray[np.float64]
    """The beam of each channel, shaped (channel, y, x)"""
    leakage: NDArray[np.float64]
    """The fractional leakage of each channel, shaped (channel, 3, y, x), or empty if it was not read"""
    pattern_scale: NDArray[np.int64]
    """Which of ``scales`` each channel uses"""
    scales: NDArray[np.float64]
    """The distinct frequency scales of the beams"""
    pattern_turn: NDArray[np.float64]
    """The cosine and sine of the angle from sky offsets to the axes of the grid"""
    pattern_pixel: NDArray[np.float64]
    """The signed pixel size of the grid along x and y, in radians"""
    pattern_centre: NDArray[np.float64]
    """The (x, y) pixel of the beam centre"""
    feed_turn: NDArray[np.float64]
    """The cosine and sine of twice the feed rotation"""
    cutoff_scales: NDArray[np.float64]
    """The response a cutoff is a fraction of in each channel"""


NO_HOLOGRAPHY = _Holography(
    pattern=np.zeros((0, 1, 1)),
    leakage=np.zeros((0, 3, 1, 1)),
    pattern_scale=np.zeros(0, dtype=np.int64),
    scales=np.zeros(0),
    pattern_turn=np.zeros(2),
    pattern_pixel=np.ones(2),
    pattern_centre=np.zeros(2),
    feed_turn=np.zeros(2),
    cutoff_scales=np.zeros(0),
)
"""What the kernels are given for an input without a holography beam"""


@functools.lru_cache(maxsize=64)
def _holography(model: BeamModel, start: int, stop: int) -> _Holography:
    """The holography beams of channels ``start`` to ``stop`` of an input, stacked.

    They are kept, as every tile an input covers needs them, and the tiles of a
    run of channels are made one after another.
    """
    beams = model._holography_beams(range(start, stop))
    first = beams[0]
    # As patchwork.primary_beam._shared_taps: channels whose scales agree share their taps
    keys = [round(beam.scale, 9) for beam in beams]
    distinct = list(dict.fromkeys(keys))
    rotation = first.alpha_rad + first.frame_rotation_rad
    return _Holography(
        pattern=np.array([beam.plane for beam in beams], dtype=np.float64),
        leakage=NO_HOLOGRAPHY.leakage
        if first.leakage is None
        else np.array([beam.leakage for beam in beams], dtype=np.float64),
        pattern_scale=np.array([distinct.index(key) for key in keys], dtype=np.int64),
        scales=np.array([beams[keys.index(key)].scale for key in distinct]),
        pattern_turn=np.array([np.cos(rotation), np.sin(rotation)]),
        pattern_pixel=np.array(first.pixel_scale_rad, dtype=np.float64),
        pattern_centre=np.array(first.centre_pixel, dtype=np.float64),
        feed_turn=np.array([np.cos(2 * first.alpha_rad), np.sin(2 * first.alpha_rad)]),
        cutoff_scales=np.array([beam.cutoff_scale for beam in beams]),
    )


class _Read(NamedTuple):
    """An input read over the region of a tile it contributes to"""

    item: MosaicInput
    """The input"""
    region: Slices
    """The (y, x) region of the tile, relative to the tile"""
    usable: NDArray[np.bool_]
    """Which of the channels hold an image"""
    x: NDArray[np.float64]
    """The x position in the blocks of each pixel of the region"""
    y: NDArray[np.float64]
    """The y position in the blocks of each pixel of the region"""
    blocks: NDArray[np.float32]
    """The part of each image that is interpolated, shaped (image, channel, y, x): the images of the input, then its Stokes I if that is not one of them"""
    stokes_i: int
    """The block of Stokes I, or -1 if its leakage is not removed"""


def channel_runs(usable: NDArray[np.bool_], pieces: int = 1) -> list[slice]:
    """The usable channels as runs of neighbouring channels, none longer than one part in ``pieces`` of them.

    Each run is one read of a cube, so a cube with enough usable channels is
    read in at least ``pieces`` reads, and any cube in at least one for each of
    its stretches of usable channels.
    """
    channels = np.flatnonzero(usable)
    longest = max(1, len(channels) // max(pieces, 1))
    runs = []
    for stretch in np.split(channels, np.flatnonzero(np.diff(channels) > 1) + 1):
        runs += [
            slice(int(stretch[start]), int(stretch[start : start + longest][-1]) + 1)
            for start in range(0, len(stretch), longest)
        ]
    return runs


def _uses_weight_images(item: MosaicInput) -> bool:
    """Whether an input is weighted by weight images, which the kernels do not interpolate."""
    return any(weight.paths for weight in item.weights)


class _Timing:
    """Where the time of a tile goes, to log once the tile is made.

    The reading and finding times are summed over the threads that did them, so
    they can exceed the time the tile took.
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.finding = 0.0
        """Seconds of thread time finding where inputs fall on the tile"""
        self.reading = 0.0
        """Seconds of thread time reading inputs"""
        self.read_bytes = 0
        """Bytes read from the inputs"""
        self.waiting = 0.0
        """Seconds the tile waited for an input to be read"""
        self.adding = 0.0
        """Seconds the kernels spent adding inputs"""
        self.inputs = 0
        """Inputs added"""
        self.finalising = 0.0
        """Seconds turning the sums into the tile"""
        self._lock = threading.Lock()

    def add(self, name: str, seconds: float, read_bytes: int = 0) -> None:
        """Add to one of the times, from any thread."""
        with self._lock:
            setattr(self, name, getattr(self, name) + seconds)
            self.read_bytes += read_bytes

    def log(
        self, tile: Slices, channels: range, threads: int, in_kernels: int, reading: int
    ) -> None:
        """Log the times of the tile, at DEBUG level."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        total = time.perf_counter() - self.started
        logger.debug(
            f"Tile rows {tile[0].start}-{tile[0].stop}, columns {tile[1].start}-{tile[1].stop}, "
            f"channels {channels.start}-{channels.stop}: {total:.1f} s on {threads} threads, "
            f"{in_kernels} in the kernels and {reading} reading; "
            f"waited {self.waiting:.1f} s for inputs, added {self.inputs} in {self.adding:.1f} s, "
            f"finalised in {self.finalising:.1f} s; the threads spent {self.finding:.1f} s finding inputs "
            f"and {self.reading:.1f} s reading {self.read_bytes / 2**20:.0f} MB of them"
        )


def _read_input(
    item: MosaicInput,
    tile: Slices,
    coordinates: TileCoordinates,
    options: MosaicOptions,
    channels: slice,
    readers: Executor | None = None,
    pieces: int = 1,
    timing: _Timing | None = None,
) -> _Read | None:
    """Find where an input falls on a tile and read it there, or None if it does not contribute to the tile.

    This is the start of ``patchwork.tile._add_input``, and may run in a
    thread while other inputs are read or added. With ``readers`` each of its
    images is read in up to ``pieces`` runs of channels, all at the same time.
    Channels without an image are not read, as nothing is made of them.
    """
    kernel = options.resample.kernel
    channel_range = range(channels.start or 0, channels.stop)
    started = time.perf_counter()
    region = tile_region(tile, item.bounding_box)
    if region is None:
        return None
    usable = np.array(
        [
            item.usable_channels[index] if item.usable_channels else True
            for index in channel_range
        ]
    )
    if not np.any(usable):
        return None
    x, y = map_to_input(coordinates, item.info.wcs, region)
    window = input_region(item.info.shape, x, y, kernel)
    if window is None:
        return None

    paths = list(item.planes)
    stokes_i = -1
    if item.stokes_i_path is not None and item.primary_beam is not None:
        if item.stokes_i_path not in paths:
            paths.append(item.stokes_i_path)
        stokes_i = paths.index(item.stokes_i_path)
    shape = (
        len(paths),
        len(channel_range),
        window[0].stop - window[0].start,
        window[1].stop - window[1].start,
    )
    blocks = np.empty(shape, dtype=np.float32)

    first = channel_range.start
    reads = [
        (index, run)
        for run in channel_runs(usable, pieces if readers is not None else 1)
        for index in range(len(paths))
    ]

    def read(index: int, run: slice) -> None:
        read_window(
            paths[index],
            window,
            slice(first + run.start, first + run.stop),
            out=blocks[index, run],
        )

    found = time.perf_counter()
    if readers is None:
        for index, run in reads:
            read(index, run)
    else:
        # Consumed, so that a read that fails is raised here
        list(readers.map(read, *zip(*reads, strict=True)))
    if timing is not None:
        timing.add("finding", found - started)
        read_bytes = (
            4 * shape[2] * shape[3] * sum(run.stop - run.start for _, run in reads)
        )
        timing.add("reading", time.perf_counter() - found, read_bytes)
    return _Read(
        item=item,
        region=region,
        usable=usable,
        x=np.ascontiguousarray(x - window[1].start, dtype=np.float64),
        y=np.ascontiguousarray(y - window[0].start, dtype=np.float64),
        blocks=blocks,
        stokes_i=stokes_i,
    )


def _task(
    read: _Read,
    frame: Frame,
    vectors: NDArray[np.float64],
    groups: tuple[int, ...],
    options: MosaicOptions,
    channels: slice,
) -> kernels.Task:
    """What the kernels need to add an input that has been read to a tile.

    Raises:
        ValueError: Raised if leakage is to be removed from an image that is not of Q, U or V, or with a beam that does not carry it
    """
    item = read.item
    channel_range = range(channels.start or 0, channels.stop)
    n_channels = len(channel_range)
    weight_options, polarisation = options.weight, options.polarisation
    kernel = options.resample.kernel
    model = item.primary_beam
    stokes_code = STOKES_CODES

    removes_leakage = read.stokes_i >= 0
    holography = NO_HOLOGRAPHY
    if model is not None and (
        removes_leakage or model.options.pb_model == "holography"
    ):
        holography = _holography(model, channel_range.start, channel_range.stop)
    if removes_leakage:
        leaky = tuple(code for code in item.stokes if code != stokes_code["I"])
        if any(code not in (2, 3, 4) for code in leaky):
            msg = f"Leakage is only removed from Q, U and V, not from Stokes codes {leaky}"
            raise ValueError(msg)
        if not len(holography.leakage):
            msg = "The leakage of this holography beam was not read"
            raise ValueError(msg)

    beam = kernels.NO_BEAM
    fwhm = np.ones(n_channels)
    cutoff = np.full(n_channels, float(options.primary_beam.pb_cutoff))
    if model is not None and weight_options.weight_type != "weight_image":
        if model.options.pb_model == "gaussian":
            beam = kernels.GAUSSIAN
            fwhm = np.array([model.channel(index).fwhm_rad for index in channel_range])  # type: ignore[union-attr]
        else:
            beam = kernels.HOLOGRAPHY
            cutoff = cutoff * holography.cutoff_scales

    rotates = stokes_code["Q"] in item.stokes and stokes_code["U"] in item.stokes
    to_centre = polarisation.output_frame == "centre"
    centre = (float(frame.wcs.wcs.crval[0]), float(frame.wcs.wcs.crval[1]))
    east, north = east_north(*item.info.beam_centre)

    # As patchwork.tile._group_weight gives the weight of each group without weight images
    weights = np.empty((max(groups) + 1, n_channels))
    for group, plane_weight in zip(groups, item.weights, strict=True):
        values = np.asarray(plane_weight.values, dtype=np.float64)
        weights[group] = values if len(values) == 1 else values[channels]

    return kernels.Task(
        blocks=read.blocks,
        x=read.x,
        y=read.y,
        code=KERNEL_CODES.get(kernel, 3),
        width=kernel_width(kernel),
        usable=read.usable,
        plane_block=np.arange(len(item.planes), dtype=np.int64),
        plane_stokes=np.array(item.stokes, dtype=np.int64),
        plane_group=np.array(groups, dtype=np.int64),
        weights=weights,
        stokes_i=read.stokes_i,
        removes_leakage=removes_leakage,
        q=item.stokes.index(stokes_code["Q"]) if rotates else -1,
        u=item.stokes.index(stokes_code["U"]) if rotates else -1,
        row=int(read.region[0].start),
        column=int(read.region[1].start),
        vectors=vectors,
        east=east,
        north=north,
        beam=beam,
        fwhm=fwhm,
        cutoff=cutoff,
        pattern=holography.pattern,
        leakage=holography.leakage,
        pattern_scale=holography.pattern_scale,
        scales=holography.scales,
        pattern_turn=holography.pattern_turn,
        pattern_pixel=holography.pattern_pixel,
        pattern_centre=holography.pattern_centre,
        feed_turn=holography.feed_turn,
        input_north=east_north(*(item.frame_centre or centre))[1],
        has_input_frame=item.frame_centre is not None,
        output_north=east_north(*centre)[1],
        has_output_frame=to_centre,
        weight_type=WEIGHT_TYPE_CODES[weight_options.weight_type],
        weight_state=WEIGHT_STATE_CODES[weight_options.weight_state],
        combine=COMBINE_CODES[options.combine.combine_type],
    )


def _read_ahead(
    inputs: list[MosaicInput],
    read: Callable[[MosaicInput, Executor | None], _Read | None],
    threads: int,
    ahead: int,
) -> Iterator[tuple[MosaicInput, _Read | None]]:
    """Each input with what was read of it, in order, reading up to ``ahead`` inputs at once with ``threads`` threads.

    Reading is mostly waiting for the file system, which serves many reads at
    once, so the images of the next few inputs are all read side by side while
    the kernels add the current one. An input weighted by weight images is not
    read here, and is given with None. With one thread and one input ahead,
    each input is read as it is asked for.
    """
    if threads <= 1 and ahead <= 1:
        for item in inputs:
            yield item, None if _uses_weight_images(item) else read(item, None)
        return

    with (
        # One finds where each input falls and waits for its images, the other reads them
        ThreadPoolExecutor(ahead, thread_name_prefix="patchwork-input") as finders,
        ThreadPoolExecutor(threads, thread_name_prefix="patchwork-read") as readers,
    ):
        remaining = iter(inputs)
        pending: deque[tuple[MosaicInput, Future[_Read | None] | None]] = deque()
        try:
            while True:
                while len(pending) < ahead:
                    following = next(remaining, None)
                    if following is None:
                        break
                    reading = (
                        None
                        if _uses_weight_images(following)
                        else finders.submit(read, following, readers)
                    )
                    pending.append((following, reading))
                if not pending:
                    return
                item, future = pending.popleft()
                yield item, None if future is None else future.result()
        finally:
            for _, future in pending:
                if future is not None:
                    future.cancel()


def mosaic_tile_fused(
    tile: Slices,
    frame: Frame,
    inputs: list[MosaicInput],
    options: MosaicOptions,
    channels: slice | None = None,
) -> TileResult:
    """Resample, weight and combine every input that covers one tile of the frame, over some channels, as ``patchwork.tile.mosaic_tile`` does, with the compiled kernels.

    An input weighted by weight images is added as the numba backend adds it,
    as the kernels do not interpolate weights.

    Args:
        tile (Slices): The (y, x) region of the frame to make
        frame (Frame): The frame of the mosaic
        inputs (list[MosaicInput]): The inputs of the mosaic
        options (MosaicOptions): The mosaicking options
        channels (slice | None, optional): The channels of the inputs to make. Defaults to None, the first channel.

    Returns:
        TileResult: The finished tile
    """
    combine_type = options.combine.combine_type
    channels = channels or slice(0, 1)
    channel_range = range(channels.start or 0, channels.stop)
    threads = kernel_threads(options.resample.tile_threads)
    in_kernels, reading = split_threads(threads, options.resample.kernel_threads)
    timing = _Timing()

    coordinates = tile_coordinates(
        frame.wcs, tile, step=options.resample.coordinate_step
    )
    vectors = np.ascontiguousarray(coordinates.vectors, dtype=np.float64)
    shape = (len(channel_range), *vectors.shape[1:])
    stokes = inputs[0].stokes if inputs else (0,)
    groups = weight_groups(stokes)
    members = group_members(groups)
    kept = (len(members), *shape) if combine_type != "weighted" else (0, 0, 0, 0)
    sums = kernels.Sums(
        totals=np.zeros((len(stokes), *shape)),
        weight=np.zeros((len(members), *shape)),
        inverse_weight=np.zeros(kept),
        n_inputs=np.zeros(kept),
    )
    # The same sums as the other backends keep them, for inputs added their way
    tile_sums = [
        TileSums(
            totals=tuple(sums.totals[index] for index in indices),
            weight=sums.weight[group],
            inverse_weight=sums.inverse_weight[group] if kept[0] else None,
            n_inputs=sums.n_inputs[group] if kept[0] else None,
        )
        for group, indices in enumerate(members)
    ]

    # An input is read over about the area of the tile, a little more where it is rotated
    held = 4 * (len(stokes) + 1) * int(np.prod(shape))
    ahead = max(1, min(threads, threads * READ_AHEAD_BYTES // held))

    def read_input(item: MosaicInput, readers: Executor | None) -> _Read | None:
        # Enough reads of each image to keep every thread reading while one input is read
        pieces = max(1, reading // (len(stokes) + 1))
        return _read_input(
            item, tile, coordinates, options, channels, readers, pieces, timing
        )

    waited = time.perf_counter()
    for item, read in _read_ahead(inputs, read_input, reading, ahead):
        started = time.perf_counter()
        timing.add("waiting", started - waited)
        if _uses_weight_images(item):
            _add_input(
                item, tile, frame, coordinates, tile_sums, members, options, channels
            )
            timing.inputs += 1
        elif read is not None:
            kernels.add_input(
                _task(read, frame, vectors, groups, options, channels),
                sums,
                in_kernels,
            )
            timing.inputs += 1
        waited = time.perf_counter()
        timing.add("adding", waited - started)

    images = np.empty((len(stokes), *shape), dtype=np.float32)
    weights = np.empty((len(members), *shape), dtype=np.float32)
    kernels.finalise(
        sums,
        np.array(groups, dtype=np.int64),
        COMBINE_CODES[combine_type],
        images,
        weights,
        in_kernels,
    )
    timing.add("finalising", time.perf_counter() - waited)
    timing.log(tile, channel_range, threads, in_kernels, reading)
    return TileResult(
        slices=tile,
        channels=slice(channel_range.start, channel_range.stop),
        images=tuple(images),
        weights=tuple(weights),
    )

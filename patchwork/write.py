"""Writing a mosaic: making its tiles with an executor, writing them to FITS or zarr, and writing its noise.

The work is handed to a ``concurrent.futures.Executor``: threads, or dask
workers through ``patchwork.pipeline.flow``.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import numpy as np
from astropy.io import fits

from patchwork.io import (
    beams_header,
    create_output_fits,
    data_location,
    read_rows,
    write_rows,
    write_tile_rows,
)
from patchwork.logging import logger
from patchwork.options import RunOptions
from patchwork.resample import resolve_backend, tile_threads
from patchwork.tile import mosaic_tile

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from numpy.typing import NDArray

    from patchwork.io import DataLocation, Slices
    from patchwork.options import MosaicOptions
    from patchwork.plan import MosaicNames, MosaicPlan
    from patchwork.tile import TileResult


def create_outputs(plan: MosaicPlan, run_options: RunOptions) -> None:
    """Create the empty outputs of a planned mosaic: its FITS images and weights, which tiles are written into, or all its zarr stores."""
    if run_options.output_format == "zarr":
        create_zarr_outputs(plan, run_options.zarr_chunk)
        return
    files = (*plan.names.images, *plan.names.weights)
    if run_options.fits_writer == "workers":
        # The tasks write the noise too, so it is created with the rest
        files = (*files, *plan.names.noises)
    for path, header in zip(files, plan.headers[: len(files)], strict=True):
        create_output_fits(path, header, plan.beams)


def _output_channels(plan: MosaicPlan, channels: slice) -> slice:
    """The channels of the outputs of a mosaic that hold some channels of its inputs."""
    start = channels.start - (plan.channels.start or 0)
    return slice(start, start + channels.stop - channels.start)


DEFAULT_BUFFER_BYTES = RunOptions().fits_buffer_mb * 2**20
"""The most bytes held in bands of tiles waiting to be written, unless told otherwise"""


@dataclass
class _Band:
    """The tiles of a band of rows of a mosaic, a row of tiles across the frame over some channels, gathered until all have arrived"""

    data: NDArray[np.float32]
    """The (file, channel, y, x) band of each output, as FITS holds it"""
    waiting: int
    """The tiles still to arrive"""


def write_tiles(
    plan: MosaicPlan,
    results: Iterable[TileResult],
    buffer_bytes: int = DEFAULT_BUFFER_BYTES,
) -> None:
    """Write finished tiles into the output FITS files of a mosaic, a band of rows at a time.

    Each row of a tile is a short run of bytes in a file, and file systems such
    as Lustre take as long to write a short run as a long one. So the tiles of
    each band, a row of tiles across the frame, are gathered until all have
    arrived, and the band is then written as one run of bytes for each channel
    of each file. No more than ``buffer_bytes`` are held in bands: the tiles of a
    band that would need more are written as they arrive, a row at a time.

    Args:
        plan (MosaicPlan): The plan of the mosaic, whose outputs are created
        results (Iterable[TileResult]): The finished tiles, in any order
        buffer_bytes (int, optional): The most bytes held in bands. Defaults to DEFAULT_BUFFER_BYTES, the default ``fits_buffer_mb``.

    Raises:
        RuntimeError: Raised if the results end before every tile of a band has arrived
    """
    # The noise is written from the weights once they are, so is not created yet
    files = tuple(map(data_location, (*plan.names.images, *plan.names.weights)))
    width = plan.frame.shape[1]
    tiles_in_band = Counter(tile[0].start for tile in plan.tiles)
    # By their first row and first channel
    bands: dict[tuple[int, int], _Band] = {}
    by_rows: set[tuple[int, int]] = set()
    held = 0
    # What was written, and the seconds it took, to log once all is
    n_tiles = n_bands = band_bytes = row_tiles = 0
    band_seconds = row_seconds = 0.0
    for result in results:
        n_tiles += 1
        channels = _output_channels(plan, result.channels)
        rows, columns = result.slices
        shape = (channels.stop - channels.start, rows.stop - rows.start)
        blocks = [
            np.reshape(block, (*shape, columns.stop - columns.start))
            for block in (*result.images, *result.weights)
        ]
        key = (rows.start, channels.start)
        band = bands.get(key)
        if band is None and key not in by_rows:
            size = 4 * len(files) * shape[0] * shape[1] * width
            if held + size <= buffer_bytes:
                data = np.empty((len(files), *shape, width), dtype=">f4")
                band = bands[key] = _Band(data, tiles_in_band[rows.start])
                held += size
            else:
                if not by_rows:
                    logger.info(
                        f"A band of tiles takes {size / 2**20:.0f} MB, more than the "
                        f"{(buffer_bytes - held) / 2**20:.0f} MB of fits_buffer_mb left, so its "
                        "tiles are written a row at a time, which is far slower on file systems "
                        "such as Lustre"
                    )
                # Every tile of it, as a band written whole would write over them
                by_rows.add(key)
        if band is None:
            started = time.perf_counter()
            for location, block in zip(files, blocks, strict=True):
                write_tile_rows(location, result.slices, channels, block)
            row_tiles += 1
            row_seconds += time.perf_counter() - started
            logger.debug(
                f"Wrote tile {result.slices} of channels {result.channels} a row at a time"
            )
            continue

        for index, block in enumerate(blocks):
            band.data[index, :, :, columns] = block
        band.waiting -= 1
        if band.waiting:
            continue
        started = time.perf_counter()
        for location, data in zip(files, band.data, strict=True):
            write_rows(location, rows, channels, data)
        del bands[key]
        held -= band.data.nbytes
        seconds = time.perf_counter() - started
        n_bands += 1
        band_bytes += band.data.nbytes
        band_seconds += seconds
        logger.debug(
            f"Wrote rows {rows.start}-{rows.stop} of channels {channels.start}-{channels.stop}, "
            f"{tiles_in_band[rows.start]} tiles ({band.data.nbytes / 2**20:.0f} MB), "
            f"in {seconds:.1f} s"
        )
    if bands:
        missing = ", ".join(
            f"rows from {row}, channels from {channel}" for row, channel in bands
        )
        msg = f"Not every tile of some bands of the mosaic arrived: {missing}"
        raise RuntimeError(msg)
    message = (
        f"Wrote {n_tiles} tiles: {n_bands} band{'' if n_bands == 1 else 's'} of rows, "
        f"{band_bytes / 2**30:.1f} GB, in {band_seconds:.1f} s"
    )
    if row_tiles:
        message += f", and {row_tiles} tiles row by row in {row_seconds:.1f} s"
    logger.info(message)


def write_noise(plan: MosaicPlan, strip_rows: int = 1024) -> None:
    """Write the noise of each weight of a written mosaic, the inverse of its square root, channel by channel.

    Each noise image has the header, and any BEAMS table, of its weight, in the
    units of the images of its group. Pixels with no weight are blank.

    Args:
        plan (MosaicPlan): The plan of the mosaic, whose files are written
        strip_rows (int, optional): Rows of the files worked on at a time. Defaults to 1024.
    """
    if not plan.names.noises:
        return
    for group, (weight_path, noise_path) in enumerate(
        zip(plan.names.weights, plan.names.noises, strict=True)
    ):
        image_path = plan.names.images[plan.groups.index(group)]
        with fits.open(weight_path, memmap=True) as weight_hdul:
            header = weight_hdul[0].header.copy()
            header["BUNIT"] = fits.getheader(image_path).get("BUNIT", "")
            create_output_fits(noise_path, header)
            if "BEAMS" in weight_hdul:
                fits.append(
                    noise_path, weight_hdul["BEAMS"].data, weight_hdul["BEAMS"].header
                )
        # Whole rows at a time, each one run of bytes, as the tiles are written
        weight_location, noise_location = (
            data_location(weight_path),
            data_location(noise_path),
        )
        n_rows = noise_location.shape[-2]
        for channel in range(noise_location.n_channels):
            for row in range(0, n_rows, strip_rows):
                rows = slice(row, min(row + strip_rows, n_rows))
                noise = noise_of(read_rows(weight_location, rows, channel))
                write_rows(
                    noise_location, rows, slice(channel, channel + 1), noise[None]
                )
        logger.info(f"Written {noise_path}")


def noise_of(weight: NDArray[np.floating]) -> NDArray[np.float32]:
    """The noise of a weight, the inverse of its square root, which is blank where there is no weight."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(weight > 0, 1 / np.sqrt(weight), np.nan).astype(np.float32)


def create_zarr_outputs(plan: MosaicPlan, chunk: int) -> None:
    """Create the zarr stores of a mosaic, laid out as rm-lite reads them.

    Each is one (channel, y, x) float32 array, chunked over every channel, with
    the FITS header of the cube and the restoring beam of each channel in its
    attributes.
    """
    import zarr  # noqa: PLC0415
    from zarr.codecs import BloscCodec  # noqa: PLC0415

    n_channels = plan.channels.stop - plan.channels.start
    for path, header in zip(
        (*plan.names.images, *plan.names.weights, *plan.names.noises),
        plan.headers,
        strict=True,
    ):
        array = zarr.create_array(
            store=str(path),
            shape=(n_channels, *plan.frame.shape),
            chunks=(
                n_channels,
                min(chunk, plan.frame.shape[0]),
                min(chunk, plan.frame.shape[1]),
            ),
            dtype="float32",
            fill_value=np.nan,
            compressors=(BloscCodec(cname="lz4", clevel=5, shuffle="shuffle"),),
            overwrite=True,
        )
        # The header holds one beam only if every channel shares it, as a FITS cube's does
        stored = header.copy()
        beams_header(stored, plan.beams)
        array.attrs["fits_header"] = stored.tostring()
        array.attrs["beams"] = [
            list(beam) if beam.is_usable() else None for beam in plan.beams
        ]


def mosaic_tile_to_zarr(
    tile: Slices,
    plan: MosaicPlan,
    options: MosaicOptions,
    channel_chunk: int,
) -> None:
    """Make a tile over every channel, a few channels at a time, and write it into the zarr stores of the mosaic.

    One task owns each chunk of the stores, so tasks never write the same chunk.
    """
    import zarr  # noqa: PLC0415

    start, stop = plan.channels.start, plan.channels.stop
    results = [
        mosaic_tile(
            tile,
            plan.frame,
            plan.inputs,
            options,
            slice(first, min(first + channel_chunk, stop)),
        )
        for first in range(start, stop, channel_chunk)
    ]
    for index, path in enumerate(plan.names.images):
        zarr.open_array(str(path), mode="r+")[(slice(None), *tile)] = np.concatenate(
            [result.images[index] for result in results]
        )
    for index, path in enumerate(plan.names.weights):
        zarr.open_array(str(path), mode="r+")[(slice(None), *tile)] = np.concatenate(
            [result.weights[index] for result in results]
        )


def noise_zarr_tile(tile: Slices, plan: MosaicPlan) -> None:
    """Write a tile of the noise stores of a mosaic from its weights."""
    import zarr  # noqa: PLC0415

    index = (slice(None), *tile)
    for weight_path, noise_path in zip(
        plan.names.weights, plan.names.noises, strict=True
    ):
        weight = np.asarray(zarr.open_array(str(weight_path), mode="r")[index])
        zarr.open_array(str(noise_path), mode="r+")[index] = noise_of(weight)


class FitsLocations(NamedTuple):
    """Where the data of each FITS output of a mosaic lies, for tasks to write their strips into"""

    images: tuple[DataLocation, ...]
    """The data of each image"""
    weights: tuple[DataLocation, ...]
    """The data of each weight"""
    noises: tuple[DataLocation, ...]
    """The data of each noise image"""


def fits_locations(plan: MosaicPlan) -> FitsLocations:
    """Find where the data of each created FITS output of a mosaic lies."""
    names = plan.names
    return FitsLocations(
        images=tuple(map(data_location, names.images)),
        weights=tuple(map(data_location, names.weights)),
        noises=tuple(map(data_location, names.noises)),
    )


def mosaic_strip_to_fits(
    plan: MosaicPlan,
    strip: Slices,
    options: MosaicOptions,
    channels: slice,
    locations: FitsLocations,
) -> None:
    """Make a strip of whole rows over some channels, and write it into the FITS outputs of the mosaic.

    Each task writes only its own rows, so no two tasks write the same bytes.
    """
    result = mosaic_tile(strip, plan.frame, plan.inputs, options, channels)
    written = _output_channels(plan, result.channels)
    files = (*locations.images, *locations.weights)
    blocks = (*result.images, *result.weights)

    def write(index: int) -> None:
        write_rows(files[index], strip[0], written, blocks[index])

    threads = 1
    if resolve_backend(options.resample.backend) == "fused":
        threads = min(tile_threads(options.resample.tile_threads), len(files))
    started = time.perf_counter()
    if threads == 1:
        for index in range(len(files)):
            write(index)
    else:
        # A tile made on several threads would otherwise wait for one to write it
        with ThreadPoolExecutor(threads, thread_name_prefix="patchwork-write") as pool:
            list(pool.map(write, range(len(files))))
    if logger.isEnabledFor(logging.DEBUG):
        written_bytes = sum(block.nbytes for block in blocks)
        logger.debug(
            f"Wrote rows {strip[0].start}-{strip[0].stop}, channels {written.start}-{written.stop} "
            f"({written_bytes / 2**20:.0f} MB) in {time.perf_counter() - started:.1f} s on {threads} threads"
        )


def noise_fits_strip(strip: Slices, locations: FitsLocations) -> None:
    """Write a strip of the FITS noise outputs of a mosaic from its weights."""
    rows = strip[0]
    for weight_location, noise_location in zip(
        locations.weights, locations.noises, strict=True
    ):
        for channel in range(weight_location.n_channels):
            noise = noise_of(read_rows(weight_location, rows, channel))
            write_rows(noise_location, rows, slice(channel, channel + 1), noise[None])


Call = tuple["Callable[..., Any]", tuple[Any, ...]]
"""A function and the arguments to call it with"""

Result = TypeVar("Result")


def _duration(seconds: float) -> str:
    """A duration such as ``42s``, ``3m 05s`` or ``1h 02m``."""
    minutes, whole_seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {whole_seconds:02d}s"
    return f"{whole_seconds}s"


def logged_progress(
    results: Iterable[Result], total: int, what: str, interval_s: float = 30.0
) -> Generator[Result, None, None]:
    """Pass results on as they arrive, logging how many of ``total`` are done every ``interval_s`` seconds, and once all are.

    A result is counted once it has been taken, so for tiles written as they
    arrive the count is of tiles made and written.

    Args:
        results (Iterable[Result]): The results, such as those of ``bounded_results``
        total (int): How many results there are
        what (str): What the results are, such as ``tiles``
        interval_s (float, optional): The least time in seconds between messages. Defaults to 30.0.

    Yields:
        Result: Each result, unchanged
    """
    start = logged = time.monotonic()
    done = 0
    for result in results:
        yield result
        done += 1
        now = time.monotonic()
        if done < total and now - logged < interval_s:
            continue
        logged, elapsed = now, now - start
        message = f"Made {done} of {total} {what} ({done / total:.0%}) in {_duration(elapsed)}"
        if done < total:
            message += f", about {_duration(elapsed / done * (total - done))} left"
        logger.info(message)


def bounded_results(
    executor: Executor,
    calls: Iterable[Call],
    limit: Callable[[], int],
    refresh_s: float = 5.0,
) -> Generator[Any, None, None]:
    """The results of calls made by an executor, as they finish, with no more than ``limit()`` submitted and not yet taken.

    The next call is submitted only once a result is taken. An executor that
    hands results back as soon as they finish, as dask's does, therefore never
    holds more than ``limit()`` of them in this process, however slowly they are
    taken. Should a call fail, or the results stop being taken, the calls still
    waiting are cancelled.

    Args:
        executor (Executor): Makes the calls
        calls (Iterable[Call]): The functions to call, and their arguments
        limit (Callable[[], int]): The most calls submitted and not yet taken. It is asked again every ``refresh_s`` seconds, so it can follow a cluster as it scales.
        refresh_s (float, optional): Seconds between asking ``limit`` again. Defaults to 5.0.

    Yields:
        Any: The result of each call, in the order they finish
    """
    remaining = iter(calls)
    pending: set[Future[Any]] = set()
    size, asked = max(limit(), 1), time.monotonic()
    try:
        while True:
            if time.monotonic() - asked > refresh_s:
                size, asked = max(limit(), 1), time.monotonic()
            while len(pending) < size and (call := next(remaining, None)) is not None:
                function, args = call
                pending.add(executor.submit(function, *args))
            if not pending:
                return
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
    finally:
        for future in pending:
            future.cancel()


def run_tiles(
    plan: MosaicPlan,
    options: MosaicOptions,
    run_options: RunOptions,
    executor: Executor,
    share: Callable[[Any], Any] | None = None,
    capacity: Callable[[], int] | None = None,
) -> MosaicNames:
    """Make the tiles of a planned mosaic with an executor, write them, and finish the mosaic.

    FITS files are written by this process as tiles arrive. Zarr stores are
    written by the tasks, one chunk each. Each tile is finished once written,
    so only the noise of the mosaic is written after them. No more than
    ``tasks_in_flight`` tiles are made at once, so this process holds no more
    than that many.

    Args:
        plan (MosaicPlan): The plan, from ``patchwork.plan.plan_mosaic``, whose outputs are created
        options (MosaicOptions): The mosaicking options
        run_options (RunOptions): How the mosaic is made and written
        executor (Executor): Runs the tasks, such as a thread pool, or a dask ``Client.get_executor()``
        share (Callable[[Any], Any] | None, optional): Turns the plan into what is sent to each task, such as dask's ``scatter``. Defaults to None, sending it as it is.
        capacity (Callable[[], int] | None, optional): How many tasks the executor runs at once, asked again as tiles are made, such as the threads of a dask cluster as it scales. Without ``tasks_in_flight`` twice this many tiles are made at once. Defaults to None, one.

    Returns:
        MosaicNames: The files the mosaic was written to
    """
    shared = plan if share is None else share(plan)
    # Only a FITS mosaic written by this process holds the finished tiles, so
    # only it needs a bound; the others queue every task, so an adaptive dask
    # cluster sees the whole mosaic waiting and grows to its maximum at once
    held_here = (
        run_options.output_format == "fits" and run_options.fits_writer != "workers"
    )

    def in_flight() -> int:
        if run_options.tasks_in_flight is not None:
            return run_options.tasks_in_flight
        if not held_here:
            return sys.maxsize
        return 2 * max(1 if capacity is None else capacity(), 1)

    def results(calls: Iterable[Call]) -> closing[Generator[Any, None, None]]:
        return closing(bounded_results(executor, calls, in_flight))

    chunks = [
        slice(first, min(first + run_options.channel_chunk, plan.channels.stop))
        for first in range(
            plan.channels.start, plan.channels.stop, run_options.channel_chunk
        )
    ]
    if run_options.output_format == "fits" and run_options.fits_writer == "workers":
        _run_strips(plan, options, shared, chunks, results)
        return plan.names
    if run_options.output_format == "fits":
        logger.info(
            f"Making {len(plan.tiles)} tiles in each of {len(chunks)} chunks of channels, "
            f"{in_flight()} at once"
        )
        with results(
            (_plan_tile, (shared, tile, options, chunk))
            for chunk in chunks
            for tile in plan.tiles
        ) as tiles_made:
            write_tiles(
                plan,
                logged_progress(tiles_made, len(plan.tiles) * len(chunks), "tiles"),
                buffer_bytes=run_options.fits_buffer_mb * 2**20,
            )
        write_noise(plan)
        return plan.names

    logger.info(
        f"Making {len(plan.tiles)} zarr chunks"
        + (
            f", {in_flight()} at once"
            if run_options.tasks_in_flight
            else ", all queued"
        )
    )
    with results(
        (mosaic_tile_to_zarr, (tile, shared, options, run_options.channel_chunk))
        for tile in plan.tiles
    ) as chunks_made:
        list(logged_progress(chunks_made, len(plan.tiles), "zarr chunks"))
    if plan.names.noises:
        with results((noise_zarr_tile, (tile, shared)) for tile in plan.tiles) as noise:
            list(noise)
    return plan.names


def _run_strips(
    plan: MosaicPlan,
    options: MosaicOptions,
    shared: Any,
    chunks: list[slice],
    results: Callable[[Iterable[Call]], closing[Generator[Any, None, None]]],
) -> None:
    """Make the strips of a planned FITS mosaic, each written by its task, then write their noise."""
    locations = fits_locations(plan)
    logger.info(
        f"Making {len(plan.tiles)} strips in each of {len(chunks)} chunks of channels, "
        "each written by its task"
    )
    with results(
        (mosaic_strip_to_fits, (shared, strip, options, chunk, locations))
        for chunk in chunks
        for strip in plan.tiles
    ) as made:
        list(logged_progress(made, len(plan.tiles) * len(chunks), "strips"))
    if plan.names.noises:
        with results(
            (noise_fits_strip, (strip, locations)) for strip in plan.tiles
        ) as noise:
            list(noise)


def _plan_tile(
    plan: MosaicPlan, tile: Slices, options: MosaicOptions, channels: slice
) -> TileResult:
    """Make one tile of a planned mosaic, over some channels."""
    return mosaic_tile(tile, plan.frame, plan.inputs, options, channels)

"""Numba kernels of the fused backend, which make a tile in one compiled pass over each input.

``add_block`` does to the pixels of an input everything that
``patchwork.tile._add_input`` does with numpy: it interpolates the input's
images, evaluates the primary beam, removes the leakage, rotates Q and U,
weights the images and adds them to the sums of the tile. It works a row of
one channel at a time, so nothing is held for the whole region between those
steps, and none of them holds the GIL. ``finalise_rows`` turns the sums into
the tile.

Each has a twin that runs it over many parts of the tile on numba's threads
(``add_chunks`` and ``finalise_chunks``). The parts share no pixels, so the
threads never write the same memory. ``add_input`` and ``finalise`` choose
between the two.

They follow the numpy code, and give the same values to round-off.
"""

from __future__ import annotations

import functools
import itertools
import threading
from typing import TYPE_CHECKING, NamedTuple

import numba
import numpy as np

from patchwork.logging import logger
from patchwork.numba_kernels import _kernel, _taps

if TYPE_CHECKING:
    from numpy.typing import NDArray

WEIGHTED, AVERAGE, SUM = 0, 1, 2
"""How the inputs are combined, as ``CombineOptions.combine_type``"""
BY_PRIMARY_BEAM, BY_WEIGHT_IMAGE, BY_BOTH = 0, 1, 2
"""Where the weights come from, as ``WeightOptions.weight_type``"""
CORRECTED, INHERENT, ALREADY_WEIGHTED = 0, 1, 2
"""What has been done to the inputs, as ``WeightOptions.weight_state``"""
NO_BEAM, GAUSSIAN, HOLOGRAPHY = 0, 1, 2
"""The primary beam that weights an input"""
STOKES_I, STOKES_V = 1, 4
"""FITS codes of Stokes parameters"""

_PARALLEL = threading.Lock()
"""Held while numba's threads run. Its default threading layer (workqueue)
stops the process if two Python threads start parallel kernels at once, so
Python threads that make tiles at the same time take turns at them."""


class Sums(NamedTuple):
    """The running sums of a tile over its inputs, as ``patchwork.combine.TileSums`` keeps for each group, stacked"""

    totals: NDArray[np.float64]
    """For each image, the sum of its weighted estimates, shaped (image, channel, y, x)"""
    weight: NDArray[np.float64]
    """For each group of images that share a weight, the sum of the weights, shaped (group, channel, y, x)"""
    inverse_weight: NDArray[np.float64]
    """For each group, the sum of the inverse weights, kept for a plain mean or a sum, and otherwise empty"""
    n_inputs: NDArray[np.float64]
    """For each group, the number of inputs that contributed, kept for a plain mean or a sum, and otherwise empty"""


class Task(NamedTuple):
    """An input over the region of a tile it contributes to, and how it is added to the tile.

    Every field has one type whatever the input, so the kernels are compiled once.
    """

    blocks: NDArray[np.float32]
    """The part of each image of the input that is interpolated, shaped (image, channel, y, x)"""
    x: NDArray[np.float64]
    """The x position in the blocks of each pixel of the region"""
    y: NDArray[np.float64]
    """The y position in the blocks of each pixel of the region"""
    code: int
    """The interpolation kernel: 0 nearest, 1 linear, 2 cubic, 3 Lanczos"""
    width: int
    """The number of taps of the kernel along each axis"""
    usable: NDArray[np.bool_]
    """Which channels of the input hold an image"""
    plane_block: NDArray[np.int64]
    """The block of each image of the mosaic"""
    plane_stokes: NDArray[np.int64]
    """The FITS Stokes code of each image of the mosaic"""
    plane_group: NDArray[np.int64]
    """The group, of images that share a weight, of each image of the mosaic"""
    weights: NDArray[np.float64]
    """The weight of the input in each group and channel, shaped (group, channel)"""
    stokes_i: int
    """The block of Stokes I, whose leakage is removed from the other images"""
    removes_leakage: bool
    """Whether the leakage of Stokes I is removed"""
    q: int
    """The image of Stokes Q, which is rotated with ``u``, or -1 if the input does not have both"""
    u: int
    """The image of Stokes U"""
    row: int
    """The first row of the region in the tile"""
    column: int
    """The first column of the region in the tile"""
    vectors: NDArray[np.float64]
    """Unit vectors of every pixel of the tile, shaped (3, y, x)"""
    east: NDArray[np.float64]
    """The unit vector pointing east at the beam centre"""
    north: NDArray[np.float64]
    """The unit vector pointing north at the beam centre"""
    beam: int
    """The primary beam that weights the input: ``NO_BEAM``, ``GAUSSIAN`` or ``HOLOGRAPHY``"""
    fwhm: NDArray[np.float64]
    """The FWHM in radians of the Gaussian beam of each channel"""
    cutoff: NDArray[np.float64]
    """The response of each channel below which the input does not contribute"""
    pattern: NDArray[np.float64]
    """The holography beam of each channel, shaped (channel, y, x)"""
    leakage: NDArray[np.float64]
    """The fractional leakage of Stokes I into (Q, U, V) of each channel, shaped (channel, 3, y, x)"""
    pattern_scale: NDArray[np.int64]
    """Which of ``scales`` each channel uses"""
    scales: NDArray[np.float64]
    """The distinct frequency scales of the holography beams, as ``HolographyBeam.scale``"""
    pattern_turn: NDArray[np.float64]
    """The cosine and sine of the angle from sky offsets to the axes of the holography grid"""
    pattern_pixel: NDArray[np.float64]
    """The signed pixel size of the holography grid along x and y, in radians"""
    pattern_centre: NDArray[np.float64]
    """The (x, y) pixel of the beam centre on the holography grid"""
    feed_turn: NDArray[np.float64]
    """The cosine and sine of twice the feed rotation, which turns the leakage of Q and U"""
    input_north: NDArray[np.float64]
    """The unit vector that is north where the Q and U of the input are referenced to"""
    has_input_frame: bool
    """Whether the Q and U of the input are referenced to ``input_north``, rather than to north at each pixel"""
    output_north: NDArray[np.float64]
    """The unit vector that is north where the Q and U of the mosaic are referenced to"""
    has_output_frame: bool
    """Whether the Q and U of the mosaic are referenced to ``output_north``, rather than to north at each pixel"""
    weight_type: int
    """Where the weights come from"""
    weight_state: int
    """What has been done to the input"""
    combine: int
    """How the inputs are combined"""


_fused = {"error_model": "numpy"}
"""Division follows IEEE arithmetic, as numpy's does, rather than raising"""


def _resample_row(
    window: NDArray[np.float32],
    wanted: NDArray[np.bool_],
    inside: NDArray[np.bool_],
    x_first: NDArray[np.int64],
    y_first: NDArray[np.int64],
    x_weights: NDArray[np.float64],
    y_weights: NDArray[np.float64],
    width: int,
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> None:
    """Interpolate one image at the wanted positions of a row, whose taps are already found, as ``numba_kernels.interpolate_stack`` does."""
    n_rows, n_columns = window.shape
    for index in range(values.shape[0]):
        if not wanted[index]:
            continue
        left, top = x_first[index], y_first[index]
        if inside[index]:
            # Every tap is on the image. Each row of taps is summed, then the rows,
            # which takes fewer multiplications than summing tap by tap and gives
            # the same value to round-off. If no tap is blank the sum is finite,
            # and nothing else needs checking
            total = 0.0
            for j in range(width):
                along = 0.0
                for i in range(width):
                    along += x_weights[index, i] * window[top + j, left + i]
                total += y_weights[index, j] * along
            if np.isfinite(total):
                values[index] = total
                valid[index] = True
                continue

        total = 0.0
        good = True
        for j in range(width):
            row = top + j
            row_outside = row < 0 or row >= n_rows
            row = min(max(row, 0), n_rows - 1)
            for i in range(width):
                column = left + i
                outside = row_outside or column < 0 or column >= n_columns
                column = min(max(column, 0), n_columns - 1)
                tap_weight = y_weights[index, j] * x_weights[index, i]
                value = window[row, column]
                blank = not np.isfinite(value)
                if abs(tap_weight) > 1e-9 and (outside or blank):
                    good = False
                total += tap_weight * (0.0 if blank else value)
        values[index] = total if good else 0.0
        valid[index] = good


resample_row = _kernel(_resample_row, **_fused)


def _pattern_taps(
    east: float,
    north: float,
    scale: float,
    turn: NDArray[np.float64],
    pixel: NDArray[np.float64],
    centre: NDArray[np.float64],
    n_rows: int,
    n_columns: int,
    weights: NDArray[np.float64],
) -> tuple[int, int]:
    """Where an offset from the beam centre falls on a holography grid: the lower pixel around it, with the weights of the four around it put in ``weights``.

    As ``HolographyBeam.taps``, which gives no weight to a position off the grid.
    """
    x = east * turn[0] - north * turn[1]
    y = east * turn[1] + north * turn[0]
    radius = np.hypot(x, y)
    offset = np.arcsin(min(radius, 1.0))
    stretch = np.sin(scale * offset) / radius if radius > 0 else scale
    x_pixel = x * stretch / pixel[0] + centre[0]
    y_pixel = y * stretch / pixel[1] + centre[1]
    on_grid = (
        y_pixel >= 0
        and y_pixel <= n_rows - 1
        and x_pixel >= 0
        and x_pixel <= n_columns - 1
    )
    if not on_grid:
        weights[:] = 0.0
        return 0, 0
    row = int(min(np.floor(y_pixel), max(n_rows - 2, 0)))
    column = int(min(np.floor(x_pixel), max(n_columns - 2, 0)))
    y_fraction, x_fraction = y_pixel - row, x_pixel - column
    weights[0] = (1 - y_fraction) * (1 - x_fraction)
    weights[1] = (1 - y_fraction) * x_fraction
    weights[2] = y_fraction * (1 - x_fraction)
    weights[3] = y_fraction * x_fraction
    return row, column


pattern_taps = _kernel(_pattern_taps, **_fused)


def _sample(
    plane: NDArray[np.float64], row: int, column: int, weights: NDArray[np.float64]
) -> float:
    """Interpolate a plane of a holography grid with the taps of a position, as ``PatternTaps.sample``."""
    row_step = min(1, plane.shape[0] - 1)
    column_step = min(1, plane.shape[1] - 1)
    value = weights[0] * plane[row, column]
    value += weights[1] * plane[row, column + column_step]
    value += weights[2] * plane[row + row_step, column]
    value += weights[3] * plane[row + row_step, column + column_step]
    return float(value)


sample = _kernel(_sample, **_fused)


def _frame_angle(north: NDArray[np.float64], x: float, y: float, z: float) -> float:
    """The position angle at a position of the direction that is north where ``north`` is, as ``patchwork.polarisation.frame_angle``."""
    along_east = -north[0] * y + north[1] * x
    along_north = -z * (north[0] * x + north[1] * y) + north[2] * (x**2 + y**2)
    return float(np.arctan2(along_east, along_north))


frame_angle = _kernel(_frame_angle, **_fused)


def _add_block(
    first_row: int,
    last_row: int,
    first_column: int,
    last_column: int,
    task: Task,
    sums: Sums,
) -> None:
    """Add an input to the sums of a tile, over a block of rows and columns of the region it contributes to.

    Blocks that share no pixels may be added at the same time.
    """
    blocks = task.blocks
    n_blocks, n_channels = blocks.shape[0], blocks.shape[1]
    window_rows, window_columns = blocks.shape[2], blocks.shape[3]
    width = task.width
    n_columns = last_column - first_column
    n_planes = task.plane_block.shape[0]
    n_groups = task.weights.shape[0]
    n_scales = task.scales.shape[0]
    samples_pattern = task.beam == HOLOGRAPHY or task.removes_leakage
    needs_offsets = task.beam != NO_BEAM or task.removes_leakage
    turns = task.q >= 0 and (task.has_input_frame or task.has_output_frame)
    has_v = False
    for plane in range(n_planes):
        if task.plane_stokes[plane] == STOKES_V:
            has_v = True

    shifts = np.pi * (np.arange(width) - (width // 2 - 1)) / max(width // 2, 1)
    shift_trig = np.vstack((np.cos(shifts), np.sin(shifts)))
    # What is known of each pixel of a row, whatever the channel
    located = np.zeros(n_columns, dtype=np.bool_)
    inside = np.zeros(n_columns, dtype=np.bool_)
    x_first = np.zeros(n_columns, dtype=np.int64)
    y_first = np.zeros(n_columns, dtype=np.int64)
    x_weights = np.zeros((n_columns, width))
    y_weights = np.zeros((n_columns, width))
    east = np.zeros(n_columns)
    north = np.zeros(n_columns)
    offset_squared = np.zeros(n_columns)
    cos_turn = np.ones(n_columns)
    sin_turn = np.zeros(n_columns)
    pattern_row = np.zeros((n_scales, n_columns), dtype=np.int64)
    pattern_column = np.zeros((n_scales, n_columns), dtype=np.int64)
    pattern_weights = np.zeros((n_scales, n_columns, 4))
    # A row of one channel: where the input reaches, its response there and its
    # images, and then the images of one of its pixels
    wanted = np.zeros(n_columns, dtype=np.bool_)
    response = np.ones(n_columns)
    values = np.zeros((n_blocks, n_columns))
    valid = np.zeros((n_blocks, n_columns), dtype=np.bool_)
    plane_values = np.zeros(n_planes)
    plane_valid = np.zeros(n_planes, dtype=np.bool_)
    leak = np.zeros(3)

    for row in range(first_row, last_row):
        tile_row = task.row + row
        tile_column = task.column + first_column
        for index in range(n_columns):
            x_position = task.x[row, first_column + index]
            y_position = task.y[row, first_column + index]
            located[index] = np.isfinite(x_position) and np.isfinite(y_position)
            if not located[index]:
                continue
            left = _taps(x_position, task.code, width, x_weights[index], shift_trig)
            top = _taps(y_position, task.code, width, y_weights[index], shift_trig)
            x_first[index], y_first[index] = left, top
            inside[index] = (
                left >= 0
                and left + width <= window_columns
                and top >= 0
                and top + width <= window_rows
            )

        if needs_offsets or turns:
            for index in range(n_columns):
                x = task.vectors[0, tile_row, tile_column + index]
                y = task.vectors[1, tile_row, tile_column + index]
                z = task.vectors[2, tile_row, tile_column + index]
                if needs_offsets:
                    east[index] = task.east[0] * x + task.east[1] * y + task.east[2] * z
                    north[index] = (
                        task.north[0] * x + task.north[1] * y + task.north[2] * z
                    )
                if turns:
                    turn = 0.0
                    if task.has_input_frame:
                        turn = frame_angle(task.input_north, x, y, z)
                    if task.has_output_frame:
                        turn = turn - frame_angle(task.output_north, x, y, z)
                    cos_turn[index] = np.cos(2 * turn)
                    sin_turn[index] = np.sin(2 * turn)
        if task.beam == GAUSSIAN:
            for index in range(n_columns):
                offset = np.arcsin(min(np.hypot(east[index], north[index]), 1.0))
                offset_squared[index] = offset**2
        if samples_pattern:
            for scale in range(n_scales):
                for index in range(n_columns):
                    pattern_row[scale, index], pattern_column[scale, index] = (
                        pattern_taps(
                            east[index],
                            north[index],
                            task.scales[scale],
                            task.pattern_turn,
                            task.pattern_pixel,
                            task.pattern_centre,
                            task.pattern.shape[1],
                            task.pattern.shape[2],
                            pattern_weights[scale, index],
                        )
                    )

        for channel in range(n_channels):
            if not task.usable[channel]:
                continue
            # The beam first, so nothing is interpolated where it is below the cutoff
            scale = task.pattern_scale[channel] if samples_pattern else 0
            any_wanted = False
            for index in range(n_columns):
                wanted[index] = located[index]
                if wanted[index] and task.beam != NO_BEAM:
                    if task.beam == GAUSSIAN:
                        response[index] = np.exp(
                            -4
                            * np.log(2)
                            * offset_squared[index]
                            / task.fwhm[channel] ** 2
                        )
                    else:
                        response[index] = sample(
                            task.pattern[channel],
                            pattern_row[scale, index],
                            pattern_column[scale, index],
                            pattern_weights[scale, index],
                        )
                    wanted[index] = response[index] >= task.cutoff[channel]
                any_wanted = any_wanted or wanted[index]
            if not any_wanted:
                continue
            for block in range(n_blocks):
                resample_row(
                    blocks[block, channel],
                    wanted,
                    inside,
                    x_first,
                    y_first,
                    x_weights,
                    y_weights,
                    width,
                    values[block],
                    valid[block],
                )

            for index in range(n_columns):
                if not wanted[index]:
                    continue

                for plane in range(n_planes):
                    source = task.plane_block[plane]
                    plane_values[plane] = values[source, index]
                    plane_valid[plane] = valid[source, index]

                if task.removes_leakage:
                    stokes_i = values[task.stokes_i, index]
                    stokes_i_valid = valid[task.stokes_i, index]
                    for stokes in range(3 if has_v else 2):
                        leak[stokes] = sample(
                            task.leakage[channel, stokes],
                            pattern_row[scale, index],
                            pattern_column[scale, index],
                            pattern_weights[scale, index],
                        )
                    q_leak, u_leak = leak[0], leak[1]
                    leak[0] = q_leak * task.feed_turn[0] - u_leak * task.feed_turn[1]
                    leak[1] = q_leak * task.feed_turn[1] + u_leak * task.feed_turn[0]
                    for plane in range(n_planes):
                        code = task.plane_stokes[plane]
                        if code != STOKES_I:
                            plane_values[plane] -= leak[code - 2] * stokes_i
                            plane_valid[plane] = plane_valid[plane] and stokes_i_valid

                if task.q >= 0:
                    q, u = plane_values[task.q], plane_values[task.u]
                    plane_values[task.q] = q * cos_turn[index] - u * sin_turn[index]
                    plane_values[task.u] = q * sin_turn[index] + u * cos_turn[index]
                    both = plane_valid[task.q] and plane_valid[task.u]
                    plane_valid[task.q] = plane_valid[task.u] = both

                for group in range(n_groups):
                    contributes = True
                    for plane in range(n_planes):
                        if task.plane_group[plane] == group and not plane_valid[plane]:
                            contributes = False
                    if not contributes:
                        continue

                    # As patchwork.weights.weighting
                    weight = task.weights[group, channel]
                    if task.weight_type == BY_WEIGHT_IMAGE:
                        if task.weight_state == CORRECTED:
                            factor = weight
                        elif task.weight_state == INHERENT:
                            factor = np.sqrt(weight) if weight > 0 else 0.0
                        else:
                            factor = 1.0
                    else:
                        scaling = weight if task.weight_type == BY_BOTH else 1.0
                        factor = weight = scaling
                        if task.beam != NO_BEAM:
                            if task.weight_state != ALREADY_WEIGHTED:
                                factor = scaling * response[index]
                            if task.weight_state == CORRECTED:
                                factor = factor * response[index]
                            weight = scaling * response[index] ** 2
                    if not (np.isfinite(weight) and weight > 0):
                        continue

                    # As patchwork.combine.accumulate
                    column = tile_column + index
                    if task.combine != WEIGHTED:
                        inverse = 1.0 / weight
                        factor = factor * inverse
                        sums.inverse_weight[group, channel, tile_row, column] += inverse
                        sums.n_inputs[group, channel, tile_row, column] += 1.0
                    sums.weight[group, channel, tile_row, column] += weight
                    for plane in range(n_planes):
                        if task.plane_group[plane] == group:
                            sums.totals[plane, channel, tile_row, column] += (
                                plane_values[plane] * factor
                            )


add_block = _kernel(_add_block, **_fused)


def _add_chunks(chunks: NDArray[np.int64], task: Task, sums: Sums) -> None:
    """Add an input to the sums of a tile over many blocks, the rows of ``chunks``, on numba's threads."""
    for index in numba.prange(chunks.shape[0]):
        add_block(
            chunks[index, 0],
            chunks[index, 1],
            chunks[index, 2],
            chunks[index, 3],
            task,
            sums,
        )


add_chunks = _kernel(_add_chunks, parallel=True, **_fused)


def _finalise_rows(
    first: int,
    last: int,
    sums: Sums,
    plane_group: NDArray[np.int64],
    combine: int,
    images: NDArray[np.float32],
    weights: NDArray[np.float32],
) -> None:
    """Turn the sums of some rows of a tile into its images and weights, as ``patchwork.combine.finalise``.

    The rows of every channel are counted together, so row ``index`` is row
    ``index % n_rows`` of channel ``index // n_rows``.
    """
    n_planes, n_groups = images.shape[0], weights.shape[0]
    n_rows, n_columns = images.shape[2], images.shape[3]
    for index in range(first, last):
        channel, row = index // n_rows, index % n_rows
        for group in range(n_groups):
            for column in range(n_columns):
                weight = sums.weight[group, channel, row, column]
                divisor = weight
                if combine == WEIGHTED:
                    covered = weight > 0
                else:
                    count = sums.n_inputs[group, channel, row, column]
                    inverse = sums.inverse_weight[group, channel, row, column]
                    covered = count > 0
                    if combine == AVERAGE:
                        divisor = count
                        weight = count**2 / inverse
                    else:
                        divisor = 1.0
                        weight = 1.0 / inverse
                weights[group, channel, row, column] = weight if covered else np.nan
                for plane in range(n_planes):
                    if plane_group[plane] == group:
                        images[plane, channel, row, column] = (
                            sums.totals[plane, channel, row, column] / divisor
                            if covered
                            else np.nan
                        )


finalise_rows = _kernel(_finalise_rows, **_fused)


def _finalise_chunks(
    chunks: NDArray[np.int64],
    sums: Sums,
    plane_group: NDArray[np.int64],
    combine: int,
    images: NDArray[np.float32],
    weights: NDArray[np.float32],
) -> None:
    """Turn the sums of a tile into its images and weights over many runs of rows, the rows of ``chunks``, on numba's threads."""
    for index in numba.prange(chunks.shape[0]):
        finalise_rows(
            chunks[index, 0],
            chunks[index, 1],
            sums,
            plane_group,
            combine,
            images,
            weights,
        )


finalise_chunks = _kernel(_finalise_chunks, parallel=True, **_fused)


def spread(chunks: NDArray[np.int64], threads: int) -> NDArray[np.int64]:
    """Reorder chunks so that each thread's share is spread over all of them.

    numba gives each thread a run of consecutive chunks. Neighbouring chunks
    take about as long as each other, and those far apart may not (an input
    covers one end of a tile and not the other), so every ``threads``'th chunk
    is put in one run.
    """
    order = np.argsort(np.arange(len(chunks)) % threads, kind="stable")
    return np.ascontiguousarray(chunks[order])


def block_chunks(
    n_rows: int, n_columns: int, threads: int, widest: int = 1024
) -> NDArray[np.int64]:
    """Split a region into blocks for ``threads`` threads to share, as rows of (first row, last row, first column, last column).

    There are about four blocks to each thread, so a thread whose blocks are
    quick is not left with nothing to do for long, and none is wider than
    ``widest`` pixels, so what is kept for a row of one stays in the cache. A
    strip of whole rows, which may be only a few rows of a wide mosaic, is
    therefore split along its rows too.
    """
    edges = np.linspace(0, n_columns, -(-n_columns // widest) + 1).astype(np.int64)
    rows = max(1, n_rows * (len(edges) - 1) // (4 * threads))
    chunks = np.array(
        [
            (row, min(row + rows, n_rows), left, right)
            for row in range(0, n_rows, rows)
            for left, right in itertools.pairwise(edges)
        ],
        dtype=np.int64,
    ).reshape(-1, 4)
    return spread(chunks, threads)


def row_chunks(n_rows: int, threads: int) -> NDArray[np.int64]:
    """Split rows into runs for ``threads`` threads to share, as rows of (first, last), about four to each thread."""
    rows = max(1, n_rows // (4 * threads))
    chunks = np.array(
        [(row, min(row + rows, n_rows)) for row in range(0, n_rows, rows)],
        dtype=np.int64,
    ).reshape(-1, 2)
    return spread(chunks, threads)


def numba_threads() -> int:
    """The most threads numba runs a kernel on: as many as the CPUs the process could run on when numba was imported, or ``NUMBA_NUM_THREADS``."""
    return int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined,unused-ignore]


@functools.cache
def _log_threads(threads: int) -> None:
    """Say, once in each process, how many threads tiles are made with, and what runs them."""
    layer = numba.threading_layer()  # type: ignore[attr-defined,unused-ignore]
    logger.info(
        f"Making tiles with {threads} threads of numba's {layer} threading layer, "
        f"which has {numba_threads()}"
    )


def add_input(task: Task, sums: Sums, threads: int = 1) -> None:
    """Add an input to the sums of a tile, on ``threads`` threads.

    One thread runs in the calling thread, as any number of Python threads may
    at once. More run on numba's threads, which one Python thread uses at a time.
    """
    n_rows, n_columns = task.x.shape
    if threads <= 1:
        add_block(0, n_rows, 0, n_columns, task, sums)
        return
    chunks = block_chunks(n_rows, n_columns, threads)
    with _PARALLEL:
        numba.set_num_threads(threads)
        add_chunks(chunks, task, sums)
    _log_threads(threads)


def finalise(
    sums: Sums,
    plane_group: NDArray[np.int64],
    combine: int,
    images: NDArray[np.float32],
    weights: NDArray[np.float32],
    threads: int = 1,
) -> None:
    """Turn the sums of a tile into its images and weights, on ``threads`` threads, as ``add_input`` uses them."""
    n_rows = images.shape[1] * images.shape[2]
    if threads <= 1:
        finalise_rows(0, n_rows, sums, plane_group, combine, images, weights)
        return
    chunks = row_chunks(n_rows, threads)
    with _PARALLEL:
        numba.set_num_threads(threads)
        finalise_chunks(chunks, sums, plane_group, combine, images, weights)

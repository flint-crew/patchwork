"""Weights of the inputs, and how they are applied to the images"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy.stats import mad_std

from patchwork.io import (
    read_block,
    read_image_info,
    read_plane,
    read_rows,
    readable_location,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def image_noise_weight(
    data: NDArray[np.floating],
    mode: Literal["mad", "std"] = "mad",
    stride: int = 8,
) -> float:
    """The inverse variance of an image from the finite pixels of every ``stride``'th row, or 0 if it has none."""
    values = np.ravel(data[::stride])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    rms = float(mad_std(values) if mode == "mad" else np.std(values))
    return 1.0 / rms**2 if np.isfinite(rms) and rms > 0 else 0.0


def measure_image_weight(
    path: str | Path,
    mode: Literal["mad", "std", "none"] = "mad",
    stride: int = 8,
) -> float:
    """The inverse variance weight of an image file from every ``stride``'th row, or 1 when ``mode`` is ``none``."""
    if mode == "none":
        return 1.0
    rows = read_plane(path, (slice(None, None, stride), slice(None)))
    return image_noise_weight(data=rows, mode=mode, stride=1)


def combined_weight(weights: list[float]) -> float:
    """The weight of images that are combined with one weight, such as Q and U: the inverse of their mean variance."""
    if any(weight <= 0 for weight in weights):
        return 0.0
    return len(weights) / sum(1 / weight for weight in weights)


def combined_weights(weights: list[NDArray[np.floating]]) -> NDArray[np.float64]:
    """Combine weights element by element as ``combined_weight`` does, giving 0 wherever any is not positive."""
    stacked = np.asarray(weights, dtype=np.float64)
    usable = np.all(stacked > 0, axis=0)
    with np.errstate(divide="ignore"):
        variance = np.sum(1 / np.where(stacked > 0, stacked, 1.0), axis=0)
    return np.where(usable, len(weights) / variance, 0.0)


NOISE_READ_BYTES = 64 * 2**20
"""The most bytes of an image read at once to measure its noise"""


def _every_row(path: str | Path, channel: int, stride: int) -> NDArray[np.float64]:
    """Every ``stride``'th row of one channel of an image or cube, from the first, as (y, x).

    Plain float32 data are read whole, in runs of whole rows of up to
    ``NOISE_READ_BYTES``, and the rows taken from them, as on a file system
    such as Lustre a read of a few MB takes little longer than one of a single
    row. Other data are read through astropy.
    """
    location = readable_location(path)
    if location is None:
        block = read_block(
            path,
            region=(slice(None, None, stride), slice(None)),
            channels=slice(channel, channel + 1),
        )
        return np.asarray(block[0], dtype=np.float64)
    n_rows, n_columns = location.shape[-2:]
    # A whole number of strides, so each run starts on a row that is kept
    at_once = max(stride, NOISE_READ_BYTES // (4 * n_columns) // stride * stride)
    return np.concatenate(
        [
            np.asarray(
                read_rows(
                    location,
                    slice(first, min(first + at_once, n_rows)),
                    channel,
                    step=stride,
                ),
                dtype=np.float64,
            )
            for first in range(0, n_rows, at_once)
        ]
    )


def measure_weights(
    paths: Sequence[str | Path],
    mode: Literal["mad", "std", "none"] = "mad",
    stride: int = 8,
    channels: slice | None = None,
    threads: int = 1,
) -> list[tuple[float, ...]]:
    """The inverse variance weight of each channel of each of several images or cubes, from every ``stride``'th row, or 1 when ``mode`` is ``none``.

    Only the channels in ``channels`` are measured (by default every channel),
    and every other channel is given no weight. Each channel of each image is
    read and measured on its own, ``threads`` at once, so many single-channel
    images are measured as quickly as the channels of one cube.

    Args:
        paths (Sequence[str | Path]): The images or cubes
        mode (Literal["mad", "std", "none"], optional): How the noise is measured: the median absolute deviation, the standard deviation, or not at all. Defaults to "mad".
        stride (int, optional): Measure every ``stride``'th row. Defaults to 8.
        channels (slice | None, optional): The channels to measure. Defaults to None, every channel.
        threads (int, optional): The channels measured at once. Defaults to 1.

    Returns:
        list[tuple[float, ...]]: The weight of each channel of each image, in the order of ``paths``
    """
    counts = [read_image_info(path).n_channels for path in paths]
    if mode == "none":
        return [(1.0,) * count for count in counts]
    planes = [
        (index, channel)
        for index, count in enumerate(counts)
        for channel in range(*(channels or slice(None)).indices(count)[:2])
    ]

    def measure(plane: tuple[int, int]) -> float:
        index, channel = plane
        rows = _every_row(paths[index], channel, stride)
        return image_noise_weight(rows, mode=mode, stride=1)

    if threads > 1 and len(planes) > 1:
        with ThreadPoolExecutor(min(threads, len(planes))) as pool:
            measured = list(pool.map(measure, planes))
    else:
        measured = [measure(plane) for plane in planes]
    weights = [[0.0] * count for count in counts]
    for (index, channel), weight in zip(planes, measured, strict=True):
        weights[index][channel] = weight
    return [tuple(values) for values in weights]


def measure_channel_weights(
    path: str | Path,
    mode: Literal["mad", "std", "none"] = "mad",
    stride: int = 8,
    channel_chunk: int = 16,  # noqa: ARG001
    channels: slice | None = None,
    threads: int = 1,
) -> tuple[float, ...]:
    """The inverse variance weight of each channel of a cube, from every ``stride``'th row, or 1 when ``mode`` is ``none``.

    Only the channels in ``channels`` are measured (by default every channel),
    and every other channel is given no weight. The channels are measured as
    ``measure_weights`` does, ``threads`` at once. ``channel_chunk`` is no
    longer used, as each channel is read on its own, and is kept so that
    calls giving it still work.
    """
    return measure_weights(
        [path], mode=mode, stride=stride, channels=channels, threads=threads
    )[0]


def read_weights_log(path: str | Path) -> tuple[float, ...]:
    """The weight of each channel from a linmos weights log, a ``#Channel Weight`` header then a line per channel.

    Raises:
        ValueError: Raised if the log has no channels, or skips any
    """
    rows = [
        line.split()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    channels = {int(row[0]): float(row[1]) for row in rows}
    if not channels or sorted(channels) != list(range(len(channels))):
        msg = f"{path} should hold a weight for each channel from 0, but holds channels {sorted(channels)}"
        raise ValueError(msg)
    return tuple(channels[channel] for channel in range(len(channels)))


def weight_from_map(
    values: NDArray[np.floating],
    map_type: Literal["weight", "variance", "rms"] = "weight",
) -> NDArray[np.float64]:
    """Convert a weight, variance or rms map to weights, which are 0 wherever the map is unusable."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if map_type == "weight":
            weights = np.asarray(values, dtype=np.float64)
        elif map_type == "variance":
            weights = 1.0 / np.asarray(values, dtype=np.float64)
        else:
            weights = 1.0 / np.asarray(values, dtype=np.float64) ** 2
    return np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)


def weighting(
    primary_beam: NDArray[np.floating] | None,
    weight: float | NDArray[np.floating],
    weight_type: Literal["primary_beam", "weight_image", "combined"] = "combined",
    weight_state: Literal["corrected", "inherent", "weighted"] = "inherent",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The factor that weights an input's images, and the weight the input adds to the mosaic.

    An input adds ``image * factor`` to the sum of weighted images and ``weight``
    to the sum of weights, and the mosaic is their ratio. With an ``inherent``
    image (one still carrying its primary beam ``pb``) and ``combined`` weights
    ``w`` this is ``sum(w pb I) / sum(w pb^2)``. Images that share a weight share
    both, so they are found once for all of them.

    Args:
        primary_beam (NDArray[np.floating] | None): Primary beam response at each pixel, or None for a flat response
        weight (float | NDArray[np.floating]): Weight of the input, or of each of its pixels
        weight_type (Literal["primary_beam", "weight_image", "combined"], optional): Where the weights come from. Defaults to "combined".
        weight_state (Literal["corrected", "inherent", "weighted"], optional): Whether the images are corrected for the primary beam, still carry it, or are already weighted. Defaults to "inherent".

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The factor and the weight, which broadcast against the images
    """
    weights = np.asarray(weight, dtype=np.float64)
    if weight_type == "weight_image":
        if weight_state == "corrected":
            return weights, weights
        if weight_state == "inherent":
            return np.sqrt(np.clip(weights, 0, None)), weights
        return np.ones_like(weights), weights

    scale = weights if weight_type == "combined" else np.ones_like(weights)
    if primary_beam is None:
        return scale, scale
    pb = np.asarray(primary_beam, dtype=np.float64)
    factor = scale if weight_state == "weighted" else scale * pb
    if weight_state == "corrected":
        factor = factor * pb
    return factor, scale * pb**2

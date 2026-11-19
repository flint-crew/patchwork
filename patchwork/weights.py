"""Weights of the inputs, and how they are applied to the images"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy.stats import mad_std

from patchwork.io import read_block, read_image_info, read_plane

if TYPE_CHECKING:
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


def measure_channel_weights(
    path: str | Path,
    mode: Literal["mad", "std", "none"] = "mad",
    stride: int = 8,
    channel_chunk: int = 16,
    channels: slice | None = None,
    threads: int = 1,
) -> tuple[float, ...]:
    """The inverse variance weight of each channel of a cube, from every ``stride``'th row, or 1 when ``mode`` is ``none``.

    Only the channels in ``channels`` are measured (by default every channel),
    and every other channel is given no weight. The channels are read and
    measured ``channel_chunk`` at a time, in ``threads`` threads; with more
    than one, the chunks are made small enough to keep every thread busy.
    """
    n_channels = read_image_info(path).n_channels
    if mode == "none":
        return (1.0,) * n_channels
    start, stop, _ = (channels or slice(None)).indices(n_channels)
    weights = [0.0] * n_channels
    if threads > 1:
        channel_chunk = max(1, min(channel_chunk, -(-(stop - start) // threads)))
    chunks = [
        slice(first, min(first + channel_chunk, stop))
        for first in range(start, stop, channel_chunk)
    ]

    def measure(chunk: slice) -> list[float]:
        block = read_block(
            path, region=(slice(None, None, stride), slice(None)), channels=chunk
        )
        return [image_noise_weight(plane, mode=mode, stride=1) for plane in block]

    if threads > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(min(threads, len(chunks))) as pool:
            measured = list(pool.map(measure, chunks))
    else:
        measured = [measure(chunk) for chunk in chunks]
    for chunk, values in zip(chunks, measured, strict=True):
        weights[chunk] = values
    return tuple(weights)


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

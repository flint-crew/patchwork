"""Linear combination of the resampled inputs of a tile"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

CombineType = Literal["weighted", "average", "sum"]


class TileSums(NamedTuple):
    """Running sums over the inputs of a tile, for a group of images that share a weight.

    Each input contributes an estimate ``e`` of each image (its weighted image
    over its weight) with a weight ``w``. A weighted mean needs only the sums of
    ``w e`` and of ``w``, so the sums a plain mean or a sum also need are only
    kept for those.
    """

    totals: tuple[NDArray[np.float64], ...]
    """For each image, the sum of ``w e`` for a weighted mean, otherwise the sum of ``e``"""
    weight: NDArray[np.float64]
    """Sum of ``w``"""
    inverse_weight: NDArray[np.float64] | None = None
    """Sum of ``1 / w``, the summed variance, kept for a plain mean or a sum"""
    n_inputs: NDArray[np.float64] | None = None
    """Number of contributing inputs, kept for a plain mean or a sum"""

    def region(self, index: tuple[slice, ...]) -> TileSums:
        """The sums over part of the tile, as views that accumulate into these."""
        return TileSums(
            totals=tuple(total[index] for total in self.totals),
            weight=self.weight[index],
            inverse_weight=None
            if self.inverse_weight is None
            else self.inverse_weight[index],
            n_inputs=None if self.n_inputs is None else self.n_inputs[index],
        )


def empty_sums(
    shape: tuple[int, ...], n_images: int = 1, combine_type: CombineType = "weighted"
) -> TileSums:
    """Sums of a tile no input has contributed to yet, for a group of ``n_images`` images."""
    kept = combine_type != "weighted"
    return TileSums(
        totals=tuple(np.zeros(shape) for _ in range(n_images)),
        weight=np.zeros(shape),
        inverse_weight=np.zeros(shape) if kept else None,
        n_inputs=np.zeros(shape) if kept else None,
    )


def accumulate(
    sums: TileSums,
    images: Sequence[NDArray[np.float64]],
    factor: NDArray[np.float64] | float,
    weight: NDArray[np.float64],
    combine_type: CombineType = "weighted",
) -> None:
    """Add the images of an input to the sums, wherever its weight is positive.

    Each image contributes ``image * factor``, its weighted image, with weight
    ``weight``, as ``patchwork.weights.weighting`` gives them.

    Args:
        sums (TileSums): The sums of the region of the tile the input covers
        images (Sequence[NDArray[np.float64]]): The input's images of the group, in the order of the sums
        factor (NDArray[np.float64] | float): What the images are multiplied by to weight them
        weight (NDArray[np.float64]): The weight of the input at every pixel, which is 0 where it does not contribute
        combine_type (CombineType, optional): How the inputs are combined. Defaults to "weighted".
    """
    contributes = np.isfinite(weight) & (weight > 0)
    if combine_type == "weighted":
        multiplier = factor
    else:
        if sums.inverse_weight is None or sums.n_inputs is None:
            msg = f"Sums made for a weighted mean cannot be combined as {combine_type}"
            raise ValueError(msg)
        inverse = np.divide(1.0, weight, out=np.zeros(weight.shape), where=contributes)
        multiplier = factor * inverse
        np.add(sums.inverse_weight, inverse, out=sums.inverse_weight, where=contributes)
        np.add(sums.n_inputs, 1.0, out=sums.n_inputs, where=contributes)
    np.add(sums.weight, weight, out=sums.weight, where=contributes)

    scaled = np.empty(contributes.shape)
    for total, image in zip(sums.totals, images, strict=True):
        np.multiply(image, multiplier, out=scaled)
        np.add(total, scaled, out=total, where=contributes)


def finalise(
    sums: TileSums, combine_type: CombineType = "weighted"
) -> tuple[tuple[NDArray[np.float32], ...], NDArray[np.float32]]:
    """The combined images of a group and their weight, which are NaN where no input contributed.

    The weighted mean has the summed weight. The plain mean and the sum have the
    inverse of their variance.

    Args:
        sums (TileSums): The sums over the inputs of a tile
        combine_type (CombineType, optional): How to combine the inputs. Defaults to "weighted".

    Returns:
        tuple[tuple[NDArray[np.float32], ...], NDArray[np.float32]]: The images, and their weight
    """

    def blank() -> NDArray[np.float32]:
        return np.full(sums.weight.shape, np.nan, dtype=np.float32)

    if combine_type == "weighted":
        covered = sums.weight > 0
        images = tuple(
            np.divide(total, sums.weight, out=blank(), where=covered)
            for total in sums.totals
        )
        return images, _filled(sums.weight, covered)

    if sums.inverse_weight is None or sums.n_inputs is None:
        msg = f"Sums made for a weighted mean cannot be combined as {combine_type}"
        raise ValueError(msg)
    covered = sums.n_inputs > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        if combine_type == "average":
            images = tuple(
                np.divide(total, sums.n_inputs, out=blank(), where=covered)
                for total in sums.totals
            )
            weight = sums.n_inputs**2 / sums.inverse_weight
        else:
            images = tuple(_filled(total, covered) for total in sums.totals)
            weight = 1.0 / sums.inverse_weight
    return images, _filled(weight, covered)


def _filled(
    values: NDArray[np.float64], covered: NDArray[np.bool]
) -> NDArray[np.float32]:
    """Values as float32, which are NaN where not covered."""
    return np.where(covered, values, np.nan).astype(np.float32)

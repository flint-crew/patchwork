from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from astropy.io import fits

import patchwork.weights
from patchwork.io import readable_location
from patchwork.weights import (
    image_noise_weight,
    measure_image_weight,
    measure_weights,
    weight_from_map,
    weighting,
)

from .conftest import strided_weights

if TYPE_CHECKING:
    from pathlib import Path

    from .conftest import ImageFactory


@pytest.mark.parametrize("mode", ["mad", "std"])
def test_noise_weight(mode: str) -> None:
    data = np.random.default_rng(0).normal(0, 0.1, (200, 200))
    data[:10] = np.nan
    assert np.isclose(image_noise_weight(data, mode=mode, stride=3), 100, rtol=0.05)  # type: ignore[arg-type]


def test_noise_weight_of_empty_image() -> None:
    assert image_noise_weight(np.full((5, 5), np.nan)) == 0
    assert image_noise_weight(np.zeros((5, 5))) == 0


def test_measure_image_weight(make_image: ImageFactory) -> None:
    path = make_image("noisy.fits", 150.0, -30.0, noise=0.5)
    assert measure_image_weight(path, mode="none") == 1
    assert np.isclose(measure_image_weight(path, mode="mad", stride=1), 4, rtol=0.1)


def test_weight_from_map() -> None:
    values = np.array([4.0, 0.0, -1.0, np.nan])
    assert np.allclose(weight_from_map(values, "weight"), [4, 0, 0, 0])
    assert np.allclose(weight_from_map(values, "variance"), [0.25, 0, 0, 0])
    assert np.allclose(weight_from_map(values, "rms"), [1 / 16, 0, 1, 0])


@pytest.mark.parametrize(
    ("weight_type", "weight_state", "value", "weight"),
    [
        ("primary_beam", "corrected", 2 * 0.25, 0.25),
        ("primary_beam", "inherent", 2 * 0.5, 0.25),
        ("primary_beam", "weighted", 2, 0.25),
        ("combined", "corrected", 2 * 3 * 0.25, 3 * 0.25),
        ("combined", "inherent", 2 * 3 * 0.5, 3 * 0.25),
        ("combined", "weighted", 2 * 3, 3 * 0.25),
        ("weight_image", "corrected", 2 * 3, 3),
        ("weight_image", "inherent", 2 * np.sqrt(3), 3),
        ("weight_image", "weighted", 2, 3),
    ],
)
def test_weighting(
    weight_type: str, weight_state: str, value: float, weight: float
) -> None:
    """An image of 2 with a primary beam of 0.5 and a weight of 3 is weighted as linmos does."""
    image, pb = np.full(3, 2.0), np.full(3, 0.5)
    factor, got_weight = weighting(pb, 3.0, weight_type, weight_state)  # type: ignore[arg-type]
    assert np.allclose(image * factor, value)
    assert np.allclose(np.broadcast_to(got_weight, image.shape), weight)


def test_weighting_without_a_primary_beam() -> None:
    factor, weight = weighting(None, np.full(2, 3.0), "combined", "inherent")
    assert np.allclose(factor, 3)
    assert np.allclose(weight, 3)
    factor, weight = weighting(None, 3.0, "primary_beam", "corrected")
    assert np.allclose(factor, 1)
    assert np.allclose(weight, 1)


@pytest.mark.parametrize("mode", ["mad", "std"])
@pytest.mark.parametrize("stride", [3, 8])
# A plane in one read, a run of one stride at a time, and runs that end part way through the plane
@pytest.mark.parametrize("read_bytes", [64 * 2**20, 4 * 200, 4 * 200 * 20])
def test_measure_weights_as_rows_read_alone(
    make_image: ImageFactory,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["mad", "std"],
    stride: int,
    read_bytes: int,
) -> None:
    """Reading whole planes and keeping every ``stride``'th row gives the weights that reading those rows alone does."""
    monkeypatch.setattr(patchwork.weights, "NOISE_READ_BYTES", read_bytes)
    paths = [
        make_image(
            f"image{index}.fits", 150.0, -30.0, noise=0.1 * (index + 1), seed=index
        )
        for index in range(3)
    ]
    with fits.open(paths[1], mode="update") as hdul:
        hdul[0].data[0, 0, 40:60] = np.nan
    expected = [strided_weights(path, mode, stride) for path in paths]
    for threads in (1, 4):
        assert (
            measure_weights(paths, mode=mode, stride=stride, threads=threads)
            == expected
        )
    assert measure_weights(paths, mode="none") == [(1.0,)] * 3


def test_measure_weights_of_data_astropy_reads(
    make_image: ImageFactory, tmp_path: Path
) -> None:
    """Data that cannot be read straight from the file, such as float64, are measured through astropy, to the same weights."""
    path = make_image("image.fits", 150.0, -30.0, noise=0.1)
    with fits.open(path) as hdul:
        double = tmp_path / "double.fits"
        fits.writeto(double, hdul[0].data.astype(np.float64), hdul[0].header)
    assert readable_location(double) is None
    assert measure_weights([double]) == measure_weights([path])

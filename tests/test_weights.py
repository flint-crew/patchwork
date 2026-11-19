from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from patchwork.weights import (
    image_noise_weight,
    measure_image_weight,
    weight_from_map,
    weighting,
)

if TYPE_CHECKING:
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

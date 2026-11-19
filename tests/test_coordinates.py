from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from patchwork.coordinates import (
    map_to_input,
    offset_vectors,
    offsets,
    ra_dec,
    tile_coordinates,
    unit_vectors,
)
from patchwork.frame import new_wcs
from patchwork.primary_beam import direction_cosines

if TYPE_CHECKING:
    from astropy.wcs import WCS


def frame_wcs(dec: float = -30.0, projection: str = "SIN") -> WCS:
    wcs = new_wcs((150.0, dec), 0.004, projection)
    wcs.wcs.crpix = [300, 300]
    wcs.wcs.set()
    return wcs


def test_unit_vectors_round_trip() -> None:
    ra, dec = np.array([0.0, 359.0, 120.0]), np.array([-90.0, -45.0, 89.0])
    back_ra, back_dec = ra_dec(unit_vectors(ra, dec))
    assert np.allclose(back_dec, dec)
    assert np.allclose(back_ra[1:], ra[1:])


def test_offsets_are_direction_cosines() -> None:
    east, north = offsets((150.0, -30.0), unit_vectors(151.0, -29.0))
    assert np.allclose((east, north), direction_cosines(150.0, -30.0, 151.0, -29.0))


@pytest.mark.parametrize("centre", [(150.0, -30.0), (10.0, -89.9), (0.0, 0.0)])
def test_offset_vectors_undo_offsets(centre: tuple[float, float]) -> None:
    ra, dec = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7))
    vectors = unit_vectors(centre[0] + ra, np.clip(centre[1] + dec, -90, 90))
    assert np.allclose(offset_vectors(centre, *offsets(centre, vectors)), vectors)


def test_tile_vectors_match_exact() -> None:
    wcs = frame_wcs(dec=-89.0)
    tile = (slice(100, 400), slice(250, 500))
    coarse = tile_coordinates(wcs, tile, step=32)
    exact = tile_coordinates(wcs, tile, step=1)
    assert coarse.vectors.shape == (3, 300, 250)
    error_pixels = np.max(np.abs(coarse.vectors - exact.vectors)) / np.radians(0.004)
    assert error_pixels < 1e-3


def test_map_to_input_matches_exact() -> None:
    frame = frame_wcs()
    input_wcs = new_wcs((152.0, -31.0), 0.0025, "SIN")
    tile = (slice(0, 256), slice(0, 256))
    region = (slice(10, 200), slice(50, 256))
    coarse_x, coarse_y = map_to_input(
        tile_coordinates(frame, tile, step=16), input_wcs, region
    )
    exact_x, exact_y = map_to_input(
        tile_coordinates(frame, tile, step=1), input_wcs, region
    )
    assert coarse_x.shape == (190, 206)
    assert np.max(np.abs(coarse_x - exact_x)) < 1e-3
    assert np.max(np.abs(coarse_y - exact_y)) < 1e-3

    # With no tolerance the positions are transformed exactly
    strict_x, _ = map_to_input(
        tile_coordinates(frame, tile, step=16), input_wcs, region, tolerance=0.0
    )
    assert np.allclose(strict_x, exact_x, atol=1e-6)


def test_map_to_unreachable_input() -> None:
    """An input whose projection cannot reach part of the tile falls back to exact transforms."""
    frame = new_wcs((0.0, 0.0), 1.0, "CAR")
    frame.wcs.crpix = [90, 45]
    frame.wcs.set()
    far_side = new_wcs((60.0, 0.0), 0.1, "SIN")
    tile = (slice(0, 90), slice(0, 180))
    x, _ = map_to_input(
        tile_coordinates(frame, tile, step=16), far_side, (slice(None), slice(None))
    )
    assert np.any(np.isfinite(x))
    assert np.any(~np.isfinite(x))


@pytest.mark.parametrize("rows", [1, 2, 100])
def test_coarse_nodes_stay_within_tolerance(rows: int) -> None:
    """Interpolation between nodes is checked on tiles of any shape, so even sparse nodes give positions within the tolerance."""
    frame = new_wcs((0.0, -88.0), 60 / 3600, "SIN")
    input_wcs = new_wcs((150.0, -80.0), 60 / 3600, "TAN")
    input_wcs.wcs.crpix = [500, 500]
    input_wcs.wcs.set()
    # Far from the frame centre, where the projections bend most
    tile = (slice(-500, -500 + rows), slice(-800, 800))
    region = (slice(None), slice(None))
    exact_x, exact_y = map_to_input(
        tile_coordinates(frame, tile, step=1), input_wcs, region
    )
    for step in (16, 64, 256):
        x, y = map_to_input(tile_coordinates(frame, tile, step=step), input_wcs, region)
        assert np.max(np.hypot(x - exact_x, y - exact_y)) < 2e-3

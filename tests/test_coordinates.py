from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from patchwork.coordinates import (
    _between_nodes,
    _linear_weights,
    _nodes,
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


def _dense_weights(positions: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """The matrix that linearly interpolates values on nodes onto positions, one column per node."""
    return np.stack(
        [np.interp(positions, nodes, column) for column in np.eye(len(nodes))], axis=1
    )


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    # One block of positions, several (on and off the nodes), one node, every pixel a node
    [(0, 100, 16), (0, 1500, 16), (-37, 1500, 16), (5, 6, 16), (0, 9, 1)],
)
def test_blocks_match_one_matrix(start: int, stop: int, step: int) -> None:
    """Interpolating a block of positions at a time is interpolating with one matrix over every node."""
    rng = np.random.default_rng(3)
    y_positions = np.arange(start, stop, dtype=np.float64)
    x_positions = np.arange(2 * start, 2 * stop, dtype=np.float64)
    y_nodes = _nodes(start, stop, step)
    x_nodes = _nodes(2 * start, 2 * stop, step)
    values = rng.normal(size=(3, len(y_nodes), len(x_nodes)))
    dense = (
        _dense_weights(y_positions, y_nodes)
        @ values
        @ _dense_weights(x_positions, x_nodes).T
    )
    y_weights = _linear_weights(y_positions, y_nodes)
    x_weights = _linear_weights(x_positions, x_nodes)
    blocks = _between_nodes(y_weights, values, x_weights)
    assert blocks.shape == dense.shape
    if len(y_weights.matrices) == len(x_weights.matrices) == 1:
        # A tile within one block is made exactly as with one matrix
        assert np.array_equal(blocks, dense)
    assert np.allclose(blocks, dense, rtol=0, atol=1e-12)

    # As are parts of it
    rows, columns = slice(3, stop - start - 2), slice(1, None)
    part = _between_nodes(y_weights.select(rows), values, x_weights.select(columns))
    assert np.allclose(part, dense[:, rows, columns], rtol=0, atol=1e-12)


def test_strip_across_a_frame_maps_exactly() -> None:
    """A strip of a few rows across a wide frame near the pole, as the workers write, maps onto a turned input as exact transforms do."""
    scale = 15 / 3600
    frame = new_wcs((246.5, -84.1), scale, "SIN")
    frame.wcs.crpix = [1100, 1150]
    frame.wcs.set()
    # About 3 degrees across the frame, so its axes are turned about 30 degrees from the frame's
    input_wcs = new_wcs((276.0, -83.5), scale, "SIN")
    tile = (slice(1000, 1003), slice(0, 2200))
    region = (slice(1, 3), slice(300, 1900))
    coarse = tile_coordinates(frame, tile, step=16)
    exact = tile_coordinates(frame, tile, step=1)
    assert coarse.vectors.shape == (3, 3, 2200)
    # Interpolated, not transformed exactly as a fallback
    assert not np.array_equal(coarse.vectors, exact.vectors)
    assert np.max(np.abs(coarse.vectors - exact.vectors)) / np.radians(scale) < 1e-3
    x, y = map_to_input(coarse, input_wcs, region)
    exact_x, exact_y = map_to_input(exact, input_wcs, region)
    assert x.shape == (2, 1600)
    assert not np.array_equal(x, exact_x)
    assert np.max(np.hypot(x - exact_x, y - exact_y)) < 1e-3

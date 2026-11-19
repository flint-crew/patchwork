"""Sky positions as unit vectors, and the mapping of output pixels onto inputs.

Unit vectors behave everywhere on the sphere, including at the poles where RA is
undefined, so sky geometry is done with them rather than with RA and Dec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales

if TYPE_CHECKING:
    from astropy.wcs import WCS
    from numpy.typing import ArrayLike, NDArray

    from patchwork.io import Slices


def unit_vectors(ra_deg: ArrayLike, dec_deg: ArrayLike) -> NDArray[np.float64]:
    """Cartesian unit vectors, along a new leading axis, of sky positions in degrees."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def ra_dec(
    vectors: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The RA and Dec in degrees of unit vectors along the leading axis."""
    x, y, z = vectors
    return np.degrees(np.arctan2(y, x)) % 360, np.degrees(np.arctan2(z, np.hypot(x, y)))


def east_north(
    ra_deg: float, dec_deg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit vectors pointing east and north at a sky position."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.array(
        [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]
    )
    return east, north


def offsets(
    centre: tuple[float, float], vectors: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The offsets (east, north) of positions from a centre, as direction cosines."""
    east, north = east_north(*centre)
    return np.tensordot(east, vectors, axes=1), np.tensordot(north, vectors, axes=1)


def offset_vectors(
    centre: tuple[float, float], east: NDArray[np.float64], north: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The unit vectors of positions at offsets (east, north) from a centre, as direction cosines, on the centre's side of the sky."""
    east_axis, north_axis = east_north(*centre)
    towards = np.sqrt(np.clip(1 - east**2 - north**2, 0.0, None))
    return np.asarray(
        np.multiply.outer(unit_vectors(*centre), towards)
        + np.multiply.outer(east_axis, east)
        + np.multiply.outer(north_axis, north)
    )


def _linear_weights(
    positions: NDArray[np.float64], nodes: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The matrix that linearly interpolates values on nodes onto positions."""
    return np.stack(
        [np.interp(positions, nodes, column) for column in np.eye(len(nodes))], axis=1
    )


def _between_nodes(
    y_weights: NDArray[np.float64],
    values: NDArray[np.float64],
    x_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate values on a grid of nodes, shaped (..., ny_nodes, nx_nodes), with weight matrices."""
    return np.asarray(y_weights @ values @ x_weights.T)


def _nodes(start: int, stop: int, step: int) -> NDArray[np.float64]:
    """Positions every ``step`` pixels from ``start``, always including the last pixel."""
    return np.unique(np.append(np.arange(start, stop, step), stop - 1)).astype(
        np.float64
    )


def _between(nodes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Positions halfway between nodes, where interpolation is checked, or the node itself when there is only one."""
    return (nodes[:-1] + nodes[1:]) / 2 if len(nodes) > 1 else nodes


def _normalised(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectors along the leading axis scaled to unit length."""
    return np.asarray(vectors / np.linalg.norm(vectors, axis=0))


class TileCoordinates(NamedTuple):
    """The sky positions of the pixels of a tile, and what is needed to map them onto inputs.

    Coordinates are found exactly on a coarse grid of nodes and interpolated
    between them, which is far cheaper than a full transform per pixel. The
    interpolation is checked halfway between the nodes, and every pixel is
    transformed exactly where it misses by more than the tolerance.
    """

    vectors: NDArray[np.float64]
    """Unit vectors of every pixel of the tile, shaped (3, ny, nx)"""
    node_world: NDArray[np.float64]
    """The (RA, Dec) of the nodes, shaped (2, ny_nodes, nx_nodes)"""
    check_world: NDArray[np.float64]
    """The (RA, Dec) between the nodes, where the interpolation is checked"""
    y_weights: NDArray[np.float64]
    """Interpolates node rows onto the rows of the tile"""
    x_weights: NDArray[np.float64]
    """Interpolates node columns onto the columns of the tile"""
    y_check_weights: NDArray[np.float64]
    """Interpolates node rows onto the rows between them"""
    x_check_weights: NDArray[np.float64]
    """Interpolates node columns onto the columns between them"""


def tile_coordinates(
    wcs: WCS, tile: Slices, step: int = 16, tolerance: float = 1e-3
) -> TileCoordinates:
    """Find the sky positions of the pixels of a tile of a frame.

    Args:
        wcs (WCS): Celestial WCS of the frame
        tile (Slices): The (y, x) region of the frame
        step (int, optional): Spacing in pixels of the nodes the coordinates are found exactly on. 1 finds every pixel exactly. Defaults to 16.
        tolerance (float, optional): Largest error in pixels of the frame of interpolated positions. Defaults to 1e-3.

    Returns:
        TileCoordinates: The coordinates of the tile
    """
    ys, xs = tile
    y_pixels = np.arange(ys.start, ys.stop, dtype=np.float64)
    x_pixels = np.arange(xs.start, xs.stop, dtype=np.float64)
    y_nodes = _nodes(ys.start, ys.stop, max(step, 1))
    x_nodes = _nodes(xs.start, xs.stop, max(step, 1))
    y_check, x_check = _between(y_nodes), _between(x_nodes)

    node_world = np.array(wcs.pixel_to_world_values(*np.meshgrid(x_nodes, y_nodes)))
    check_world = np.array(wcs.pixel_to_world_values(*np.meshgrid(x_check, y_check)))
    y_weights = _linear_weights(y_pixels, y_nodes)
    x_weights = _linear_weights(x_pixels, x_nodes)
    y_check_weights = _linear_weights(y_check, y_nodes)
    x_check_weights = _linear_weights(x_check, x_nodes)

    vectors = None
    if (
        step > 1
        and np.all(np.isfinite(node_world))
        and np.all(np.isfinite(check_world))
    ):
        node_vectors = unit_vectors(*node_world)
        predicted = _normalised(
            _between_nodes(y_check_weights, node_vectors, x_check_weights)
        )
        # The chord between unit vectors is their separation in radians
        error = np.linalg.norm(predicted - unit_vectors(*check_world), axis=0)
        pixel_rad = np.radians(np.min(proj_plane_pixel_scales(wcs)))
        if np.max(error) <= tolerance * pixel_rad:
            vectors = _normalised(_between_nodes(y_weights, node_vectors, x_weights))
    if vectors is None:
        world = np.array(wcs.pixel_to_world_values(*np.meshgrid(x_pixels, y_pixels)))
        vectors = unit_vectors(*world)

    return TileCoordinates(
        vectors=vectors,
        node_world=node_world,
        check_world=check_world,
        y_weights=y_weights,
        x_weights=x_weights,
        y_check_weights=y_check_weights,
        x_check_weights=x_check_weights,
    )


def map_to_input(
    coordinates: TileCoordinates,
    wcs: WCS,
    region: Slices,
    tolerance: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The pixel positions in an input of a region of a tile.

    The positions are interpolated from the nodes of the tile, unless that
    misses the exact positions between the nodes by more than ``tolerance``
    pixels, or the input cannot be reached from every node, in which case every
    pixel is found exactly.

    Args:
        coordinates (TileCoordinates): The coordinates of the tile
        wcs (WCS): Celestial WCS of the input
        region (Slices): The (y, x) region of the tile, relative to the tile
        tolerance (float, optional): Largest error in pixels of interpolated positions. Defaults to 1e-3.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Zero-based x and y positions in the input
    """
    rows, columns = region
    nodes = np.array(wcs.world_to_pixel_values(*coordinates.node_world))
    if np.all(np.isfinite(nodes)) and coordinates.check_world.size:
        exact = np.array(wcs.world_to_pixel_values(*coordinates.check_world))
        predicted = _between_nodes(
            coordinates.y_check_weights, nodes, coordinates.x_check_weights
        )
        if np.max(np.abs(predicted - exact)) <= tolerance:
            x, y = _between_nodes(
                coordinates.y_weights[rows], nodes, coordinates.x_weights[columns]
            )
            return x, y

    x, y = wcs.world_to_pixel_values(*ra_dec(coordinates.vectors[:, rows, columns]))
    return np.asarray(x), np.asarray(y)

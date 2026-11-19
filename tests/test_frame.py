from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_area

from patchwork.coordinates import offset_vectors, ra_dec
from patchwork.frame import (
    Frame,
    build_frame,
    central_index,
    fitted_frame,
    frame_from_header,
    input_bounding_box,
    intersect_boxes,
    reference_frame,
    strips,
    tile_region,
    tiles,
    world_bounding_box,
)
from patchwork.io import read_image_info
from patchwork.options import FrameOptions

if TYPE_CHECKING:
    from .conftest import ImageFactory


def test_single_input_keeps_its_grid(make_image: ImageFactory) -> None:
    info = read_image_info(make_image("a.fits", 150.0, -30.0, size=100))
    frame = reference_frame([info])
    assert frame.shape == (100, 100)
    assert np.allclose(frame.wcs.wcs.crpix, info.wcs.wcs.crpix)
    assert np.allclose(frame.wcs.wcs.crval, info.wcs.wcs.crval)


def test_reference_frame_covers_inputs(make_image: ImageFactory) -> None:
    infos = [
        read_image_info(make_image("a.fits", 150.0, -30.0, size=100)),
        read_image_info(make_image("b.fits", 150.0, -29.6, size=100)),
    ]
    frame = reference_frame(infos, reference_index=0)
    assert frame.shape == (200, 100)
    assert np.allclose(frame.wcs.wcs.crval, infos[0].wcs.wcs.crval)

    ra, dec = frame.wcs.pixel_to_world_values(50, 150)
    assert np.isclose(dec, -29.6, atol=0.004)
    assert np.isclose(ra, 150.0, atol=0.01)


def test_central_index(make_image: ImageFactory) -> None:
    infos = [
        read_image_info(make_image(f"{index}.fits", 150.0 + offset, -30.0, size=20))
        for index, offset in enumerate((-1.0, 0.1, 1.0, 2.0))
    ]
    assert central_index(infos) == 1


def test_reference_frame_with_new_centre(make_image: ImageFactory) -> None:
    info = read_image_info(make_image("a.fits", 150.0, -30.0, size=100))
    frame = reference_frame([info], centre=(150.1, -30.05), projection="TAN")
    assert frame.wcs.wcs.ctype[0] == "RA---TAN"
    assert np.allclose(frame.wcs.wcs.crval, [150.1, -30.05])
    assert np.isclose(proj_plane_pixel_area(frame.wcs), 0.004**2)


def test_fitted_frame(make_image: ImageFactory) -> None:
    infos = [
        read_image_info(make_image("a.fits", 150.0, -30.0, size=100, cdelt=0.004)),
        read_image_info(make_image("b.fits", 150.4, -30.0, size=200, cdelt=0.002)),
        read_image_info(make_image("c.fits", 150.8, -30.0, size=134, cdelt=0.003)),
    ]
    frame = fitted_frame(infos, FrameOptions(frame_type="fit"))
    assert np.isclose(np.sqrt(proj_plane_pixel_area(frame.wcs)), 0.003)
    assert np.isclose(frame.wcs.wcs.crval[0], 150.4, atol=0.01)
    assert np.isclose(frame.wcs.wcs.crval[1], -30.0, atol=0.01)

    sized = fitted_frame(
        infos,
        FrameOptions(frame_type="fit", image_size=(64, 32), pixel_scale_type="max"),
    )
    assert sized.shape == (32, 64)
    assert np.isclose(np.sqrt(proj_plane_pixel_area(sized.wcs)), 0.004)


def test_fitted_frame_across_zero_ra(make_image: ImageFactory) -> None:
    infos = [
        read_image_info(make_image("a.fits", 359.8, 10.0, size=50)),
        read_image_info(make_image("b.fits", 0.2, 10.0, size=50)),
    ]
    frame = build_frame(infos, FrameOptions(frame_type="fit"))
    assert np.isclose(frame.wcs.wcs.crval[0], 0.0, atol=0.01) or np.isclose(
        frame.wcs.wcs.crval[0], 360.0, atol=0.01
    )
    assert frame.shape[1] < 200


def test_most_covered_centre(make_image: ImageFactory) -> None:
    infos = [
        read_image_info(make_image(f"{index}.fits", 150.0, dec, size=50))
        for index, dec in enumerate((-30.0, -30.0, -30.0, -25.0))
    ]
    frame = fitted_frame(infos, FrameOptions(frame_type="fit", centre_type="most"))
    assert np.isclose(frame.wcs.wcs.crval[1], -30.0, atol=0.01)


def test_frame_header_round_trip(make_image: ImageFactory) -> None:
    info = read_image_info(make_image("a.fits", 150.0, -30.0, size=100))
    frame = reference_frame([info])
    header = frame.header(info.header)
    assert header["NAXIS"] == 4
    assert header["CTYPE3"] == "FREQ"
    restored = frame_from_header(header)
    assert restored.shape == frame.shape
    assert np.allclose(restored.wcs.wcs.crval, frame.wcs.wcs.crval)


def test_tiles_and_regions(make_image: ImageFactory) -> None:
    info = read_image_info(make_image("a.fits", 150.0, -30.0, size=100))
    frame = Frame(wcs=info.wcs, shape=(100, 250))
    frame_tiles = tiles(frame, tile_size=64)
    assert len(frame_tiles) == 2 * 4
    assert frame_tiles[-1] == (slice(64, 100), slice(192, 250))

    box = input_bounding_box(frame, info, margin=0)
    assert box == (0, 99, 0, 100)
    assert tile_region(frame_tiles[0], box) == (slice(0, 64), slice(0, 64))
    assert tile_region(frame_tiles[1], box) == (slice(0, 64), slice(0, 37))
    assert tile_region(frame_tiles[3], box) is None
    assert tile_region(frame_tiles[0], None) is None


def test_strips(make_image: ImageFactory) -> None:
    """Strips of whole rows cover the frame once, each about as large as a square tile."""
    info = read_image_info(make_image("a.fits", 150.0, -30.0, size=100))
    frame = Frame(wcs=info.wcs, shape=(100, 250))
    frame_strips = strips(frame, tile_size=64)
    # 64^2 pixels are 17 rows of 250, rounded up
    assert frame_strips[0] == (slice(0, 17), slice(0, 250))
    assert frame_strips[-1] == (slice(85, 100), slice(0, 250))
    covered = np.zeros(frame.shape, dtype=int)
    for ys, xs in frame_strips:
        covered[ys, xs] += 1
    assert np.all(covered == 1)
    # A frame narrower than a tile has strips of at least a tile's area
    assert strips(Frame(wcs=info.wcs, shape=(100, 20)), tile_size=64)[0] == (
        slice(0, 100),
        slice(0, 20),
    )


def test_circle_box(make_image: ImageFactory) -> None:
    info = read_image_info(make_image("a.fits", 150.0, -30.0, size=200))
    frame = Frame(wcs=info.wcs, shape=(200, 200))
    radius, angles = np.radians(0.1), np.linspace(0, 2 * np.pi, 256, endpoint=False)
    circle = offset_vectors(
        (150.0, -30.0), np.sin(radius) * np.sin(angles), np.sin(radius) * np.cos(angles)
    )
    box = world_bounding_box(frame, np.array(ra_dec(circle)), margin=0)
    assert box is not None
    # A radius of 0.1 deg is 25 pixels of 0.004 deg about the centre pixel (100, 100)
    assert np.allclose(box, (75, 125, 75, 125), atol=1)
    assert intersect_boxes(box, (0, 80, 0, 199)) == (box[0], 80, box[2], box[3])
    assert intersect_boxes(box, (150, 199, 0, 199)) is None


def test_frame_from_file(make_image: ImageFactory) -> None:
    path = make_image("a.fits", 150.0, -30.0, size=40)
    frame = frame_from_header(path)
    assert frame.shape == (40, 40)
    assert fits.getheader(path)["CRVAL1"] == frame.wcs.wcs.crval[0]


def test_fitted_frame_around_the_pole(make_image: ImageFactory) -> None:
    """Around a pole the inputs span every RA, which an RA range cannot centre."""
    infos = [
        read_image_info(make_image(f"{index}.fits", ra, -88.5, size=100, cdelt=0.02))
        for index, ra in enumerate((0.0, 90.0, 180.0, 270.0))
    ]
    frame = fitted_frame(infos, FrameOptions(frame_type="fit"))
    assert np.isclose(frame.wcs.wcs.crval[1], -90.0, atol=0.01)
    # The frame covers all four inputs but no more than their extent across the pole
    assert frame.shape[0] < 2 * (1.5 + 1.0) / 0.02 + 10
    _, dec = frame.wcs.pixel_to_world_values(frame.shape[1] / 2, frame.shape[0] / 2)
    assert np.isclose(dec, -90.0, atol=0.1)

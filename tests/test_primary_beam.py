from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from astropy.constants import c as speed_of_light
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.ndimage import map_coordinates

import patchwork.primary_beam
from patchwork.io import read_image_info, read_plane_at
from patchwork.logging import logger
from patchwork.mosaic import mosaic
from patchwork.options import MosaicOptions, PrimaryBeamOptions, WeightOptions
from patchwork.primary_beam import (
    BeamModel,
    GaussianBeam,
    bilinear_taps,
    direction_cosines,
    gaussian_beam,
    load_holography_beam,
    make_primary_beam,
    nearest_planes,
    pb_rotation_from_pol_axis,
    usable_planes,
)

from .conftest import FREQUENCY_HZ, ImageFactory

if TYPE_CHECKING:
    from pathlib import Path


def write_holography_cube(
    path: Path,
    freqs_hz: np.ndarray,
    size: int = 101,
    cdelt_deg: float = 0.0345,
    elongation: float = 1.0,
    n_beams: int = 2,
) -> Path:
    """A holography cube of beams whose planes are the Gaussian beam of each frequency."""
    centre = size // 2
    y, x = np.mgrid[:size, :size]
    offset_x = np.radians((x - centre) * cdelt_deg)
    offset_y = np.radians((y - centre) * cdelt_deg)
    planes = []
    for freq in freqs_hz:
        fwhm = gaussian_beam(freq).fwhm_rad
        planes.append(
            np.exp(
                -4 * np.log(2) * ((offset_x / elongation) ** 2 + offset_y**2) / fwhm**2
            )
        )
    beam = np.array(planes)[None]
    data = np.repeat(np.repeat(beam, 4, axis=0)[None], n_beams, axis=0)

    header = fits.Header()
    for axis, (ctype, crval, cdelt, crpix) in enumerate(
        [
            ("RA---SIN", 0.0, -cdelt_deg, centre + 1),
            ("DEC--SIN", 0.0, cdelt_deg, centre + 1),
            ("FREQ", freqs_hz[0], freqs_hz[1] - freqs_hz[0], 1),
        ],
        start=1,
    ):
        header[f"CTYPE{axis}"] = ctype
        header[f"CRVAL{axis}"] = crval
        header[f"CDELT{axis}"] = cdelt
        header[f"CRPIX{axis}"] = crpix
    fits.writeto(path, data.astype(np.float32), header)
    return path


def test_gaussian_half_power_at_half_width() -> None:
    beam = GaussianBeam(fwhm_rad=np.radians(1.0))
    east = np.array([np.sin(np.radians(0.5)), 0.0])
    north = np.array([0.0, 0.0])
    assert np.allclose(beam.response(east, north), [0.5, 1.0])


def test_gaussian_beam_width() -> None:
    wavelength = speed_of_light.value / FREQUENCY_HZ
    assert np.isclose(gaussian_beam(FREQUENCY_HZ).fwhm_rad, 1.09 * wavelength / 12)


def test_direction_cosines_match_astropy() -> None:
    centre = SkyCoord(150.0, -30.0, unit="deg")
    position = SkyCoord(151.2, -29.1, unit="deg")
    east, north = direction_cosines(150.0, -30.0, 151.2, -29.1)
    separation = centre.separation(position).rad
    position_angle = centre.position_angle(position).rad
    assert np.isclose(np.hypot(east, north), np.sin(separation))
    assert np.isclose(np.arctan2(east, north), position_angle)


def test_holography_matches_gaussian(tmp_path: Path) -> None:
    freqs = np.array([9.0e8, 9.4e8, 9.8e8])
    cube = write_holography_cube(tmp_path / "cube.fits", freqs)
    offsets = np.radians(np.linspace(0, 1.2, 25))

    for frequency in (9.4e8, 9.45e8):
        beam = load_holography_beam(cube, beam_number=1, frequency_hz=frequency)
        expected = gaussian_beam(frequency).response(
            np.sin(offsets), np.zeros_like(offsets)
        )
        assert np.allclose(
            beam.response(np.sin(offsets), np.zeros_like(offsets)), expected, atol=5e-3
        )


def test_holography_rotation(tmp_path: Path) -> None:
    cube = write_holography_cube(
        tmp_path / "cube.fits", np.array([9.4e8, 9.5e8]), elongation=2.0
    )
    offset = np.sin(np.radians(0.8))
    plain = load_holography_beam(cube, beam_number=0, frequency_hz=9.4e8)
    rotated = load_holography_beam(
        cube, beam_number=0, frequency_hz=9.4e8, alpha_rad=np.pi / 2
    )
    assert np.isclose(
        rotated.response(np.array([0.0]), np.array([offset]))[0],
        plain.response(np.array([-offset]), np.array([0.0]))[0],
    )
    assert (
        plain.response(np.array([offset]), np.array([0.0]))[0]
        > plain.response(np.array([0.0]), np.array([offset]))[0]
    )


def test_holography_off_cube_is_zero(tmp_path: Path) -> None:
    cube = write_holography_cube(tmp_path / "cube.fits", np.array([9.4e8, 9.5e8]))
    beam = load_holography_beam(cube, beam_number=0, frequency_hz=9.4e8)
    assert beam.response(np.array([np.sin(np.radians(3.0))]), np.array([0.0]))[0] == 0


def test_make_primary_beam(make_image: ImageFactory, tmp_path: Path) -> None:
    info = read_image_info(make_image("SB1.field.beam01.fits", 150.0, -30.0))
    assert info.beam_number == 1
    assert make_primary_beam(PrimaryBeamOptions(pb_model="none"), info) is None
    assert isinstance(make_primary_beam(PrimaryBeamOptions(), info), GaussianBeam)

    with pytest.raises(ValueError, match="holography cube"):
        make_primary_beam(PrimaryBeamOptions(pb_model="holography"), info)
    cube = write_holography_cube(tmp_path / "cube.fits", np.array([9.4e8, 9.5e8]))
    holography = make_primary_beam(
        PrimaryBeamOptions(pb_model="holography", holography_cube=cube), info
    )
    assert holography is not None
    assert np.isclose(holography.response(np.zeros(1), np.zeros(1))[0], 1.0)


def flag_plane(cube: Path, beam_number: int, plane: int) -> None:
    """Blank one plane of a beam of a holography cube, as flagged channels are."""
    with fits.open(cube, mode="update") as hdul:
        hdul[0].data[beam_number, :, plane] = np.nan


def test_nearest_planes_skip_flagged(tmp_path: Path) -> None:
    cube = write_holography_cube(
        tmp_path / "cube.fits", np.array([9.0e8, 9.4e8, 9.8e8])
    )
    flag_plane(cube, beam_number=1, plane=1)
    planes, freqs = nearest_planes(cube, 1, [9.3e8, 9.79e8])
    assert planes.tolist() == [0, 2]
    assert np.allclose(freqs, [9.0e8, 9.8e8])
    assert nearest_planes(cube, 0, [9.3e8, 9.79e8])[0].tolist() == [1, 2]


def scanned_nearest_planes(
    cube: Path, beam_number: int, frequencies_hz: np.ndarray
) -> list[int]:
    """The usable planes nearest some frequencies as found by reading the centre of every plane, as patchwork found them before."""
    freqs_hz, usable, _ = usable_planes(cube, beam_number)
    wanted = np.reshape(frequencies_hz, (-1, 1))
    return [
        int(plane)
        for plane in usable[np.argmin(np.abs(freqs_hz[usable] - wanted), axis=1)]
    ]


def flagged_cube(path: Path, n_planes: int = 40, n_beams: int = 3) -> Path:
    """A cube of beams with many planes flagged in different ways, and a centre between pixels."""
    rng = np.random.default_rng(3)
    cube = write_holography_cube(
        path, 8e8 + 1e6 * np.arange(n_planes), size=21, n_beams=n_beams
    )
    with fits.open(cube, mode="update") as hdul:
        hdul[0].header["CRPIX1"] += 0.3
        hdul[0].header["CRPIX2"] -= 0.4
        data = hdul[0].data
        # Leakage that differs between the Stokes planes
        data[:, 1:] = rng.normal(0, 0.01, data[:, 1:].shape)
        for beam in range(n_beams):
            for plane in rng.choice(n_planes, size=n_planes // 3, replace=False):
                how = rng.integers(3)
                if how == 0:
                    data[beam, 0, plane] = np.nan
                elif how == 1:
                    data[beam, 0, plane] = 0.0
                else:
                    # Only the pixels around the centre
                    data[beam, 0, plane, 9:12, 9:12] = -1.0
    return cube


def test_nearest_planes_match_a_scan_of_every_plane(tmp_path: Path) -> None:
    """Looking out from the nearest plane picks the planes that reading the centre of every plane does, ties included."""
    cube = flagged_cube(tmp_path / "cube.fits")
    freqs = 8e8 + 1e6 * np.arange(40)
    # On planes, between them, exactly halfway between them, and beyond the ends
    wanted = np.concatenate([freqs, freqs[:-1] + 3e5, freqs[:-1] + 5e5, [7e8, 9e8]])
    for beam in range(3):
        planes, plane_freqs = nearest_planes(cube, beam, wanted)
        assert planes.tolist() == scanned_nearest_planes(cube, beam, wanted)
        assert np.array_equal(plane_freqs, freqs[planes])


@pytest.mark.parametrize("normalise", [True, False])
def test_holography_beam_as_read_through_astropy(
    tmp_path: Path, normalise: bool
) -> None:
    """A beam read a plane at a time holds what astropy reads of the cube, normalised by the response at the centre as before."""
    cube = flagged_cube(tmp_path / "cube.fits")
    with fits.open(cube) as hdul:
        data = np.nan_to_num(np.asarray(hdul[0].data, dtype=np.float64))
        x0, y0 = hdul[0].header["CRPIX1"] - 1, hdul[0].header["CRPIX2"] - 1
    for beam_number in range(3):
        for frequency in (8.0e8, 8.123e8, 8.3e8):
            plane = scanned_nearest_planes(cube, beam_number, np.array([frequency]))[0]
            beam = load_holography_beam(
                cube,
                beam_number,
                frequency,
                leakage=True,
                normalise=normalise,
            )
            stokes = data[beam_number, :, plane]
            centre = usable_planes(cube, beam_number)[2][plane]
            assert np.array_equal(
                beam.plane, stokes[0] / centre if normalise else stokes[0]
            )
            at_centre = map_coordinates(
                stokes[1:],
                np.array([np.arange(3), np.full(3, y0), np.full(3, x0)]),
                order=1,
            )
            assert beam.leakage is not None
            assert np.array_equal(beam.leakage, stokes[1:] - at_centre[:, None, None])
            assert beam.centre_pixel == (x0, y0)


def test_holography_planes_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each plane is read once, however many beams with other rotations are built from it."""
    cube = write_holography_cube(
        tmp_path / "cube.fits", np.array([9.0e8, 9.4e8, 9.8e8])
    )
    reads: list[tuple[int, ...]] = []
    read = read_plane_at

    def counted(location: Any, index: tuple[int, ...]) -> np.ndarray:
        reads.append(index)
        return read(location, index)

    monkeypatch.setattr(patchwork.primary_beam, "read_plane_at", counted)
    beams = [
        load_holography_beam(cube, 0, 9.4e8, leakage=True, frame_rotation_rad=rotation)
        for rotation in (0.0, 0.3, 1.1)
    ]
    assert sorted(reads) == [(0, stokes, 1) for stokes in range(4)]
    assert all(np.array_equal(beam.plane, beams[0].plane) for beam in beams)
    # Each beam has its own copy, so changing one leaves the kept plane as it was
    expected = beams[0].plane.copy()
    beams[0].plane[:] = 0
    again = load_holography_beam(cube, 0, 9.4e8, frame_rotation_rad=2.0)
    assert np.array_equal(again.plane, expected)
    assert len(reads) == 4


def test_holography_cube_read_again_once_rewritten(tmp_path: Path) -> None:
    """A cube written again at the same path is read again, not taken from what was kept of it."""
    cube = write_holography_cube(
        tmp_path / "cube.fits", np.array([9.0e8, 9.4e8, 9.8e8])
    )
    before = load_holography_beam(cube, 0, 9.4e8, normalise=False)
    with fits.open(cube, mode="update") as hdul:
        hdul[0].data *= 2
    status = cube.stat()
    os.utime(cube, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
    after = load_holography_beam(cube, 0, 9.4e8, normalise=False)
    assert np.allclose(after.plane, 2 * before.plane)


def test_holography_cube_astropy_reads(tmp_path: Path) -> None:
    """A cube whose data cannot be read straight from the file, such as float64, gives the same beams through astropy."""
    cube = flagged_cube(tmp_path / "cube.fits")
    with fits.open(cube) as hdul:
        double = tmp_path / "double.fits"
        fits.writeto(double, hdul[0].data.astype(np.float64), hdul[0].header)
    for frequency in (8.0e8, 8.123e8):
        got = load_holography_beam(double, 1, frequency, leakage=True)
        expected = load_holography_beam(cube, 1, frequency, leakage=True)
        assert np.array_equal(got.plane, expected.plane)
        assert np.array_equal(got.leakage, expected.leakage)  # type: ignore[arg-type]


@pytest.mark.parametrize("flagged", [False, True])
def test_warns_of_distant_holography_planes(
    make_image: ImageFactory,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    flagged: bool,
) -> None:
    """A channel amid flagged planes takes its beam from a plane far from it, which is warned of."""
    image = make_image("SB1.field.beam01.fits", 150.0, -30.0)
    cube = write_holography_cube(
        tmp_path / "cube.fits", np.array([9.0e8, FREQUENCY_HZ, 9.8e8])
    )
    if flagged:
        flag_plane(cube, beam_number=1, plane=1)
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(pb_model="holography", holography_cube=cube),
        weight=WeightOptions(image_weight="none"),
    )
    mosaic([image], tmp_path / "field", options)
    assert ("40.0 MHz away" in caplog.text) is flagged


def test_pb_rotation_from_pol_axis() -> None:
    assert np.isclose(pb_rotation_from_pol_axis(-np.pi / 4), 0.0)


def test_gaussian_support_radius() -> None:
    beam = gaussian_beam(FREQUENCY_HZ)
    radius = beam.support_radius(0.1)
    assert radius is not None
    assert np.isclose(beam.response(np.array([np.sin(radius)]), np.zeros(1))[0], 0.1)
    assert beam.support_radius(0.0) is None


def test_gaussian_reach_is_the_widest_circle() -> None:
    model = BeamModel(options=PrimaryBeamOptions(), frequencies_hz=(1.0e9, 8.0e8))
    reach = model.reach(0.1)
    radius = gaussian_beam(8.0e8).support_radius(0.1)
    assert reach is not None
    assert radius is not None
    # The polygon through the points touches the circle with its edges
    assert np.allclose(np.hypot(*reach), np.sin(radius / np.cos(np.pi / 64)))
    assert model.reach(0.0) is None


@pytest.mark.parametrize("alpha", [0.0, 0.4, np.pi / 4])
def test_holography_reach_holds_every_channel(tmp_path: Path, alpha: float) -> None:
    """Wherever some channel reaches the cutoff is within the beam's reach, here a sidelobe of the highest plane beyond where the lowest reaches."""
    cube = write_holography_cube(
        tmp_path / "cube.fits", np.array([9.4e8, 9.5e8, 9.6e8])
    )
    with fits.open(cube, mode="update") as hdul:
        hdul[0].data[:, :, 2, 95:98, 5:8] = 0.3
    model = BeamModel(
        options=PrimaryBeamOptions(
            pb_model="holography", holography_cube=cube, holography_alpha_rad=alpha
        ),
        frequencies_hz=(9.4e8, 9.55e8, 9.6e8),
        beam_number=1,
    )
    reach = model.reach(0.1)
    assert reach is not None

    east, north = np.meshgrid(
        np.linspace(-0.04, 0.04, 801), np.linspace(-0.04, 0.04, 801)
    )
    reached = np.any(model.responses(range(3), east, north) >= 0.1, axis=0)
    lowest = model.responses(range(1), east, north)[0] >= 0.1
    assert np.max(np.hypot(east, north)[reached]) > 1.4 * np.max(
        np.hypot(east, north)[lowest]
    )
    # Every position reached is within the reach, which is no more than two of the
    # cube's pixels wider on any side
    pixel = np.radians(0.0345)
    for offsets, bound in ((east, reach[0]), (north, reach[1])):
        assert np.min(bound) <= np.min(offsets[reached])
        assert np.max(offsets[reached]) <= np.max(bound)
        assert np.min(offsets[reached]) - np.min(bound) < 2 * pixel
        assert np.max(bound) - np.max(offsets[reached]) < 2 * pixel


def test_holography_reach_of_a_full_grid_is_its_square(tmp_path: Path) -> None:
    """A beam above the cutoff over its whole grid reaches that square, a pixel wider for interpolation, not the circle around it."""
    cube = write_holography_cube(tmp_path / "cube.fits", np.array([9.4e8, 9.5e8]))
    model = BeamModel(
        options=PrimaryBeamOptions(pb_model="holography", holography_cube=cube),
        frequencies_hz=(9.4e8,),
        beam_number=0,
    )
    reach = model.reach(0.001)
    assert reach is not None
    half_width = np.radians(51 * 0.0345)
    assert np.isclose(np.max(np.abs(reach[0])), half_width, rtol=1e-6)
    assert np.isclose(np.max(np.abs(reach[1])), half_width, rtol=1e-6)
    # No channel reaches a cutoff above the beam's peak
    nowhere = model.reach(2.0)
    assert nowhere is not None
    assert nowhere[0].size == 0


def test_holography_frame_rotation(tmp_path: Path) -> None:
    """Turning the frame of the input turns the beam pattern as the feeds do."""
    cube = write_holography_cube(
        tmp_path / "cube.fits", np.array([9.4e8, 9.5e8]), elongation=2.0
    )
    by_alpha = load_holography_beam(
        cube, beam_number=0, frequency_hz=9.4e8, alpha_rad=0.4
    )
    by_frame = load_holography_beam(
        cube, beam_number=0, frequency_hz=9.4e8, frame_rotation_rad=0.4
    )
    east, north = (
        np.sin(np.radians([0.3, 0.7, -0.5])),
        np.sin(np.radians([0.6, -0.2, 0.4])),
    )
    assert np.allclose(by_alpha.response(east, north), by_frame.response(east, north))


def test_holography_sampling(tmp_path: Path) -> None:
    """The beam and its leakage are read from the cube at the positions, rotation and scale that docs/holography.md gives."""
    size, cdelt_deg, alpha = 41, 0.1, 0.7
    freqs = np.array([9.4e8, 9.5e8])
    rng = np.random.default_rng(1)
    planes = rng.normal(0.0, 0.05, size=(4, len(freqs), size, size))
    planes[0] += 1.02
    header = fits.Header()
    for axis, (ctype, crval, cdelt, crpix) in enumerate(
        [
            ("RA---SIN", 0.0, -cdelt_deg, size // 2 + 1),
            ("DEC--SIN", 0.0, cdelt_deg, size // 2 + 1),
            ("FREQ", freqs[0], freqs[1] - freqs[0], 1),
        ],
        start=1,
    ):
        header[f"CTYPE{axis}"] = ctype
        header[f"CRVAL{axis}"] = crval
        header[f"CDELT{axis}"] = cdelt
        header[f"CRPIX{axis}"] = crpix
    path = tmp_path / "cube.fits"
    fits.writeto(path, planes[None].astype(np.float32), header)

    # The second plane, with the leakage at the beam centre removed
    cube = fits.getdata(path)[0, :, 1].astype(np.float64)
    centre = size // 2
    cube[1:] -= cube[1:, centre, centre][:, None, None]
    # Offsets r at position angles PA from the beam centre, and the pixels of the cube
    # they fall on at the frequency of the plane, where the stretch s is 1
    r = np.radians(rng.uniform(0.0, 2.5, 500))
    pa = rng.uniform(-np.pi, np.pi, 500)
    x = np.sin(r) * np.sin(pa - alpha)
    y = np.sin(r) * np.cos(pa - alpha)
    pixels = np.array(
        [y / np.radians(cdelt_deg) + centre, x / np.radians(-cdelt_deg) + centre]
    )

    def sample(plane: np.ndarray) -> np.ndarray:
        return np.asarray(
            map_coordinates(plane, pixels, order=1, mode="constant", cval=0.0)
        )

    leakage_q, leakage_u = sample(cube[1]), sample(cube[2])
    expected_leakage = [
        leakage_q * np.cos(2 * alpha) - leakage_u * np.sin(2 * alpha),
        leakage_q * np.sin(2 * alpha) + leakage_u * np.cos(2 * alpha),
        sample(cube[3]),
    ]

    beam = load_holography_beam(
        path, beam_number=0, frequency_hz=freqs[1], alpha_rad=alpha, leakage=True
    )
    east, north = np.sin(r) * np.sin(pa), np.sin(r) * np.cos(pa)
    assert np.allclose(
        beam.response(east, north),
        sample(cube[0]) / cube[0, centre, centre],
        rtol=0,
        atol=1e-12,
    )
    assert np.allclose(
        beam.leakage_response(east, north), expected_leakage, rtol=0, atol=1e-12
    )

    # Used as the cube stores it, the beam is not scaled, and is cut off from its peak
    stored = load_holography_beam(
        path, beam_number=0, frequency_hz=freqs[1], alpha_rad=alpha, normalise=False
    )
    assert np.allclose(
        stored.response(east, north), sample(cube[0]), rtol=0, atol=1e-12
    )
    assert stored.cutoff_scale == np.max(cube[0])


def test_holography_leakage_needs_stokes(tmp_path: Path) -> None:
    cube = write_holography_cube(tmp_path / "cube.fits", np.array([9.4e8, 9.5e8]))
    beam = load_holography_beam(cube, beam_number=0, frequency_hz=9.4e8)
    with pytest.raises(ValueError, match="leakage"):
        beam.leakage_response(np.zeros(1), np.zeros(1))
    leaky = load_holography_beam(cube, beam_number=0, frequency_hz=9.4e8, leakage=True)
    # The synthetic cube repeats Stokes I in every Stokes plane, so with the centre removed the leakage is 0 there
    assert np.allclose(leaky.leakage_response(np.zeros(1), np.zeros(1)), 0)


@pytest.mark.parametrize("shape", [(101, 101), (7, 5), (1, 6)])
def test_bilinear_taps_match_scipy(shape: tuple[int, int]) -> None:
    """Linear interpolation of a holography plane matches scipy's, on, at and off the edges of the grid."""
    rng = np.random.default_rng(0)
    plane = rng.normal(size=shape)
    ny, nx = shape
    edges = [
        [0, ny - 1, ny - 1, -1e-9, ny - 1 + 1e-9, -0.4],
        [0, nx - 1, 0.3, 1, 0, 0.5],
    ]
    pixels = np.concatenate([rng.uniform(-2, max(shape) + 1, (2, 2000)), edges], axis=1)
    expected = map_coordinates(plane, pixels, order=1, mode="constant", cval=0.0)
    got = bilinear_taps(pixels, shape, (pixels.shape[1],)).sample(plane)
    assert np.allclose(got, expected, rtol=0, atol=1e-12)
    assert np.all(got[-3:] == 0)


def test_beam_model_shares_taps_between_channels(tmp_path: Path) -> None:
    """A beam model gives every channel the response and leakage its own holography beam gives."""
    frequencies = np.array([9.4e8, 9.5e8, 9.6e8])
    cube = write_holography_cube(tmp_path / "cube.fits", frequencies)
    options = PrimaryBeamOptions(
        pb_model="holography", holography_cube=cube, holography_alpha_rad=0.3
    )
    # Channels on the planes share one set of taps, and one between them has its own
    channel_frequencies = (9.4e8, 9.5e8, 9.55e8, 9.6e8)
    model = BeamModel(
        options=options,
        frequencies_hz=channel_frequencies,
        beam_number=1,
        leakage=True,
        normalise_leakage=False,
    )
    east, north = np.meshgrid(
        np.linspace(-0.03, 0.03, 40), np.linspace(-0.02, 0.025, 30)
    )
    responses = model.responses(range(4), east, north)
    leakage = model.leakage_responses(range(4), east, north)
    without_v = model.leakage_responses(range(4), east, north, include_v=False)
    assert responses.shape == (4, 30, 40)
    assert leakage.shape == (4, 3, 30, 40)
    assert np.array_equal(without_v, leakage[:, :2])
    for channel, frequency in enumerate(channel_frequencies):
        beam = load_holography_beam(
            cube,
            beam_number=1,
            frequency_hz=frequency,
            alpha_rad=0.3,
            leakage=True,
            normalise_leakage=False,
        )
        assert np.allclose(responses[channel], beam.response(east, north), atol=1e-12)
        assert np.allclose(
            leakage[channel], beam.leakage_response(east, north), atol=1e-12
        )


def holography_beam_images(make_image: ImageFactory) -> list[Path]:
    """Three overlapping images named for holography beams 0 to 2."""
    return [
        make_image(
            f"SB1.field.beam{index:02d}.fits",
            150.0 + 0.3 * index,
            -30.0,
            noise=0.1,
            seed=index,
        )
        for index in range(3)
    ]


def test_holography_scale_stored(make_image: ImageFactory, tmp_path: Path) -> None:
    """Holography beams used as the cube stores them divide inherent images by their value at the centre."""
    cube = write_holography_cube(
        tmp_path / "akpb.fits", np.array([FREQUENCY_HZ, 9.5e8]), n_beams=3
    )
    with fits.open(cube, mode="update") as hdul:
        hdul[0].data *= 1.05
    images = holography_beam_images(make_image)

    def mosaicked(scale: Literal["centre", "stored"]) -> np.ndarray:
        options = MosaicOptions(
            primary_beam=PrimaryBeamOptions(
                pb_model="holography",
                holography_cube=cube,
                holography_scale=scale,
                pb_cutoff=0.1,
            ),
            weight=WeightOptions(image_weight="none"),
        )
        names = mosaic(images, tmp_path / scale, options)
        return np.array(fits.getdata(names.image))

    centre, stored = mosaicked("centre"), mosaicked("stored")
    # The beams peak at their centre, so the cutoff keeps the same pixels
    good = np.isfinite(centre)
    assert np.mean(good) > 0.3
    assert np.array_equal(good, np.isfinite(stored))
    assert np.allclose(stored[good] * 1.05, centre[good], rtol=1e-5)


def test_beam_numbers_name_the_holography_beams(
    make_image: ImageFactory, tmp_path: Path
) -> None:
    """Images whose names hold no beam number take their holography beams from ``beam_numbers``."""
    cube = write_holography_cube(
        tmp_path / "akpb.fits", np.array([FREQUENCY_HZ, 9.5e8]), n_beams=3
    )
    # Beams of different heights, used as stored, so the mosaic depends on which beam each image is given
    with fits.open(cube, mode="update") as hdul:
        hdul[0].data *= np.array([1.0, 1.1, 1.2])[:, None, None, None, None]
    named = holography_beam_images(make_image)
    unnamed = []
    for index, image in enumerate(named):
        copy = tmp_path / f"image{index}.fits"
        shutil.copy(image, copy)
        unnamed.append(copy)
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(
            pb_model="holography",
            holography_cube=cube,
            holography_scale="stored",
            pb_cutoff=0.1,
        ),
        weight=WeightOptions(image_weight="none"),
    )

    def mosaicked(
        base: str, images: list[Path], beam_numbers: list[int] | None = None
    ) -> np.ndarray:
        names = mosaic(images, tmp_path / base, options, beam_numbers=beam_numbers)
        return np.array(fits.getdata(names.image))

    by_name = mosaicked("named", named)
    assert np.array_equal(
        mosaicked("numbered", unnamed, [0, 1, 2]), by_name, equal_nan=True
    )
    assert not np.allclose(
        mosaicked("swapped", unnamed, [2, 1, 0]), by_name, equal_nan=True
    )
    with pytest.raises(ValueError, match="needs a holography cube and a beam number"):
        mosaicked("unnumbered", unnamed)
    with pytest.raises(ValueError, match="2 beam numbers for 3 images"):
        mosaicked("short", unnamed, [0, 1])


def test_mosaic_logs_its_early_stages(
    make_image: ImageFactory, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The stages before any tile is made are logged as they start, and timed in the time the mosaic took."""
    cube = write_holography_cube(
        tmp_path / "akpb.fits", np.array([FREQUENCY_HZ, 9.5e8]), n_beams=3
    )
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(
            pb_model="holography", holography_cube=cube, pb_cutoff=0.1
        ),
        weight=WeightOptions(image_weight="std"),
    )
    with caplog.at_level(logging.INFO, logger=logger.name):
        mosaic(holography_beam_images(make_image), tmp_path / "field", options)
    messages = [record.message for record in caplog.records]
    for expected in (
        "Reading the headers of 3 images",
        "Reading the holography beams of 3 images, 3 beams of akpb.fits",
        "Measuring the noise of 3 images",
    ):
        assert expected in messages
    assert any(
        message.startswith("Measured the noise of 3 images in ") for message in messages
    )
    assert any(
        message.startswith("Wrote ") and " tiles: " in message for message in messages
    )
    took = next(message for message in messages if message.startswith("Took "))
    for stage in (
        "reading headers",
        "reading beams",
        "measuring noise",
        "making the mosaic",
    ):
        assert stage in took

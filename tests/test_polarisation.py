from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from patchwork.coordinates import east_north, offset_vectors, ra_dec, unit_vectors
from patchwork.frame import new_wcs
from patchwork.io import celestial_wcs, read_image_info
from patchwork.mosaic import mosaic
from patchwork.options import (
    FrameOptions,
    MosaicOptions,
    PolarisationOptions,
    PrimaryBeamOptions,
    RunOptions,
    WeightOptions,
)
from patchwork.polarisation import (
    NORTH,
    LinearFrame,
    frame_angle,
    header_frame,
    remove_leakage,
    rotate_linear,
    write_frame_keywords,
)
from patchwork.primary_beam import direction_cosines, gaussian_beam

from .conftest import FREQUENCY_HZ, image_header

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    LeakageOnSky = Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ]

SOURCES = [
    # RA, Dec, polarised flux, polarisation angle east of local north (deg)
    (0.0, -87.0, 1.0, 30.0),
    (100.0, -88.2, 0.6, -50.0),
    (230.0, -87.5, 0.8, 75.0),
    (60.0, -89.4, 0.5, 10.0),
]


def polarised_sky(
    ra: NDArray[np.float64], dec: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Q + iU of extended sources, referenced to local north."""
    sky = np.zeros(np.shape(ra), dtype=complex)
    for source_ra, source_dec, flux, angle in SOURCES:
        east, north = direction_cosines(source_ra, source_dec, ra, dec)
        profile = np.exp(
            -4 * np.log(2) * (np.degrees(np.hypot(east, north)) / 0.3) ** 2
        )
        sky += flux * profile * np.exp(2j * np.radians(angle))
    return sky


def write_polarised_beam(
    tmp_path: Path,
    name: str,
    centre: tuple[float, float],
    frame_centre: tuple[float, float] | None,
    keywords: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Q and U images of one beam, referenced to north at ``frame_centre`` (or its own centre) as an imager makes them, with any extra header ``keywords``."""
    header = image_header(*centre, size=240, cdelt=0.02)
    header.update(keywords or {})
    wcs = celestial_wcs(header)
    y, x = np.mgrid[:240, :240]
    ra, dec = wcs.pixel_to_world_values(x, y)
    vectors = unit_vectors(ra, dec)
    reference = frame_centre or centre
    sky = polarised_sky(ra, dec) * np.exp(-2j * frame_angle(reference, vectors))
    sky *= gaussian_beam(FREQUENCY_HZ).response(*direction_cosines(*centre, ra, dec))

    paths = []
    for stokes, plane in ((2, sky.real), (3, sky.imag)):
        header["CRVAL4"] = stokes
        path = tmp_path / f"{name}.{'qu'[stokes - 2]}.fits"
        fits.writeto(path, plane[None, None].astype(np.float32), header)
        paths.append(path)
    return paths[0], paths[1]


def frame_options(frame_centre: tuple[float, float] | None) -> PolarisationOptions:
    """The options of beams written by ``write_polarised_beam`` with this ``frame_centre``."""
    if frame_centre is None:
        return PolarisationOptions(input_frame="crval")
    return PolarisationOptions(frame_centre=frame_centre)


def read_complex(
    q_path: Path, u_path: Path
) -> tuple[NDArray[np.complex128], fits.Header]:
    with fits.open(q_path) as q_hdul, fits.open(u_path) as u_hdul:
        return q_hdul[0].data[0, 0] + 1j * u_hdul[0].data[0, 0], q_hdul[0].header.copy()


def angle_error_deg(
    got: NDArray[np.complex128], expected: NDArray[np.complex128]
) -> float:
    """The largest difference in polarisation angle, in degrees."""
    return float(np.degrees(np.max(np.abs(np.angle(got / expected)))) / 2)


def test_frame_angle_matches_astropy() -> None:
    rng = np.random.default_rng(2)
    for _ in range(50):
        reference = (rng.uniform(0, 360), rng.uniform(-89, -20))
        point = (
            reference[0] + rng.uniform(-30, 30),
            reference[1] + rng.uniform(-0.9, 0.9),
        )
        # A small step along the frame direction, whose position angle astropy measures
        _, north = east_north(*reference)
        vector = unit_vectors(*point)
        direction = north - (north @ vector) * vector
        moved = vector + 1e-6 * direction / np.linalg.norm(direction)
        moved_ra = np.degrees(np.arctan2(moved[1], moved[0]))
        moved_dec = np.degrees(np.arcsin(moved[2] / np.linalg.norm(moved)))
        expected = (
            SkyCoord(*point, unit="deg")
            .position_angle(SkyCoord(moved_ra, moved_dec, unit="deg"))
            .rad
        )
        got = frame_angle(reference, vector[:, None])[0]
        assert np.isclose(np.angle(np.exp(1j * (got - expected))), 0, atol=1e-7)

    assert np.isclose(
        frame_angle((10.0, -40.0), unit_vectors(10.0, -40.0)[:, None])[0], 0
    )
    assert np.isfinite(frame_angle((10.0, -80.0), unit_vectors(0.0, -90.0)[:, None])[0])


def test_rotate_linear() -> None:
    q, u = np.array([1.0]), np.array([0.0])
    rotated_q, rotated_u = rotate_linear(q, u, np.radians(30))
    assert np.isclose(np.degrees(np.arctan2(rotated_u, rotated_q))[0] / 2, 30)
    assert np.isclose(np.hypot(rotated_q, rotated_u)[0], 1)


def test_remove_leakage() -> None:
    stokes_i = np.array([2.0])
    leakage = np.array([[0.1], [0.2], [0.3]])
    q, v = remove_leakage((np.array([1.0]), np.array([1.0])), (2, 4), stokes_i, leakage)
    assert np.isclose(q[0], 0.8)
    assert np.isclose(v[0], 0.4)
    with pytest.raises(ValueError, match="Stokes"):
        remove_leakage((np.array([1.0]),), (1,), stokes_i, leakage)


@pytest.mark.parametrize("frame_centre", [None, (0.0, -89.0)])
def test_mosaic_around_the_pole(
    tmp_path: Path, frame_centre: tuple[float, float] | None
) -> None:
    """Q and U of beams around the SCP, each in its own frame, combine into the true sky."""
    beams = [
        write_polarised_beam(tmp_path, f"beam{index:02d}", centre, frame_centre)
        for index, centre in enumerate(
            [(0.0, -87.5), (120.0, -87.5), (240.0, -87.5), (0.0, -90.0)]
        )
    ]
    q_images, u_images = [beam[0] for beam in beams], [beam[1] for beam in beams]
    weights = WeightOptions(image_weight="none")
    frame = FrameOptions(frame_type="fit")

    options = MosaicOptions(
        frame=frame,
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.2),
        weight=weights,
        polarisation=frame_options(frame_centre),
    )
    names = mosaic([*q_images, *u_images], tmp_path / "rotated", options)
    assert [path.name for path in names.images] == [
        "rotated.q.mosaic.fits",
        "rotated.u.mosaic.fits",
    ]
    got, header = read_complex(*names.images)
    assert header["CRVAL4"] == 2
    assert fits.getheader(names.images[1])["CRVAL4"] == 3
    assert fits.getheader(names.weight)["CRVAL4"] == 2

    wcs = celestial_wcs(header)
    y, x = np.mgrid[: header["NAXIS2"], : header["NAXIS1"]]
    ra, dec = wcs.pixel_to_world_values(x, y)
    expected = polarised_sky(ra, dec)
    bright = np.isfinite(got) & (np.abs(expected) > 0.2)
    assert np.sum(bright) > 500
    assert np.max(np.abs(got[bright] - expected[bright])) < 5e-3
    assert angle_error_deg(got[bright], expected[bright]) < 0.5

    # Combining Q and U as linmos does, as though already referenced to local north, is badly wrong here
    linmos = options._replace(polarisation=PolarisationOptions(input_frame="north"))
    wrong, _ = read_complex(
        *mosaic([*q_images, *u_images], tmp_path / "linmos", linmos).images
    )
    assert angle_error_deg(wrong[bright], expected[bright]) > 10

    # Referenced to the north of the frame centre, the mosaic is the true sky turned into that frame
    centred = options._replace(
        polarisation=frame_options(frame_centre).model_copy(
            update={"output_frame": "centre"}
        )
    )
    turned, _ = read_complex(
        *mosaic([*q_images, *u_images], tmp_path / "centred", centred).images
    )
    to_centre = np.exp(-2j * frame_angle(tuple(wcs.wcs.crval), unit_vectors(ra, dec)))
    assert np.max(np.abs(turned[bright] - (expected * to_centre)[bright])) < 5e-3


def test_polarisation_inputs_are_checked(tmp_path: Path) -> None:
    q_image, u_image = write_polarised_beam(tmp_path, "beam00", (0.0, -87.5), None)
    with pytest.raises(ValueError, match="1 weights for 2 images"):
        mosaic([q_image, u_image], tmp_path / "short", weights=[q_image])
    with pytest.raises(ValueError, match="Stokes I image"):
        mosaic(
            [q_image, u_image],
            tmp_path / "leakage",
            MosaicOptions(polarisation=PolarisationOptions(remove_leakage=True)),
        )


def write_leaky_cube(path: Path, alpha: float) -> tuple[Path, LeakageOnSky]:
    """A holography cube of Gaussian beams with an analytic leakage, and that leakage as seen on the sky."""
    size, cdelt = 101, 0.0345
    centre = size // 2
    y, x = np.mgrid[:size, :size]
    plane_x = np.radians((x - centre) * -cdelt)
    plane_y = np.radians((y - centre) * cdelt)

    def leakage(
        px: NDArray[np.float64], py: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        scale = np.radians(1.0)
        return np.array(
            [
                0.05 * px / scale + 0.01,
                0.03 * py / scale - 0.02,
                0.02 * (px**2 + py**2) / scale**2,
            ]
        )

    fwhm = gaussian_beam(FREQUENCY_HZ).fwhm_rad
    stokes_i = np.exp(-4 * np.log(2) * (plane_x**2 + plane_y**2) / fwhm**2)
    planes = np.concatenate([stokes_i[None], leakage(plane_x, plane_y)])[None, :, None]
    data = np.repeat(np.repeat(planes, 2, axis=2), 2, axis=0)
    header = fits.Header()
    for axis, (ctype, crval, step, crpix) in enumerate(
        [
            ("RA---SIN", 0.0, -cdelt, centre + 1),
            ("DEC--SIN", 0.0, cdelt, centre + 1),
            ("FREQ", FREQUENCY_HZ, 1e7, 1),
        ],
        start=1,
    ):
        (
            header[f"CTYPE{axis}"],
            header[f"CRVAL{axis}"],
            header[f"CDELT{axis}"],
            header[f"CRPIX{axis}"],
        ) = (
            ctype,
            crval,
            step,
            crpix,
        )
    fits.writeto(path, data.astype(np.float32), header)

    def on_sky(
        east: NDArray[np.float64], north: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The normalised leakage at sky offsets, with Q and U turned into the frame of the image."""
        px = east * np.cos(alpha) - north * np.sin(alpha)
        py = east * np.sin(alpha) + north * np.cos(alpha)
        q, u, v = leakage(px, py) - leakage(np.zeros(()), np.zeros(()))[:, None, None]
        return np.array([*rotate_linear(q, u, alpha), v])

    return path, on_sky


def test_leakage_removal(tmp_path: Path) -> None:
    alpha = 0.3
    cube, leakage_on_sky = write_leaky_cube(tmp_path / "cube.fits", alpha)
    centres = [(150.0, -30.0), (151.2, -30.0)]
    q_images: list[Path] = []
    u_images: list[Path] = []
    v_images: list[Path] = []
    i_images: list[Path] = []
    for index, centre in enumerate(centres):
        header = image_header(*centre, size=200, cdelt=0.01)
        wcs = celestial_wcs(header)
        y, x = np.mgrid[:200, :200]
        ra, dec = wcs.pixel_to_world_values(x, y)
        east, north = direction_cosines(*centre, ra, dec)
        primary_beam = gaussian_beam(FREQUENCY_HZ).response(east, north)
        stokes_i = 2.0 * primary_beam
        leakage = leakage_on_sky(east, north)
        true_q, true_u, true_v = 0.1, -0.05, 0.02
        # Q and U carry no frame rotation here, so only the leakage is tested
        planes = {
            "q": (2, true_q * primary_beam + leakage[0] * stokes_i),
            "u": (3, true_u * primary_beam + leakage[1] * stokes_i),
            "v": (4, true_v * primary_beam + leakage[2] * stokes_i),
            "i": (1, stokes_i),
        }
        for name, (stokes, plane) in planes.items():
            header["CRVAL4"] = stokes
            path = tmp_path / f"field.beam{index:02d}.{name}.fits"
            fits.writeto(path, plane[None, None].astype(np.float32), header)
            {"q": q_images, "u": u_images, "v": v_images, "i": i_images}[name].append(
                path
            )

    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(
            pb_model="holography",
            holography_cube=cube,
            holography_alpha_rad=alpha,
            pb_cutoff=0.2,
        ),
        weight=WeightOptions(image_weight="none"),
        polarisation=PolarisationOptions(input_frame="north", remove_leakage=True),
    )
    names = mosaic(
        [*q_images, *u_images], tmp_path / "pol", options, stokes_i_images=i_images
    )
    got, _ = read_complex(*names.images)
    good = np.isfinite(got)
    assert np.mean(good) > 0.3
    assert np.allclose(got[good], 0.1 - 0.05j, atol=2e-3)

    v_names = mosaic(v_images, tmp_path / "v", options, stokes_i_images=i_images)
    stokes_v = fits.getdata(v_names.image)[0, 0]
    assert np.allclose(stokes_v[np.isfinite(stokes_v)], 0.02, atol=2e-3)

    # Without removing it the leakage is plain to see
    kept = options._replace(polarisation=PolarisationOptions(input_frame="north"))
    leaky, _ = read_complex(
        *mosaic([*q_images, *u_images], tmp_path / "leaky", kept).images
    )
    assert np.max(np.abs(leaky[good] - (0.1 - 0.05j))) > 0.02

    assert read_image_info(i_images[0]).stokes == 1


@pytest.mark.parametrize("dec", [-30.0, -80.0, -90.0])
def test_grid_frame_is_projected_north(dec: float) -> None:
    """The y axis of a SIN image follows north at its centre, as an imager references Q and U."""
    wcs = new_wcs((40.0, dec), 0.01, "SIN")
    wcs.wcs.crpix = [151, 151]
    wcs.wcs.set()
    y, x = np.mgrid[0:301:20, 0:301:20].astype(float)
    ra, dec_grid = wcs.pixel_to_world_values(x, y)
    ra_up, dec_up = wcs.pixel_to_world_values(x, y + 0.01)
    grid_up = (
        SkyCoord(ra, dec_grid, unit="deg")
        .position_angle(SkyCoord(ra_up, dec_up, unit="deg"))
        .rad
    )
    frame = frame_angle((40.0, dec), unit_vectors(ra, dec_grid))
    difference = np.degrees(np.abs(np.angle(np.exp(1j * (grid_up - frame)))))
    assert np.max(difference) < 0.05


def test_leakage_errors_come_before_any_work(tmp_path: Path) -> None:
    """Leakage that cannot be removed is an error before anything is convolved or written."""
    rng = np.random.default_rng(0)
    no_stokes, polarised = [], []
    for beam, ra in enumerate((150.0, 150.6)):
        header = image_header(ra, -30.0, size=50, cdelt=0.01)
        header["BMAJ"] = header["BMIN"] = (15 + 5 * beam) / 3600
        for code, name in ((1, "i"), (2, "q")):
            header["CRVAL4"] = code
            path = tmp_path / f"field.beam{beam:02d}.{name}.fits"
            fits.writeto(
                path, rng.normal(size=(1, 1, 50, 50)).astype(np.float32), header
            )
            polarised.append(path)
        flat = header.copy()
        for key in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4"):
            del flat[key]
        flat["NAXIS"] = 3
        path = tmp_path / f"flat.beam{beam:02d}.fits"
        fits.writeto(path, rng.normal(size=(1, 50, 50)).astype(np.float32), flat)
        no_stokes.append(path)

    leakage = PolarisationOptions(remove_leakage=True)
    with pytest.raises(ValueError, match="no STOKES axis"):
        mosaic(
            no_stokes,
            tmp_path / "flat",
            MosaicOptions(polarisation=leakage),
            stokes_i_images=polarised[::2],
        )
    with pytest.raises(ValueError, match="needs the holography primary beam"):
        mosaic(
            polarised,
            tmp_path / "gaussian",
            MosaicOptions(polarisation=leakage),
            RunOptions(convolve="natural"),
        )
    assert not list(tmp_path.glob("flat.mosaic*"))
    assert not list(tmp_path.glob("*.conv.fits"))


def footprint(
    boresight: tuple[float, float], n_side: int, pitch_deg: float = 1.05
) -> list[tuple[float, float]]:
    """The centres of a square footprint of beams about a boresight, as ASKAP's."""
    steps = np.sin(np.radians((np.arange(n_side) - (n_side - 1) / 2) * pitch_deg))
    east, north = np.meshgrid(steps, steps)
    ra, dec = ra_dec(offset_vectors(boresight, east.ravel(), north.ravel()))
    return [(float(a), float(b)) for a, b in zip(ra, dec, strict=True)]


def sky_error_deg(names: Any) -> float:
    """The largest error in polarisation angle of a mosaic of ``SOURCES``, where they are bright."""
    got, header = read_complex(*names.images)
    wcs = celestial_wcs(header)
    y, x = np.mgrid[: header["NAXIS2"], : header["NAXIS1"]]
    expected = polarised_sky(*wcs.pixel_to_world_values(x, y))
    bright = np.isfinite(got) & (np.abs(expected) > 0.2)
    assert np.sum(bright) > 500
    return angle_error_deg(got[bright], expected[bright])


BEAM_OPTIONS = MosaicOptions(
    frame=FrameOptions(frame_type="fit"),
    primary_beam=PrimaryBeamOptions(pb_cutoff=0.2),
    weight=WeightOptions(image_weight="none"),
)
"""Options for mosaicking beams written by ``write_polarised_beam``"""


def test_askap_beams_need_their_boresight(tmp_path: Path) -> None:
    """Every beam of an ASKAP observation is referenced to its boresight, and its mosaic is right only when told so."""
    boresight = (0.0, -88.0)
    images = [
        path
        for index, centre in enumerate(footprint(boresight, 3))
        for path in write_polarised_beam(
            tmp_path, f"beam{index:02d}", centre, boresight
        )
    ]
    with pytest.raises(ValueError, match="--frame-centre"):
        mosaic(images, tmp_path / "unstated", BEAM_OPTIONS)
    assert not list(tmp_path.glob("unstated*"))

    names = mosaic(
        images,
        tmp_path / "boresight",
        BEAM_OPTIONS._replace(polarisation=PolarisationOptions(frame_centre=boresight)),
    )
    assert sky_error_deg(names) < 0.5
    # Each beam referenced to its own centre, as an ASKAP beam is not, is far out
    beams = mosaic(
        images,
        tmp_path / "beams",
        BEAM_OPTIONS._replace(polarisation=PolarisationOptions(input_frame="crval")),
    )
    assert sky_error_deg(beams) > 5


def test_keywords_give_each_observation_its_boresight(tmp_path: Path) -> None:
    """The beams of two observations mosaic in one pass when their keywords give each one's boresight."""
    images: list[Path] = []
    for observation, boresight in enumerate([(0.0, -88.0), (120.0, -88.0)]):
        keywords = {"POLFRAME": "FIXED", "POLRA": boresight[0], "POLDEC": boresight[1]}
        for index, centre in enumerate(footprint(boresight, 2)):
            images += write_polarised_beam(
                tmp_path,
                f"sb{observation}.beam{index:02d}",
                centre,
                boresight,
                keywords,
            )
    assert sky_error_deg(mosaic(images, tmp_path / "both", BEAM_OPTIONS)) < 0.5

    # One boresight for both observations contradicts the keywords of the other
    one = BEAM_OPTIONS._replace(
        polarisation=PolarisationOptions(frame_centre=(0.0, -88.0))
    )
    with pytest.raises(ValueError, match=r"POLFRAME keywords of .*sb1"):
        mosaic(images, tmp_path / "one", one)


def test_mosaics_record_their_frame(tmp_path: Path) -> None:
    """A mosaic records the frame of its Q and U, so mosaicking it again needs no frame options and never turns it twice."""
    boresight = (0.0, -88.0)
    images = [
        path
        for index, centre in enumerate(footprint(boresight, 2))
        for path in write_polarised_beam(
            tmp_path, f"beam{index:02d}", centre, boresight
        )
    ]
    to_north = PolarisationOptions(frame_centre=boresight)
    north = mosaic(
        images, tmp_path / "north", BEAM_OPTIONS._replace(polarisation=to_north)
    )
    centred = mosaic(
        images,
        tmp_path / "centred",
        BEAM_OPTIONS._replace(
            polarisation=to_north.model_copy(update={"output_frame": "centre"})
        ),
    )
    for path in north.images:
        assert fits.getheader(path)["POLFRAME"] == "NORTH"
    assert "POLFRAME" not in fits.getheader(north.weight)
    centre_header = fits.getheader(centred.images[1])
    assert centre_header["POLFRAME"] == "FIXED"
    assert np.isclose(centre_header["POLRA"], centre_header["CRVAL1"])
    assert np.isclose(centre_header["POLDEC"], centre_header["CRVAL2"])

    # Each mosaic is an input on its own grid, already corrected for the primary beam
    again = MosaicOptions(
        primary_beam=PrimaryBeamOptions(pb_model="none"),
        weight=WeightOptions(image_weight="none"),
    )
    original, _ = read_complex(*north.images)
    for name, names in (("unturned", north), ("turned", centred)):
        remade, _ = read_complex(
            *mosaic(list(names.images), tmp_path / name, again).images
        )
        good = np.isfinite(original) & np.isfinite(remade)
        assert np.sum(good) > 500
        assert np.allclose(remade[good], original[good], rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="twice"):
        mosaic(
            list(north.images),
            tmp_path / "twice",
            again._replace(polarisation=to_north),
        )


def test_frame_keywords_and_options_are_checked(tmp_path: Path) -> None:
    q_image, u_image = write_polarised_beam(tmp_path, "beam00", (0.0, -87.5), None)
    with pytest.raises(ValueError, match="does not go with"):
        mosaic(
            [q_image, u_image],
            tmp_path / "both",
            MosaicOptions(
                polarisation=PolarisationOptions(
                    input_frame="north", frame_centre=(0.0, -87.5)
                )
            ),
        )
    for path, frame in ((q_image, NORTH), (u_image, LinearFrame((0.0, -87.5)))):
        with fits.open(path, mode="update") as hdul:
            write_frame_keywords(hdul[0].header, frame)
    with pytest.raises(ValueError, match="disagree"):
        mosaic([q_image, u_image], tmp_path / "mixed")


def test_header_frame() -> None:
    header = fits.Header()
    assert header_frame(header) is None
    write_frame_keywords(header, LinearFrame((370.0, -30.0)))
    assert header_frame(header) == LinearFrame((10.0, -30.0))
    write_frame_keywords(header, NORTH)
    assert header_frame(header) == NORTH
    assert "POLRA" not in header
    for cards, match in (
        ({"POLFRAME": "LOCAL"}, "NORTH or FIXED"),
        ({"POLRA": 1.0}, "no POLFRAME"),
        ({"POLFRAME": "FIXED", "POLRA": 1.0}, "not a position"),
        ({"POLFRAME": "FIXED", "POLRA": 1.0, "POLDEC": 95.0}, "not a position"),
    ):
        with pytest.raises(ValueError, match=match):
            header_frame(fits.Header(list(cards.items())), "image.fits")
    assert LinearFrame((0.0, -84.0)).matches(LinearFrame((360 - 1e-5, -84.0)))
    assert not LinearFrame((0.0, -84.0)).matches(LinearFrame((0.0, -83.99)))
    assert not NORTH.matches(LinearFrame((0.0, -84.0)))


def write_stokes_i(q_image: Path, path: Path, keywords: dict[str, Any]) -> Path:
    """A Stokes I image on the grid of a Q image, with extra header ``keywords``."""
    with fits.open(q_image) as hdul:
        header = hdul[0].header.copy()
        data = np.abs(hdul[0].data) + 1
    header["CRVAL4"] = 1
    header.update(keywords)
    fits.writeto(path, data.astype(np.float32), header)
    return path


def test_only_the_keywords_of_q_and_u_can_stop_a_mosaic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The frame of Q and U is read from them alone. Keywords on images that are not rotated only orient holography beams, so never stop a mosaic."""
    q_image, u_image = write_polarised_beam(tmp_path, "beam00", (0.0, -87.5), None)
    with fits.open(q_image, mode="update") as hdul:
        write_frame_keywords(hdul[0].header, LinearFrame((0.0, -87.5)))
    with pytest.raises(ValueError, match="does not"):
        mosaic([q_image, u_image], tmp_path / "half", BEAM_OPTIONS)
    with fits.open(u_image, mode="update") as hdul:
        write_frame_keywords(hdul[0].header, LinearFrame((0.0, -87.5)))

    unreadable = write_stokes_i(
        q_image, tmp_path / "beam00.i.fits", {"POLFRAME": "LINEAR"}
    )
    names = mosaic([unreadable, q_image, u_image], tmp_path / "iqu", BEAM_OPTIONS)
    assert fits.getheader(names.images[1])["POLFRAME"] == "NORTH"

    with caplog.at_level(logging.WARNING, logger="patchwork"):
        mosaic([unreadable], tmp_path / "unreadable", BEAM_OPTIONS)
    assert "LINEAR" in caplog.text
    assert "ignored" in caplog.text

    caplog.clear()
    fixed = write_stokes_i(
        q_image,
        tmp_path / "other.i.fits",
        {"POLFRAME": "FIXED", "POLRA": 0.0, "POLDEC": -87.5},
    )
    with caplog.at_level(logging.WARNING, logger="patchwork"):
        mosaic(
            [fixed],
            tmp_path / "options",
            BEAM_OPTIONS._replace(
                polarisation=PolarisationOptions(frame_centre=(10.0, -80.0))
            ),
        )
    assert "the options are used" in caplog.text

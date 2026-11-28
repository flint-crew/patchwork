from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from astropy.constants import c as speed_of_light
from astropy.io import fits
from distributed import Client, LocalCluster

import patchwork.mosaic
import patchwork.write
from patchwork.convolve import convolve_cube
from patchwork.coordinates import unit_vectors
from patchwork.frame import strips
from patchwork.io import (
    NO_BEAM,
    BeamShape,
    celestial_wcs,
    create_output_fits,
    frequencies_from_header,
    read_image_info,
)
from patchwork.mosaic import get_parser, mosaic, options_from_args
from patchwork.options import (
    ConvolveOptions,
    FrameOptions,
    MosaicOptions,
    PolarisationOptions,
    PrimaryBeamOptions,
    ResampleOptions,
    RunOptions,
    WeightOptions,
)
from patchwork.pipeline.flow import run_pipeline
from patchwork.plan import check_spectral_axes, group_inputs, output_names
from patchwork.polarisation import frame_angle
from patchwork.primary_beam import direction_cosines, gaussian_beam
from patchwork.tile import PlaneWeight
from patchwork.weights import measure_channel_weights, measure_weights

from .conftest import image_header, strided_weights

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from patchwork.convolve import ConvolveBackend

FREQUENCIES = 9.0e8 + 2e7 * np.arange(5)
IN_BEAM_FRAME = PolarisationOptions(input_frame="crval")
"""The frame ``write_beam_cubes`` references Q and U to"""
SOURCES = [
    # RA, Dec, Stokes I at 900 MHz, fractional polarisation, angle at 900 MHz (deg), RM (rad/m^2)
    (0.0, -87.0, 1.0, 0.3, 30.0, 20.0),
    (100.0, -88.2, 0.8, 0.5, -50.0, -40.0),
    (230.0, -87.5, 1.2, 0.2, 75.0, 5.0),
]


def true_sky(
    ra: NDArray[np.float64], dec: NDArray[np.float64], frequency: float
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Stokes I and Q + iU (referenced to local north) of the sources at one frequency."""
    stokes_i = np.zeros(np.shape(ra))
    linear = np.zeros(np.shape(ra), dtype=complex)
    wavelength_squared = (speed_of_light.value / frequency) ** 2
    for source_ra, source_dec, flux, fraction, angle, rm in SOURCES:
        east, north = direction_cosines(source_ra, source_dec, ra, dec)
        profile = np.exp(
            -4 * np.log(2) * (np.degrees(np.hypot(east, north)) / 0.3) ** 2
        )
        flux_now = flux * (frequency / FREQUENCIES[0]) ** -0.7 * profile
        stokes_i += flux_now
        linear += (
            fraction
            * flux_now
            * np.exp(2j * (np.radians(angle) + rm * wavelength_squared))
        )
    return stokes_i, linear


def cube_header(
    centre: tuple[float, float], stokes: int, size: int, casa_order: bool
) -> fits.Header:
    header = image_header(*centre, size=size, cdelt=0.02, frequency_hz=FREQUENCIES[0])
    header["NAXIS3"] = len(FREQUENCIES)
    header["CDELT3"] = FREQUENCIES[1] - FREQUENCIES[0]
    header["CRVAL4"] = stokes
    if casa_order:
        for key in ("NAXIS", "CTYPE", "CRVAL", "CDELT", "CRPIX"):
            header[f"{key}3"], header[f"{key}4"] = header[f"{key}4"], header[f"{key}3"]
    return header


def write_beam_cubes(
    tmp_path: Path,
    name: str,
    centre: tuple[float, float],
    size: int = 160,
    casa_order: bool = False,
    blank_channel: int | None = None,
    beams: tuple[BeamShape, ...] | None = None,
) -> dict[str, Path]:
    """I, Q and U cubes of one beam, still carrying the Gaussian primary beam, with Q and U in the frame of the beam."""
    header = cube_header(centre, 1, size, casa_order)
    wcs = celestial_wcs(header)
    y, x = np.mgrid[:size, :size]
    ra, dec = wcs.pixel_to_world_values(x, y)
    to_frame = np.exp(-2j * frame_angle(centre, unit_vectors(ra, dec)))
    planes: dict[str, list[NDArray[np.float64]]] = {"i": [], "q": [], "u": []}
    for channel, frequency in enumerate(FREQUENCIES):
        primary_beam = gaussian_beam(frequency).response(
            *direction_cosines(*centre, ra, dec)
        )
        stokes_i, linear = true_sky(ra, dec, frequency)
        linear = linear * to_frame * primary_beam
        blank = channel == blank_channel
        planes["i"].append(
            np.full_like(stokes_i, np.nan) if blank else stokes_i * primary_beam
        )
        planes["q"].append(np.full_like(stokes_i, np.nan) if blank else linear.real)
        planes["u"].append(np.full_like(stokes_i, np.nan) if blank else linear.imag)

    channel_beams = beams or tuple(
        NO_BEAM if channel == blank_channel else BeamShape(36.0, 36.0, 0.0)
        for channel in range(len(FREQUENCIES))
    )
    paths = {}
    for stokes, (key, data) in enumerate(planes.items(), start=1):
        stokes_header = cube_header(centre, stokes, size, casa_order)
        path = create_output_fits(
            tmp_path / f"{name}.{key}.fits", stokes_header, channel_beams
        )
        cube = np.array(data, dtype=np.float32)
        with fits.open(path, mode="update") as hdul:
            hdul[0].data[...] = cube[:, None] if casa_order else cube[None]
        paths[key] = path
    return paths


BEAM_CENTRES = [(0.0, -87.6), (120.0, -87.6), (240.0, -87.6), (0.0, -90.0)]


@pytest.fixture
def beam_cubes(tmp_path: Path) -> list[dict[str, Path]]:
    return [
        write_beam_cubes(tmp_path, f"field.beam{index:02d}", centre)
        for index, centre in enumerate(BEAM_CENTRES)
    ]


def cube_options_for_test(**kwargs: object) -> MosaicOptions:
    return MosaicOptions(
        frame=FrameOptions(frame_type="fit"),
        weight=WeightOptions(image_weight="none"),
        **{
            "primary_beam": PrimaryBeamOptions(pb_cutoff=0.2),
            # The Q and U of each beam are referenced to north at its own centre
            "polarisation": IN_BEAM_FRAME,
            **kwargs,
        },  # type: ignore[arg-type]
    )


def read_cube(path: Path) -> tuple[NDArray[np.float64], fits.Header]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        return data.reshape(-1, *data.shape[-2:]), hdul[0].header.copy()


def expected_cubes(
    header: fits.Header,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    wcs = celestial_wcs(header)
    y, x = np.mgrid[: header["NAXIS2"], : header["NAXIS1"]]
    ra, dec = wcs.pixel_to_world_values(x, y)
    frequencies = list(frequencies_from_header(header))
    expected = [true_sky(ra, dec, frequency) for frequency in frequencies]
    return np.array([stokes_i for stokes_i, _ in expected]), np.array(
        [linear for _, linear in expected]
    )


def weight_at(path: Path, position: tuple[float, float]) -> NDArray[np.float64]:
    """The weight of each channel at a sky position."""
    weight, header = read_cube(path)
    x, y = celestial_wcs(header).world_to_pixel_values(*position)
    return weight[:, round(float(y)), round(float(x))]


def all_cubes(beam_cubes: list[dict[str, Path]]) -> list[Path]:
    return [path for beam in beam_cubes for path in beam.values()]


def test_iqu_cube_mosaic_recovers_the_sky(
    beam_cubes: list[dict[str, Path]], tmp_path: Path
) -> None:
    names = mosaic(
        all_cubes(beam_cubes),
        tmp_path / "field",
        cube_options_for_test(),
        max_workers=2,
    )
    assert [path.name for path in names.images] == [
        "field.i.cube.fits",
        "field.q.cube.fits",
        "field.u.cube.fits",
    ]
    assert [path.name for path in names.weights] == [
        "field.i.weight.cube.fits",
        "field.qu.weight.cube.fits",
    ]

    stokes_i, header = read_cube(names.images[0])
    q, q_header = read_cube(names.images[1])
    u, _ = read_cube(names.images[2])
    assert header["NAXIS3"] == len(FREQUENCIES)
    assert q_header["CRVAL4"] == 2
    # Each weight is labelled with the first Stokes parameter of its group
    assert [fits.getheader(path)["CRVAL4"] for path in names.weights] == [1, 2]
    assert np.isclose(header["BMAJ"], 36 / 3600)

    expected_i, expected_linear = expected_cubes(header)
    bright = np.isfinite(stokes_i) & (expected_i > 0.2)
    assert np.sum(bright) > 1000
    assert np.max(np.abs(stokes_i - expected_i)[bright]) < 3e-3
    linear = q + 1j * u
    polarised = bright & (np.abs(expected_linear) > 0.1)
    assert np.max(np.abs(linear - expected_linear)[polarised]) < 3e-3
    # The angle turns with frequency through the Faraday rotation, channel by channel
    assert (
        np.degrees(
            np.max(np.abs(np.angle(linear[polarised] / expected_linear[polarised])))
        )
        / 2
        < 0.5
    )


def test_images_mosaic_as_one_channel_of_a_cube(tmp_path: Path) -> None:
    """Images are mosaicked exactly as the channel of a cube they are taken from."""
    beams = [
        write_beam_cubes(tmp_path, f"beam{index:02d}", centre, size=100)
        for index, centre in enumerate(BEAM_CENTRES[:2])
    ]
    single = []
    for beam in beams:
        for path in beam.values():
            with fits.open(path) as hdul:
                header = hdul[0].header.copy()
                header["NAXIS3"] = 1
                header.remove("CASAMBM", ignore_missing=True)
                BeamShape(36.0, 36.0, 0.0).to_header(header)
                plane = hdul[0].data[:, :1]
            single.append(tmp_path / f"single.{path.stem}.fits")
            fits.writeto(single[-1], plane, header)

    options = cube_options_for_test()
    cube = mosaic(all_cubes(beams), tmp_path / "cube", options)
    images = mosaic(single, tmp_path / "images", options)
    assert [path.name for path in images.images] == [
        "images.i.mosaic.fits",
        "images.q.mosaic.fits",
        "images.u.mosaic.fits",
    ]
    for cube_path, image_path in zip(cube.images, images.images, strict=True):
        assert np.array_equal(
            read_cube(cube_path)[0][:1], read_cube(image_path)[0], equal_nan=True
        )


def test_channel_subset_in_casa_order(tmp_path: Path) -> None:
    beams = [
        write_beam_cubes(
            tmp_path, f"beam{index:02d}", centre, size=100, casa_order=True
        )
        for index, centre in enumerate(BEAM_CENTRES[:2])
    ]
    names = mosaic(
        all_cubes(beams),
        tmp_path / "subset",
        cube_options_for_test(),
        RunOptions(channels=(1, 4), channel_chunk=2),
    )
    data, header = read_cube(names.images[0])
    assert data.shape[0] == 3
    assert header["CTYPE4"] == "FREQ"
    assert np.allclose(list(frequencies_from_header(header)), FREQUENCIES[1:4])
    expected_i, _ = expected_cubes(header)
    bright = np.isfinite(data) & (expected_i > 0.2)
    assert np.max(np.abs(data - expected_i)[bright]) < 3e-3


def test_blank_channel_is_left_out(tmp_path: Path) -> None:
    beams = [
        write_beam_cubes(
            tmp_path, "beam00", BEAM_CENTRES[0], size=100, blank_channel=2
        ),
        write_beam_cubes(tmp_path, "beam01", BEAM_CENTRES[1], size=100),
    ]
    names = mosaic(all_cubes(beams), tmp_path / "blank", cube_options_for_test())
    # At its own centre beam 00 gives most of the weight, but none in its blank channel
    weight = weight_at(names.weights[0], BEAM_CENTRES[0])
    assert weight[1] > 0.9
    assert not weight[2] > 0.5
    stokes_i, header = read_cube(names.images[0])
    expected_i, _ = expected_cubes(header)
    bright = np.isfinite(stokes_i) & (expected_i > 0.2)
    assert np.max(np.abs(stokes_i - expected_i)[bright]) < 3e-3


def test_weights_logs_and_cubes(tmp_path: Path) -> None:
    beams = [
        write_beam_cubes(tmp_path, f"beam{index:02d}", centre, size=100)
        for index, centre in enumerate(BEAM_CENTRES[:2])
    ]
    cubes = all_cubes(beams)
    logs, weight_cubes = [], []
    for index, path in enumerate(cubes):
        weights = 1 + index + 0.1 * np.arange(len(FREQUENCIES))
        log = path.with_suffix(".txt")
        log.write_text(
            "#Channel Weight\n"
            + "".join(f"{channel} {weight}\n" for channel, weight in enumerate(weights))
        )
        logs.append(log)
        info = read_image_info(path)
        weight_cube = create_output_fits(path.with_suffix(".weight.fits"), info.header)
        with fits.open(weight_cube, mode="update") as hdul:
            hdul[0].data[...] = weights[None, :, None, None]
        weight_cubes.append(weight_cube)

    options = MosaicOptions(
        frame=FrameOptions(frame_type="fit"),
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.2),
        polarisation=IN_BEAM_FRAME,
    )
    from_logs = mosaic(cubes, tmp_path / "logs", options, weights=logs)
    from_cubes = mosaic(cubes, tmp_path / "cubes", options, weights=weight_cubes)
    for log_path, cube_path in zip(
        (*from_logs.images, *from_logs.weights),
        (*from_cubes.images, *from_cubes.weights),
        strict=True,
    ):
        assert np.allclose(
            fits.getdata(log_path), fits.getdata(cube_path), equal_nan=True, rtol=1e-5
        )

    short = tmp_path / "short.txt"
    short.write_text("#Channel Weight\n0 1\n")
    with pytest.raises(ValueError, match="channels"):
        mosaic(cubes, tmp_path / "short", options, weights=[short] * len(cubes))


def test_grouping(
    beam_cubes: list[dict[str, Path]], caplog: pytest.LogCaptureFixture
) -> None:
    infos = [read_image_info(path) for path in all_cubes(beam_cubes)]
    weights = [PlaneWeight(values=(1.0,) * len(FREQUENCIES))] * len(infos)
    assert len(group_inputs(infos, weights)) == len(BEAM_CENTRES)

    with pytest.raises(ValueError, match="cannot be paired"):
        group_inputs([*infos, infos[0]], [*weights, weights[0]])
    with pytest.raises(ValueError, match="same Stokes"):
        group_inputs(infos[:-1], weights[:-1])
    only_qu = [info for info in infos if info.stokes != 1]
    with pytest.raises(ValueError, match="Stokes I"):
        group_inputs(only_qu, weights[: len(only_qu)], remove_leakage=True)

    # Q alone cannot be rotated, so is mosaicked as it is, as linmos does
    only_q = [info for info in infos if info.stokes == 2]
    assert len(group_inputs(only_q, weights[: len(only_q)])) == len(BEAM_CENTRES)
    assert "without the other" in caplog.text
    # Images of one Stokes parameter on one grid, such as epochs, are each an input
    assert (
        len(group_inputs([*only_q, only_q[0]], weights[: len(only_q) + 1]))
        == len(BEAM_CENTRES) + 1
    )
    # Stokes I images that are not mosaicked give the leakage to remove
    stokes_i = [info for info in infos if info.stokes == 1]
    leaky = group_inputs(
        only_qu, weights[: len(only_qu)], remove_leakage=True, stokes_i=stokes_i
    )
    assert [item.stokes_i_path for item in leaky] == [info.path for info in stokes_i]

    shifted = infos[1]._replace(
        frequencies_hz=tuple(np.array(infos[1].frequencies_hz) + 1e6)
    )
    with pytest.raises(ValueError, match="channels"):
        check_spectral_axes([infos[0], shifted])


def test_names() -> None:
    names = output_names("base", (1, 2, 3, 4), cube=True, output_format="zarr")
    assert [path.name for path in names.images] == [
        "base.i.cube.zarr",
        "base.q.cube.zarr",
        "base.u.cube.zarr",
        "base.v.cube.zarr",
    ]
    assert [path.name for path in names.weights] == [
        "base.i.weight.cube.zarr",
        "base.qu.weight.cube.zarr",
        "base.v.weight.cube.zarr",
    ]
    assert output_names("base", (0,), cube=True).images[0].name == "base.cube.fits"
    images = output_names("base", (2,), noise=True)
    assert [
        path.name for path in (*images.images, *images.weights, *images.noises)
    ] == [
        "base.q.mosaic.fits",
        "base.q.weight.fits",
        "base.q.noise.fits",
    ]
    assert output_names("base", (0,)).weights[0].name == "base.weight.fits"


needs_convolve_rs = pytest.mark.skipif(
    importlib.util.find_spec("convolve_rs") is None, reason="needs convolve-rs"
)


@pytest.mark.parametrize(
    "backend", ["scipy", pytest.param("rust", marks=needs_convolve_rs)]
)
@pytest.mark.parametrize("mode", ["natural", "total"])
def test_convolve_to_common_beams(
    tmp_path: Path, mode: str, backend: ConvolveBackend
) -> None:
    beams = [
        write_beam_cubes(
            tmp_path,
            "beam00",
            BEAM_CENTRES[0],
            size=100,
            beams=tuple(
                BeamShape(40 + 5 * channel, 36, 0)
                for channel in range(len(FREQUENCIES))
            ),
        ),
        write_beam_cubes(
            tmp_path,
            "beam01",
            BEAM_CENTRES[1],
            size=100,
            beams=tuple(BeamShape(36, 36, 0) for _ in range(len(FREQUENCIES))),
        ),
    ]
    names = mosaic(
        all_cubes(beams),
        tmp_path / mode,
        cube_options_for_test(),
        RunOptions(convolve=mode),  # type: ignore[arg-type]
        ConvolveOptions(convolve_backend=backend),
    )
    info = read_image_info(names.images[0])
    majors = [beam.bmaj_arcsec for beam in info.channel_beams if beam is not None]
    if mode == "total":
        assert np.allclose(majors, majors[0])
        assert majors[0] >= 60
    else:
        assert majors == sorted(majors)
        assert all(major >= 40 + 5 * channel for channel, major in enumerate(majors))
        assert info.header["CASAMBM"]
    assert not list(tmp_path.glob("*.conv.fits"))


def test_convolution_blanks_coarse_channels(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    beams = [
        write_beam_cubes(
            tmp_path,
            "beam00",
            BEAM_CENTRES[0],
            size=100,
            beams=(*(BeamShape(36, 36, 0),) * 4, BeamShape(100, 100, 0)),
        ),
        write_beam_cubes(tmp_path, "beam01", BEAM_CENTRES[1], size=100),
    ]
    names = mosaic(
        all_cubes(beams),
        tmp_path / "cutoff",
        cube_options_for_test(),
        RunOptions(convolve="natural", keep_convolved=True),
        ConvolveOptions(beam_cutoff_arcsec=60),
    )
    convolved = read_image_info(tmp_path / "beam00.i.conv.fits")
    assert not convolved.channel_beams[-1].is_usable()  # type: ignore[union-attr]
    common = read_image_info(tmp_path / "beam01.i.conv.fits").channel_beams[-1]
    assert (
        f'Common beam of channel 4 (980.000 MHz) is {common}, from 3 of 6 beams (3 larger than 60")'
        in caplog.text
    )
    weight = weight_at(names.weights[0], BEAM_CENTRES[0])
    assert weight[0] > 0.9
    assert not weight[-1] > 0.5


@pytest.mark.parametrize("channels", [(3, 9), (4, 2), (-2, 3), (2, 2)])
def test_channels_must_be_a_range_of_the_cubes(
    tmp_path: Path, channels: tuple[int, int]
) -> None:
    cubes = write_beam_cubes(tmp_path, "beam00", BEAM_CENTRES[0], size=40)
    with pytest.raises(ValueError, match="not a range"):
        mosaic(
            list(cubes.values()),
            tmp_path / "bad",
            cube_options_for_test(),
            RunOptions(channels=channels),
        )
    assert not list(tmp_path.glob("bad*"))


def test_outputs_never_replace_inputs(tmp_path: Path) -> None:
    """An output named like an input is refused before any work, even when overwriting."""
    cubes = write_beam_cubes(tmp_path, "beam00", BEAM_CENTRES[0], size=40)
    taken = cubes["i"].rename(tmp_path / "field.i.cube.fits")
    before = fits.getdata(taken).copy()
    with pytest.raises(FileExistsError, match="also an input"):
        mosaic(
            [taken, cubes["q"], cubes["u"]],
            tmp_path / "field",
            cube_options_for_test(),
            overwrite=True,
        )
    assert np.array_equal(fits.getdata(taken), before, equal_nan=True)


def test_convolve_some_channels(tmp_path: Path) -> None:
    """Only the channels mosaicked are convolved, into cubes named by those channels, matching a run over every channel."""
    beams = [
        write_beam_cubes(
            tmp_path,
            f"beam{index:02d}",
            centre,
            size=100,
            beams=tuple(
                BeamShape(40 + 5 * channel, 36, 0)
                for channel in range(len(FREQUENCIES))
            ),
        )
        for index, centre in enumerate(BEAM_CENTRES[:2])
    ]
    options = cube_options_for_test()
    every = mosaic(
        all_cubes(beams), tmp_path / "every", options, RunOptions(convolve="natural")
    )
    some = mosaic(
        all_cubes(beams),
        tmp_path / "some",
        options,
        RunOptions(convolve="natural", channels=(1, 3), keep_convolved=True),
    )

    kept = tmp_path / "beam00.i.conv.ch1-3.fits"
    usable = [
        beam is not None and beam.is_usable()
        for beam in read_image_info(kept).channel_beams
    ]
    assert usable == [False, True, True, False, False]
    # Channels outside the range are never written
    assert np.all(fits.getdata(kept)[0, [0, 3, 4]] == 0)
    every_i, _ = read_cube(every.images[0])
    some_i, _ = read_cube(some.images[0])
    assert np.allclose(some_i, every_i[1:3], equal_nan=True)


def test_measure_some_channel_weights(beam_cubes: list[dict[str, Path]]) -> None:
    path = beam_cubes[0]["i"]
    every = measure_channel_weights(path, mode="std", stride=1)
    some = measure_channel_weights(path, mode="std", stride=1, channels=slice(1, 3))
    assert some == (0.0, every[1], every[2], 0.0, 0.0)


def test_convolve_cube_in_threads(tmp_path: Path) -> None:
    """Channels convolved in threads give the cube that one thread gives, with the same beams."""
    cube = write_beam_cubes(
        tmp_path,
        "beam00",
        BEAM_CENTRES[0],
        size=60,
        beams=tuple(BeamShape(40 + 5 * channel, 36, 0) for channel in range(5)),
    )["i"]
    info = read_image_info(cube)
    targets = tuple(
        BeamShape(70, 70, 0) if channel != 2 else NO_BEAM for channel in range(5)
    )
    serial = convolve_cube(
        cube, targets, tmp_path / "serial.fits", channels=slice(1, 5)
    )
    threaded = convolve_cube(
        cube, targets, tmp_path / "threaded.fits", channels=slice(1, 5), threads=3
    )
    assert np.array_equal(fits.getdata(serial), fits.getdata(threaded), equal_nan=True)
    assert (
        read_image_info(threaded).channel_beams == read_image_info(serial).channel_beams
    )
    assert info.n_channels == 5
    assert np.all(np.isnan(fits.getdata(threaded)[0, 2]))


@needs_convolve_rs
def test_convolve_cube_rust_matches_scipy(tmp_path: Path) -> None:
    """A cube convolved by convolve-rs in threads matches scipy's, NaNs and beams included."""
    cube = write_beam_cubes(
        tmp_path,
        "beam00",
        BEAM_CENTRES[0],
        size=60,
        beams=tuple(BeamShape(40 + 5 * channel, 36, 0) for channel in range(5)),
    )["i"]
    targets = tuple(
        BeamShape(70, 70, 0) if channel != 2 else NO_BEAM for channel in range(5)
    )
    scipy = convolve_cube(cube, targets, tmp_path / "scipy.fits")
    rust = convolve_cube(
        cube, targets, tmp_path / "rust.fits", threads=3, backend="rust"
    )
    scipy_data, rust_data = fits.getdata(scipy), fits.getdata(rust)
    assert np.array_equal(np.isnan(scipy_data), np.isnan(rust_data))
    finite = np.isfinite(scipy_data)
    scale = np.max(np.abs(scipy_data[finite]))
    assert np.max(np.abs(scipy_data[finite] - rust_data[finite])) < 1e-5 * scale
    assert read_image_info(rust).channel_beams == read_image_info(scipy).channel_beams


@pytest.mark.parametrize("channel_chunk", [1, 2, 16])
def test_measure_channel_weights_in_threads(
    beam_cubes: list[dict[str, Path]], channel_chunk: int
) -> None:
    path = beam_cubes[0]["i"]
    serial = measure_channel_weights(path, mode="std", stride=2)
    threaded = measure_channel_weights(
        path, mode="std", stride=2, channel_chunk=channel_chunk, threads=3
    )
    assert threaded == serial
    some = measure_channel_weights(
        path, mode="std", stride=2, channels=slice(1, 4), threads=3
    )
    assert some == (0.0, *serial[1:4], 0.0)
    assert measure_channel_weights(path, mode="none", threads=3) == (1.0,) * 5


def test_noise_is_measured_with_the_tile_threads(
    beam_cubes: list[dict[str, Path]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every backend measures the noise of the images in the threads a tile is made with, sharing them between the tasks that run at once."""
    threads_used: list[int] = []
    measured: list[Path] = []

    def recording(paths: list[Path], **kwargs: Any) -> list[tuple[float, ...]]:
        threads_used.append(kwargs["threads"])
        measured.extend(paths)
        return measure_weights(paths, **kwargs)

    monkeypatch.setattr(patchwork.mosaic, "measure_weights", recording)
    # mosaic shares the CPUs between its workers under the name it imported
    monkeypatch.setattr(patchwork.mosaic, "available_cpus", lambda: 12)
    options = cube_options_for_test(
        resample=ResampleOptions(backend="numpy", tile_threads=0, tile_size=64)
    )._replace(weight=WeightOptions(image_weight="std"))
    mosaic(
        [cube["i"] for cube in beam_cubes],
        output_base=tmp_path / "threaded",
        options=options,
        max_workers=3,
    )
    assert threads_used == [4] * min(3, len(beam_cubes))
    assert measured == [cube["i"] for cube in beam_cubes]


def test_measure_weights_of_cubes(beam_cubes: list[dict[str, Path]]) -> None:
    """The channels of cubes are each read whole and measured, to the weights reading every ``stride``'th row alone gives."""
    paths = [cube[stokes] for cube in beam_cubes for stokes in ("i", "q")]
    for channels in (None, slice(1, 4)):
        expected = [strided_weights(path, "mad", 2, channels) for path in paths]
        for threads in (1, 3):
            assert (
                measure_weights(
                    paths, mode="mad", stride=2, channels=channels, threads=threads
                )
                == expected
            )


def test_noise_of_each_image_is_a_task_where_the_capacity_scales(
    beam_cubes: list[dict[str, Path]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a capacity that follows a cluster, each image's noise is a task of its own, so the cluster sees every one waiting."""
    groups: list[list[Path]] = []

    def recording(paths: list[Path], **kwargs: Any) -> list[tuple[float, ...]]:
        groups.append(list(paths))
        return measure_weights(paths, **kwargs)

    monkeypatch.setattr(patchwork.mosaic, "measure_weights", recording)
    options = cube_options_for_test(
        resample=ResampleOptions(backend="numpy", tile_size=64)
    )._replace(weight=WeightOptions(image_weight="std"))
    with ThreadPoolExecutor(2) as pool:
        mosaic(
            [cube["i"] for cube in beam_cubes],
            output_base=tmp_path / "scaling",
            options=options,
            executor=pool,
            capacity=lambda: 2,
        )
    assert sorted(groups) == sorted([cube["i"]] for cube in beam_cubes)


zarr_installed = pytest.mark.skipif(
    importlib.util.find_spec("zarr") is None, reason="zarr is not installed"
)


@zarr_installed
def test_zarr_matches_fits(beam_cubes: list[dict[str, Path]], tmp_path: Path) -> None:
    import zarr  # noqa: PLC0415

    # The same tiles for both, so their coordinate grids are the same
    options = cube_options_for_test(resample=ResampleOptions(tile_size=64))
    as_fits = mosaic(
        all_cubes(beam_cubes),
        tmp_path / "fits",
        options,
        RunOptions(channel_chunk=2),
        max_workers=2,
    )
    as_zarr = mosaic(
        all_cubes(beam_cubes),
        tmp_path / "zarr",
        options,
        RunOptions(output_format="zarr", zarr_chunk=64),
        max_workers=2,
    )
    for fits_path, zarr_path in zip(
        (*as_fits.images, *as_fits.weights),
        (*as_zarr.images, *as_zarr.weights),
        strict=True,
    ):
        array = zarr.open_array(str(zarr_path), mode="r")
        assert array.chunks == (len(FREQUENCIES), 64, 64)
        data, _ = read_cube(fits_path)
        assert np.array_equal(array[:], data, equal_nan=True)
        header = fits.Header.fromstring(array.attrs["fits_header"])
        assert header["NAXIS3"] == len(FREQUENCIES)
        beams = array.attrs["beams"]
        assert isinstance(beams, list)
        assert beams[0] == [36.0, 36.0, 0.0]


CENTRE_LEAKAGE = 0.03 - 0.02j


def leakage_slopes(beam: int, channel: int) -> complex:
    """How fast the Q and U leakage of a synthetic holography beam grow along and across its feeds."""
    return (
        4.0 * (1 + 0.5 * beam) * (1, -1, 2, 0.5, -1.5)[channel]
        + 3.0j * (1 + 0.3 * beam) * (-1, 1.5, 0.5, -2, 1)[channel]
    )


def write_leaky_holography_cube(
    path: Path, size: int = 101, cdelt_deg: float = 0.0345
) -> Path:
    """A holography cube of Gaussian beams whose leakage grows linearly from the centre, differently in every beam and channel.

    Bilinear sampling reproduces linear leakage exactly, so the leakage at any
    offset is known.
    """
    centre = size // 2
    y, x = np.mgrid[:size, :size]
    along = (x - centre) * np.radians(-cdelt_deg)
    across = (y - centre) * np.radians(cdelt_deg)
    data = np.zeros((len(BEAM_CENTRES), 4, len(FREQUENCIES), size, size))
    for channel, frequency in enumerate(FREQUENCIES):
        fwhm = gaussian_beam(frequency).fwhm_rad
        data[:, 0, channel] = np.exp(-4 * np.log(2) * (along**2 + across**2) / fwhm**2)
        for beam in range(len(BEAM_CENTRES)):
            slopes = leakage_slopes(beam, channel)
            data[beam, 1, channel] = CENTRE_LEAKAGE.real + slopes.real * along
            data[beam, 2, channel] = CENTRE_LEAKAGE.imag + slopes.imag * across

    header = fits.Header()
    for axis, (ctype, crval, cdelt, crpix) in enumerate(
        [
            ("RA---SIN", 0.0, -cdelt_deg, centre + 1),
            ("DEC--SIN", 0.0, cdelt_deg, centre + 1),
            ("FREQ", FREQUENCIES[0], FREQUENCIES[1] - FREQUENCIES[0], 1),
        ],
        start=1,
    ):
        header[f"CTYPE{axis}"] = ctype
        header[f"CRVAL{axis}"] = crval
        header[f"CDELT{axis}"] = cdelt
        header[f"CRPIX{axis}"] = crpix
    fits.writeto(path, data.astype(np.float32), header)
    return path


def add_leakage(
    cubes: dict[str, Path], beam: int, alpha_rad: float, centre_leakage: complex
) -> None:
    """Leak the Stokes I of a beam into its Q and U as the synthetic holography beam does, with the given leakage at its centre.

    The pattern is turned by the feed rotation on the sky, and its Q and U by
    twice that, into the frame of the beam.
    """
    with fits.open(cubes["i"]) as hdul:
        stokes_i = np.asarray(hdul[0].data[0], dtype=np.float64)
        header = hdul[0].header.copy()
    y, x = np.mgrid[: header["NAXIS2"], : header["NAXIS1"]]
    east, north = direction_cosines(
        header["CRVAL1"],
        header["CRVAL2"],
        *celestial_wcs(header).pixel_to_world_values(x, y),
    )
    along = east * np.cos(alpha_rad) - north * np.sin(alpha_rad)
    across = east * np.sin(alpha_rad) + north * np.cos(alpha_rad)
    with (
        fits.open(cubes["q"], mode="update") as q_hdul,
        fits.open(cubes["u"], mode="update") as u_hdul,
    ):
        for channel in range(len(FREQUENCIES)):
            slopes = leakage_slopes(beam, channel)
            leakage = (
                centre_leakage + slopes.real * along + 1j * slopes.imag * across
            ) * np.exp(2j * alpha_rad)
            q_hdul[0].data[0, channel] += leakage.real * stokes_i[channel]
            u_hdul[0].data[0, channel] += leakage.imag * stokes_i[channel]


def test_leakage_removed_channel_by_channel(tmp_path: Path) -> None:
    """Leakage that differs in every beam and channel is removed, leaving the true Q and U of each channel.

    Calibration removes the leakage at the centre of each beam, so by default it
    is taken out of the holography leakage before that is subtracted.
    """
    alpha_rad = 0.4
    holography = write_leaky_holography_cube(tmp_path / "holo.fits")
    options = cube_options_for_test(
        primary_beam=PrimaryBeamOptions(
            pb_model="holography",
            holography_cube=holography,
            holography_alpha_rad=alpha_rad,
            pb_cutoff=0.2,
        )
    )

    def worst_error(
        name: str, centre_leakage: complex, polarisation: PolarisationOptions
    ) -> NDArray[np.float64]:
        """The largest error of Q + iU over the bright pixels of each channel of a mosaic of leaky beams."""
        beams = [
            write_beam_cubes(tmp_path, f"{name}.beam{index:02d}", centre, size=100)
            for index, centre in enumerate(BEAM_CENTRES)
        ]
        for index, beam in enumerate(beams):
            add_leakage(beam, index, alpha_rad, centre_leakage)
        names = mosaic(
            all_cubes(beams),
            tmp_path / name,
            options._replace(polarisation=polarisation),
        )
        q, header = read_cube(names.images[1])
        u, _ = read_cube(names.images[2])
        expected_i, expected_linear = expected_cubes(header)
        bright = np.isfinite(q) & (expected_i > 0.2)
        assert np.all(np.sum(bright, axis=(1, 2)) > 500)
        error = np.abs(q + 1j * u - expected_linear)
        return np.array(
            [
                np.max(error[channel][bright[channel]])
                for channel in range(len(FREQUENCIES))
            ]
        )

    removed = PolarisationOptions(input_frame="crval", remove_leakage=True)
    kept = removed.model_copy(update={"keep_centre_leakage": True})
    assert np.all(worst_error("removed", 0, removed) < 1e-3)
    assert np.all(worst_error("leaky", 0, IN_BEAM_FRAME) > 3e-2)
    assert np.all(worst_error("kept", CENTRE_LEAKAGE, kept) < 1e-3)
    assert np.all(worst_error("uncalibrated", CENTRE_LEAKAGE, removed) > 1e-2)


def test_pipeline_matches_threads(
    beam_cubes: list[dict[str, Path]], tmp_path: Path
) -> None:
    options = cube_options_for_test()
    with ThreadPoolExecutor(max_workers=2) as pool:
        threaded = mosaic(
            all_cubes(beam_cubes), tmp_path / "threads", options, executor=pool
        )
    with (
        LocalCluster(
            n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None
        ) as cluster,
        Client(cluster) as client,
    ):
        dasked = run_pipeline(
            all_cubes(beam_cubes), tmp_path / "dask", client=client, options=options
        )
    for thread_path, dask_path in zip(
        (*threaded.images, *threaded.weights),
        (*dasked.images, *dasked.weights),
        strict=True,
    ):
        assert np.array_equal(
            fits.getdata(thread_path), fits.getdata(dask_path), equal_nan=True
        )


def test_parser(tmp_path: Path) -> None:
    args = get_parser().parse_args(
        [
            "a.i.fits",
            "a.q.fits",
            "--weights",
            "a.txt",
            "b.txt",
            "--convolve",
            "natural",
            "--output-format",
            "zarr",
            "--channels",
            "2",
            "10",
            "--output-base",
            str(tmp_path / "x"),
        ]
    )
    options, cube_options, convolve_options = options_from_args(args)
    assert cube_options.convolve == "natural"
    assert cube_options.channels == (2, 10)
    assert cube_options.output_format == "zarr"
    assert convolve_options.target_beam is None
    assert options.frame.full_frame is False


@zarr_installed
def test_zarr_header_describes_every_channel(tmp_path: Path) -> None:
    """The FITS header of a zarr store holds no single beam when its channels' beams differ."""
    import zarr  # noqa: PLC0415

    beams = [
        write_beam_cubes(
            tmp_path,
            f"beam{index:02d}",
            centre,
            size=60,
            beams=tuple(
                BeamShape(40 + 5 * channel, 36, 0)
                for channel in range(len(FREQUENCIES))
            ),
        )
        for index, centre in enumerate(BEAM_CENTRES[:2])
    ]
    names = mosaic(
        all_cubes(beams),
        tmp_path / "field",
        cube_options_for_test(),
        RunOptions(convolve="natural", output_format="zarr"),
    )
    store = zarr.open_array(str(names.images[0]), mode="r")
    header = fits.Header.fromstring(store.attrs["fits_header"])
    assert "BMAJ" not in header
    assert header["CASAMBM"]
    stored_beams = store.attrs["beams"]
    assert isinstance(stored_beams, list)
    assert len(stored_beams) == len(FREQUENCIES)


@pytest.mark.parametrize(
    "output_format",
    ["fits", pytest.param("zarr", marks=zarr_installed)],
)
def test_noise_cubes(tmp_path: Path, output_format: str) -> None:
    """Each group of images that share a weight has a noise cube, the inverse square root of its weight."""
    beams = [
        write_beam_cubes(tmp_path, f"beam{index:02d}", centre, size=60)
        for index, centre in enumerate(BEAM_CENTRES[:2])
    ]
    options = cube_options_for_test()._replace(
        weight=WeightOptions(image_weight="none", noise_image=True)
    )
    names = mosaic(
        all_cubes(beams),
        tmp_path / "field",
        options,
        RunOptions(output_format=output_format),  # type: ignore[arg-type]
    )
    assert [path.name for path in names.noises] == [
        f"field.i.noise.cube.{output_format}",
        f"field.qu.noise.cube.{output_format}",
    ]

    def read(path: Path) -> NDArray[np.float64]:
        if output_format == "fits":
            return read_cube(path)[0]
        import zarr  # noqa: PLC0415

        return np.asarray(zarr.open_array(str(path), mode="r")[:], dtype=np.float64)

    for weight_path, noise_path in zip(names.weights, names.noises, strict=True):
        weight, noise = read(weight_path), read(noise_path)
        assert np.any(np.isfinite(noise))
        assert np.allclose(noise, 1 / np.sqrt(weight), equal_nan=True)
    if output_format == "fits":
        assert fits.getheader(names.noises[1])["BUNIT"] == "Jy/beam"


def assert_same_fits(got: Path, expected: Path, rtol: float = 0.0) -> None:
    """Two FITS files hold the same data, to ``rtol``, and the same header cards and extensions."""
    with fits.open(got) as got_hdul, fits.open(expected) as expected_hdul:
        assert len(got_hdul) == len(expected_hdul)
        got_data, expected_data = got_hdul[0].data, expected_hdul[0].data
        assert np.array_equal(np.isnan(got_data), np.isnan(expected_data))
        assert np.allclose(got_data, expected_data, rtol=rtol, atol=0.0, equal_nan=True)
        assert sorted(map(str, got_hdul[0].header.cards)) == sorted(
            map(str, expected_hdul[0].header.cards)
        )


def test_workers_write_fits(
    beam_cubes: list[dict[str, Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strips written by their own tasks make the files the driver makes of the same strips, blanked and with their noise."""
    options = cube_options_for_test()._replace(
        weight=WeightOptions(image_weight="none", noise_image=True)
    )
    by_workers = mosaic(
        all_cubes(beam_cubes),
        tmp_path / "workers",
        options,
        RunOptions(channel_chunk=2, fits_writer="workers"),
        max_workers=3,
    )
    monkeypatch.setattr("patchwork.plan.tiles", strips)
    by_driver = mosaic(
        all_cubes(beam_cubes),
        tmp_path / "driver",
        options,
        RunOptions(channel_chunk=2),
        max_workers=3,
    )
    outputs = zip(
        (*by_workers.images, *by_workers.weights, *by_workers.noises),
        (*by_driver.images, *by_driver.weights, *by_driver.noises),
        strict=True,
    )
    for got, expected in outputs:
        assert_same_fits(got, expected)
    weight, _ = read_cube(by_workers.weights[0])
    # The primary beam cutoff blanked the edges of the mosaic
    assert np.any(np.isnan(weight[:, 1:-1, 1:-1]))
    assert np.any(np.isfinite(weight))


def test_workers_write_fits_on_dask(
    beam_cubes: list[dict[str, Path]], tmp_path: Path
) -> None:
    """Separate processes write the strips of one file as threads do.

    Tiles made in separate processes differ from those made in threads by the
    rounding of float32, whoever writes them.
    """
    options = cube_options_for_test()
    run_options = RunOptions(channel_chunk=2, fits_writer="workers")
    threaded = mosaic(
        all_cubes(beam_cubes), tmp_path / "threads", options, run_options, max_workers=2
    )
    with (
        LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
        ) as cluster,
        Client(cluster) as client,
    ):
        dasked = run_pipeline(
            all_cubes(beam_cubes),
            tmp_path / "dask",
            client=client,
            options=options,
            run_options=run_options,
        )
    for got, expected in zip(
        (*dasked.images, *dasked.weights),
        (*threaded.images, *threaded.weights),
        strict=True,
    ):
        assert_same_fits(got, expected, rtol=1e-6)


def test_workers_writing_queue_every_strip(
    beam_cubes: list[dict[str, Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is held here when the workers write, so every strip is queued for an adaptive cluster to see; the main process writing keeps twice its tasks."""
    limits: list[int] = []
    real = patchwork.write.bounded_results

    def recording(executor: Any, calls: Any, limit: Any, **kwargs: Any) -> Any:
        limits.append(limit())
        return real(executor, calls, limit, **kwargs)

    monkeypatch.setattr(patchwork.write, "bounded_results", recording)
    options = cube_options_for_test()
    mosaic(
        all_cubes(beam_cubes),
        tmp_path / "workers",
        options,
        RunOptions(channel_chunk=2, fits_writer="workers"),
        max_workers=2,
    )
    assert limits == [sys.maxsize]
    limits.clear()
    mosaic(
        all_cubes(beam_cubes),
        tmp_path / "capped",
        options,
        RunOptions(channel_chunk=2, fits_writer="workers", tasks_in_flight=3),
        max_workers=2,
    )
    assert limits == [3]
    limits.clear()
    mosaic(
        all_cubes(beam_cubes),
        tmp_path / "driver",
        options,
        RunOptions(channel_chunk=2),
        max_workers=2,
    )
    assert limits == [4]

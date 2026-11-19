from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import patchwork.rotate
from patchwork.coordinates import unit_vectors
from patchwork.io import BeamShape, celestial_wcs, read_block, read_image_info
from patchwork.mosaic import mosaic
from patchwork.options import PolarisationOptions, RotateOptions
from patchwork.polarisation import frame_angle
from patchwork.rotate import cli, get_parser, rotate_cubes, rotated_name

from .test_cube import BEAM_CENTRES, cube_options_for_test, write_beam_cubes
from .test_polarisation import (
    angle_error_deg,
    polarised_sky,
    read_complex,
    write_polarised_beam,
)

if TYPE_CHECKING:
    from pathlib import Path

    from patchwork.plan import MosaicNames


@pytest.fixture
def beam_cubes(tmp_path: Path) -> list[dict[str, Path]]:
    return [
        write_beam_cubes(tmp_path, f"field.beam{index:02d}", centre, size=100)
        for index, centre in enumerate(BEAM_CENTRES)
    ]


def linear_cubes(beam_cubes: list[dict[str, Path]]) -> list[Path]:
    """The Q and U cubes of every beam."""
    return [beam[key] for beam in beam_cubes for key in ("q", "u")]


def assert_close_mosaics(got: MosaicNames, expected: MosaicNames) -> None:
    """The mosaics agree to the rounding of float32 and the 0.001 pixel tolerance of the interpolated coordinates on each grid, which near a pole turns the angle by 1e-5 radians."""
    for got_path, expected_path in zip(got.images, expected.images, strict=True):
        got_data, expected_data = read_block(got_path), read_block(expected_path)
        assert np.array_equal(np.isfinite(got_data), np.isfinite(expected_data))
        scale = np.nanmax(np.abs(expected_data))
        assert np.allclose(
            got_data, expected_data, rtol=0, atol=1e-4 * scale, equal_nan=True
        ), got_path.name


def test_rotated_cubes_mosaic_as_the_originals_do(
    beam_cubes: list[dict[str, Path]], tmp_path: Path
) -> None:
    """Cubes rotated to north mosaic as the originals rotated in flight do, and their keywords carry the frame without options."""
    originals = linear_cubes(beam_cubes)
    rotated = rotate_cubes(originals, RotateOptions(input_frame="crval"))
    assert rotated == [rotated_name(path) for path in originals]

    options = cube_options_for_test()  # the originals are in the frame of each beam
    expected = mosaic(originals, tmp_path / "original", options)
    north = mosaic(
        rotated,
        tmp_path / "north",
        options._replace(polarisation=PolarisationOptions(input_frame="north")),
    )
    assert_close_mosaics(north, expected)
    by_keywords = mosaic(
        rotated,
        tmp_path / "keywords",
        options._replace(polarisation=PolarisationOptions()),
    )
    assert_close_mosaics(by_keywords, expected)


def test_rotation_recovers_the_sky(tmp_path: Path) -> None:
    """Images referenced to north at a boresight away from their centre are turned to the angles of the sky."""
    centre, boresight = (30.0, -88.5), (0.0, -89.0)
    q_path, u_path = write_polarised_beam(tmp_path, "beam", centre, boresight)
    q_out, u_out = rotate_cubes([q_path, u_path], RotateOptions(frame_centre=boresight))

    got, header = read_complex(q_out, u_out)
    wcs = celestial_wcs(header)
    y, x = np.mgrid[: got.shape[0], : got.shape[1]]
    ra, dec = wcs.pixel_to_world_values(x, y)
    expected = polarised_sky(
        ra, dec
    )  # the primary beam scales the amplitude, not the angle
    bright = np.abs(got) > 0.1 * np.nanmax(np.abs(got))
    assert angle_error_deg(got[bright], expected[bright]) < 0.05

    # The originals were off by the frame angle, which is large this close to the pole
    original, _ = read_complex(q_path, u_path)
    assert angle_error_deg(original[bright], expected[bright]) > 5
    turned = original * np.exp(2j * frame_angle(boresight, unit_vectors(ra, dec)))
    assert angle_error_deg(got[bright], turned[bright]) < 1e-3


def test_reverse_rotation_turns_the_other_way(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """--reverse-rotation turns Q and U by minus the frame angle, and says so in the log and in the HISTORY of the outputs."""
    centre, boresight = (30.0, -88.5), (0.0, -89.0)
    q_path, u_path = write_polarised_beam(tmp_path, "beam", centre, boresight)
    usual = rotate_cubes([q_path, u_path], RotateOptions(frame_centre=boresight))
    with caplog.at_level("WARNING", logger="patchwork"):
        cli(
            get_parser().parse_args(
                [
                    str(q_path),
                    str(u_path),
                    "--frame-centre",
                    str(boresight[0]),
                    str(boresight[1]),
                    "--reverse-rotation",
                    "--suffix",
                    "reversed",
                ]
            )
        )
    assert "The rotation is reversed" in caplog.text
    reverse = (rotated_name(q_path, "reversed"), rotated_name(u_path, "reversed"))

    original, header = read_complex(q_path, u_path)
    y, x = np.mgrid[: original.shape[0], : original.shape[1]]
    ra, dec = celestial_wcs(header).pixel_to_world_values(x, y)
    angle = frame_angle(boresight, unit_vectors(ra, dec))
    bright = np.abs(original) > 0.1 * np.nanmax(np.abs(original))
    got, got_header = read_complex(*reverse)
    turned_back = original * np.exp(-2j * angle)
    assert angle_error_deg(got[bright], turned_back[bright]) < 1e-3
    assert np.allclose(np.abs(got[bright]), np.abs(original[bright]), rtol=1e-5)

    # Twice the frame angle from the usual rotation, which is large this close to the pole
    forward, forward_header = read_complex(*usual)
    assert angle_error_deg(got[bright], forward[bright]) > 5
    assert got_header["POLFRAME"] == "NORTH"
    assert any("rotation was reversed" in card for card in got_header["HISTORY"])
    assert not any("reversed" in card for card in forward_header["HISTORY"])


def test_rotated_cubes_record_their_frame_and_keep_their_beams(tmp_path: Path) -> None:
    """The outputs say they are north at each pixel, keep the beam of every channel, and are float32 of the input's shape."""
    beams = tuple(BeamShape(36.0 + channel, 30.0, 10.0) for channel in range(5))
    cubes = write_beam_cubes(tmp_path, "beam", BEAM_CENTRES[1], beams=beams)
    outputs = rotate_cubes([cubes["q"], cubes["u"]], RotateOptions(input_frame="crval"))
    for path, output in zip((cubes["q"], cubes["u"]), outputs, strict=True):
        before, after = read_image_info(path), read_image_info(output)
        assert after.channel_beams == beams
        assert after.shape == before.shape
        assert after.n_channels == before.n_channels
        assert after.stokes == before.stokes
        assert after.header["POLFRAME"] == "NORTH"
        assert "POLRA" not in after.header
        assert after.header["BITPIX"] == -32
        assert any("Rotated by patchwork" in card for card in after.header["HISTORY"])
        assert np.array_equal(np.isnan(read_block(output)), np.isnan(read_block(path)))


def test_blocks_of_rows_in_threads_match_one_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Rotating in several blocks of rows on several threads gives exactly the serial result, and each block logs the rows done so far."""
    cubes = write_beam_cubes(tmp_path, "beam", BEAM_CENTRES[0], blank_channel=2)
    # 25 rows a block, so 7 blocks of 160 rows
    monkeypatch.setattr(patchwork.rotate, "ROTATE_BLOCK_BYTES", 2 * 5 * 160 * 8 * 25)
    pair = [cubes["q"], cubes["u"]]
    serial = rotate_cubes(pair, RotateOptions(input_frame="crval", suffix="one"))
    with caplog.at_level("DEBUG", logger="patchwork"):
        threaded = rotate_cubes(
            pair, RotateOptions(input_frame="crval", suffix="three"), threads=3
        )
    for one, three in zip(serial, threaded, strict=True):
        assert np.array_equal(read_block(one), read_block(three), equal_nan=True)
    progress = [line for line in caplog.text.splitlines() if "of 160 rows" in line]
    assert len(progress) == 7
    assert "160 of 160 rows (100%) done" in progress[-1]
    assert "Rotated rows 0-25 of" in caplog.text


def test_refusals(tmp_path: Path) -> None:
    """Anything but a pair of Q and U whose frame is known and not already north is refused before any output is written."""
    cubes = write_beam_cubes(tmp_path, "beam", BEAM_CENTRES[0])
    pair = [cubes["q"], cubes["u"]]
    in_beam_frame = RotateOptions(input_frame="crval")

    with pytest.raises(ValueError, match="Only Q and U"):
        rotate_cubes([cubes["i"], *pair], in_beam_frame)
    with pytest.raises(ValueError, match="both are needed"):
        rotate_cubes([cubes["q"]], in_beam_frame)
    with pytest.raises(ValueError, match="nothing says what their angles"):
        rotate_cubes(pair)
    assert not rotated_name(cubes["q"]).exists()

    rotated = rotate_cubes(pair, in_beam_frame)
    with pytest.raises(ValueError, match="already referenced to north"):
        rotate_cubes(rotated)
    with pytest.raises(ValueError, match="twice"):
        rotate_cubes(rotated, in_beam_frame)
    with pytest.raises(FileExistsError, match="already exists"):
        rotate_cubes(pair, in_beam_frame)
    rotate_cubes(pair, in_beam_frame, overwrite=True)

    (tmp_path / "other.q.fits").symlink_to(cubes["q"])
    (tmp_path / "other.q.rot.fits").symlink_to(cubes["q"])
    (tmp_path / "other.u.fits").symlink_to(cubes["u"])
    with pytest.raises(FileExistsError, match="also an input"):
        rotate_cubes(
            [tmp_path / "other.q.fits", tmp_path / "other.u.fits"],
            in_beam_frame,
            overwrite=True,
        )


def test_rotate_command(tmp_path: Path) -> None:
    """The command rotates with the frame, suffix and threads it is given."""
    cubes = write_beam_cubes(tmp_path, "beam", BEAM_CENTRES[2])
    cli(
        get_parser().parse_args(
            [
                str(cubes["u"]),
                str(cubes["q"]),
                "--input-frame",
                "crval",
                "--suffix",
                "north",
                "--threads",
                "2",
            ]
        )
    )
    for key in ("q", "u"):
        output = tmp_path / f"beam.{key}.north.fits"
        assert read_image_info(output).header["POLFRAME"] == "NORTH"

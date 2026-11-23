from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits

from patchwork.io import (
    NO_BEAM,
    BeamShape,
    beam_number_from_name,
    check_output_fits,
    create_output_fits,
    frequency_from_header,
    read_block,
    read_image_info,
    read_plane,
    read_plane_at,
    readable_location,
    write_block,
)

from .conftest import FREQUENCY_HZ, ImageFactory, image_header

if TYPE_CHECKING:
    from pathlib import Path


def test_beam_number_from_name() -> None:
    assert beam_number_from_name("SB1234.RACS_0012+00.beam07.round2.i.fits") == 7
    assert beam_number_from_name("image.restored.i.SB1.cont.taylor.0.Beam35.fits") == 35
    assert beam_number_from_name("mosaic.fits") is None


def test_frequency_from_header() -> None:
    header = image_header(150.0, -30.0)
    assert frequency_from_header(header) == FREQUENCY_HZ
    plain = fits.Header({"NAXIS": 2, "RESTFRQ": 1.4e9})
    assert frequency_from_header(plain) == 1.4e9
    assert frequency_from_header(fits.Header({"NAXIS": 2})) is None


def test_read_image_info(make_image: ImageFactory) -> None:
    info = read_image_info(make_image("SB9.beam03.fits", 150.0, -30.0, size=40))
    assert info.shape == (40, 40)
    assert info.beam_number == 3
    assert info.frequency_hz == FREQUENCY_HZ
    assert info.beam_centre == (150.0, -30.0)
    assert info.beam == pytest.approx(BeamShape(36.0, 36.0, 0.0))

    override = read_image_info(info.path, beam_number=5, beam_centre=(1.0, 2.0))
    assert override.beam_number == 5
    assert override.beam_centre == (1.0, 2.0)


def test_cubes(tmp_path: Path) -> None:
    """A cube along FREQ is read; one along STOKES is refused."""
    header = image_header(150.0, -30.0, size=10)
    path = tmp_path / "cube.fits"
    fits.writeto(path, np.arange(300, dtype=np.float32).reshape(1, 3, 10, 10), header)
    info = read_image_info(path)
    assert info.n_channels == 3
    assert info.frequencies_hz == (FREQUENCY_HZ, FREQUENCY_HZ + 1e6, FREQUENCY_HZ + 2e6)
    assert info.channel_beams == (BeamShape(36.0, 36.0, 0.0),) * 3
    block = read_block(path, (slice(2, 4), slice(0, 5)), slice(1, 3))
    assert block.shape == (2, 2, 5)
    assert block[1, 0, 0] == 220
    with pytest.raises(ValueError, match="channels"):
        read_plane(path)

    stokes_path = tmp_path / "stokes.fits"
    fits.writeto(stokes_path, np.zeros((3, 1, 10, 10), dtype=np.float32), header)
    with pytest.raises(ValueError, match="one Stokes parameter"):
        read_image_info(stokes_path)


def test_cube_axis_orders(tmp_path: Path) -> None:
    """Cubes ordered (RA, Dec, STOKES, FREQ), as CASA and arrakis write them, read as (RA, Dec, FREQ, STOKES) do."""
    header = image_header(150.0, -30.0, size=10)
    for key in ("CTYPE", "CRVAL", "CDELT", "CRPIX"):
        header[f"{key}3"], header[f"{key}4"] = header[f"{key}4"], header[f"{key}3"]
    data = np.arange(300, dtype=np.float32).reshape(3, 1, 10, 10)
    path = tmp_path / "casa.fits"
    fits.writeto(path, data, header)
    info = read_image_info(path)
    assert info.n_channels == 3
    assert info.stokes == 1
    assert np.array_equal(read_block(path, channels=slice(1, 2))[0], data[1, 0])


def test_beam_table_round_trip(tmp_path: Path) -> None:
    header = image_header(150.0, -30.0, size=10)
    header["NAXIS3"] = 3
    beams = (BeamShape(10, 8, 30), NO_BEAM, BeamShape(12, 9, -20))
    path = create_output_fits(tmp_path / "beams.fits", header, beams)
    info = read_image_info(path)
    assert info.channel_beams[0] == pytest.approx(beams[0])
    assert not info.channel_beams[1].is_usable()  # type: ignore[union-attr]
    assert "BMAJ" not in info.header
    assert info.header["CASAMBM"]

    same = create_output_fits(tmp_path / "same.fits", header, (beams[0],) * 3)
    with fits.open(same) as hdul:
        assert "BEAMS" not in hdul
    assert all(
        beam == pytest.approx(beams[0]) for beam in read_image_info(same).channel_beams
    )


@pytest.mark.parametrize("unit", ["deg", "DEGREES"])
def test_beam_table_in_degrees(tmp_path: Path, unit: str) -> None:
    """A BEAMS table whose axes are in degrees, as AIPS writes them, is read in arcseconds."""
    header = image_header(150.0, -30.0, size=10)
    header["NAXIS3"] = 2
    path = create_output_fits(
        tmp_path / "beams.fits", header, (BeamShape(1, 1, 0), BeamShape(2, 2, 0))
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column("BMAJ", "D", unit=unit, array=np.array([10, 12]) / 3600),
            fits.Column("BMIN", "D", unit=unit, array=np.array([8, 9]) / 3600),
            fits.Column("BPA", "D", unit="deg", array=[30, -20]),
            fits.Column("CHAN", "J", array=[0, 1]),
            fits.Column("POL", "J", array=[0, 0]),
        ],
        name="BEAMS",
    )
    with fits.open(path, mode="update") as hdul:
        hdul[hdul.index_of("BEAMS")] = table

    beams = read_image_info(path).channel_beams
    assert beams[0] == pytest.approx(BeamShape(10, 8, 30), rel=1e-6)
    assert beams[1] == pytest.approx(BeamShape(12, 9, -20), rel=1e-6)


def test_read_plane_region(make_image: ImageFactory) -> None:
    path = make_image("a.fits", 150.0, -30.0, size=40)
    full = read_plane(path)
    assert full.shape == (40, 40)
    assert np.array_equal(
        read_plane(path, (slice(5, 10), slice(0, 3))), full[5:10, 0:3]
    )


def test_create_and_write_output(tmp_path: Path) -> None:
    header = image_header(150.0, -30.0, size=50)
    header["BITPIX"] = 16
    path = create_output_fits(tmp_path / "out.fits", header)
    with fits.open(path, mode="update", memmap=True) as hdul:
        assert hdul[0].data.shape == (1, 1, 50, 50)
        write_block(hdul[0], (slice(0, 10), slice(20, 30)), np.ones((1, 10, 10)))

    with fits.open(path) as hdul:
        data = hdul[0].data
        assert hdul[0].header["BITPIX"] == -32
        assert hdul[0].header["CRVAL1"] == 150.0
        assert np.sum(data) == 100
        assert np.all(data[0, 0, 0:10, 20:30] == 1)


def test_check_output_fits(tmp_path: Path) -> None:
    path = tmp_path / "exists.fits"
    check_output_fits(path, overwrite=False)
    path.touch()
    with pytest.raises(FileExistsError, match="already exists"):
        check_output_fits(path, overwrite=False)
    check_output_fits(path, overwrite=True, input_fits=[None])
    # An input is never written over, as its output is created empty before it is read
    for overwrite in (False, True):
        with pytest.raises(FileExistsError, match="also an input"):
            check_output_fits(path, overwrite=overwrite, input_fits=[None, path])


def test_output_drops_checksums(tmp_path: Path) -> None:
    """Checksums of the data of the header an output is made from do not hold for the output."""
    source = tmp_path / "source.fits"
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32)).writeto(source, checksum=True)
    output = create_output_fits(tmp_path / "output.fits", fits.getheader(source))
    header = fits.getheader(output)
    assert "CHECKSUM" not in header
    assert "DATASUM" not in header


def test_read_plane_at_matches_astropy(tmp_path: Path) -> None:
    """A plane at any index of the other axes is read with one call as astropy reads it."""
    data = np.random.default_rng(0).normal(size=(3, 4, 5, 7, 6)).astype(np.float32)
    path = tmp_path / "cube.fits"
    fits.writeto(path, data)
    location = readable_location(path)
    assert location is not None
    for index in [(0, 0, 0), (2, 3, 4), (1, 2, 3)]:
        assert np.array_equal(read_plane_at(location, index), data[index])
    with pytest.raises(ValueError, match="does not pick a plane"):
        read_plane_at(location, (0, 0))
    with pytest.raises(ValueError, match="invalid entry"):
        read_plane_at(location, (3, 0, 0))
    # A file that ends early, after its layout was found
    with path.open("r+b") as file:
        file.truncate(location.offset + 4 * 6 * 7 * 10)
    with pytest.raises(ValueError, match="ends before the plane"):
        read_plane_at(location, (2, 3, 4))


def test_readable_location_only_of_plain_float32(tmp_path: Path) -> None:
    """Data that are not plain float32, or not all there, are left to astropy to read."""
    data = np.ones((4, 5), dtype=np.float32)
    plain, double, scaled = (
        tmp_path / f"{name}.fits" for name in ("plain", "double", "scaled")
    )
    fits.writeto(plain, data)
    fits.writeto(double, data.astype(np.float64))
    hdu = fits.PrimaryHDU(data)
    hdu.header["BSCALE"] = 2.0
    hdu.writeto(scaled)
    assert readable_location(plain) is not None
    assert readable_location(double) is None
    assert readable_location(scaled) is None

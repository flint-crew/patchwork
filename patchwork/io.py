"""Reading input images and cubes, and writing mosaics to FITS"""

from __future__ import annotations

import functools
import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, NamedTuple

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from radio_beam import Beam, Beams

from patchwork.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

Slices = tuple[slice, slice]

BLANK_BEAM = float(np.finfo(np.float32).tiny)
"""The value fitscube writes into a BEAMS table for a channel with no restoring beam, as CASA rejects NaN"""


class BeamShape(NamedTuple):
    """A restoring beam held as plain floats, as astropy quantities do not travel
    well to dask workers"""

    bmaj_arcsec: float
    """Major axis FWHM in arcseconds"""
    bmin_arcsec: float
    """Minor axis FWHM in arcseconds"""
    bpa_deg: float
    """Position angle of the major axis, east of north, in degrees"""

    def __str__(self) -> str:
        return f'{self.bmaj_arcsec:g}" x {self.bmin_arcsec:g}", PA {self.bpa_deg:g} deg'

    @classmethod
    def from_header(cls, header: fits.Header) -> BeamShape | None:
        """The beam described by the BMAJ, BMIN and BPA of a header, if it has one."""
        if "BMAJ" not in header or "BMIN" not in header:
            return None
        return cls(
            bmaj_arcsec=float(header["BMAJ"]) * 3600,
            bmin_arcsec=float(header["BMIN"]) * 3600,
            bpa_deg=float(header.get("BPA", 0.0)),
        )

    @classmethod
    def from_radio_beam(cls, beam: Beam) -> BeamShape:
        """Convert a ``radio_beam.Beam``."""
        return cls(
            bmaj_arcsec=float(beam.major.to(u.arcsec).value),
            bmin_arcsec=float(beam.minor.to(u.arcsec).value),
            bpa_deg=float(beam.pa.to(u.deg).value),
        )

    def to_radio_beam(self) -> Beam:
        """Convert to a ``radio_beam.Beam``."""
        return Beam(
            major=self.bmaj_arcsec * u.arcsec,
            minor=self.bmin_arcsec * u.arcsec,
            pa=self.bpa_deg * u.deg,
        )

    def is_usable(self) -> bool:
        """Whether the beam is a real one: finite, and larger than zero or the blank placeholder."""
        return bool(
            np.all(np.isfinite(self))
            and self.bmaj_arcsec > BLANK_BEAM
            and self.bmin_arcsec > BLANK_BEAM
        )

    def to_header(self, header: fits.Header) -> None:
        """Write the beam into a header."""
        header["BMAJ"] = self.bmaj_arcsec / 3600
        header["BMIN"] = self.bmin_arcsec / 3600
        header["BPA"] = self.bpa_deg


NO_BEAM = BeamShape(np.nan, np.nan, np.nan)
"""The beam of a channel that holds no image"""


class ImageInfo(NamedTuple):
    """What the mosaicking needs to know about an input image or cube, read from its header"""

    path: Path
    """Path to the image"""
    shape: tuple[int, int]
    """The (ny, nx) shape of the image plane"""
    wcs: WCS
    """Celestial WCS of the image plane"""
    header: fits.Header
    """The full header of the image"""
    frequency_hz: float | None
    """Frequency of the image (its first channel), if known"""
    beam: BeamShape | None
    """Restoring beam of the image (its first channel), if known"""
    beam_number: int | None
    """The beam number, used to pick the beam from a holography cube"""
    beam_centre: tuple[float, float]
    """The (RA, Dec) in degrees of the primary beam centre"""
    stokes: int | None = None
    """The FITS code of the Stokes parameter of the image (1 to 4 for I, Q, U and V), if it has a STOKES axis"""
    n_channels: int = 1
    """The number of channels along the FREQ axis"""
    frequencies_hz: tuple[float, ...] = ()
    """The frequency of each channel, if known"""
    channel_beams: tuple[BeamShape | None, ...] = ()
    """The restoring beam of each channel, from a BEAMS table or the header, where None is unknown"""


def beam_number_from_name(path: str | Path) -> int | None:
    """The beam number in a name such as ``SB1234.field.beam07.fits``, if there is one."""
    matches = re.findall(r"beam(\d+)", Path(path).name, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def _find_axis(header: fits.Header, prefix: str) -> int | None:
    """The FITS number of the first axis whose CTYPE starts with ``prefix``."""
    for axis in range(1, header.get("NAXIS", 0) + 1):
        if str(header.get(f"CTYPE{axis}", "")).upper().startswith(prefix):
            return axis
    return None


def _axis_values(header: fits.Header, axis: int) -> NDArray[np.float64]:
    """The world value of each pixel along an axis."""
    pixels = np.arange(header[f"NAXIS{axis}"]) + 1
    crpix = header.get(f"CRPIX{axis}", 1.0)
    return np.asarray(
        header[f"CRVAL{axis}"] + (pixels - crpix) * header.get(f"CDELT{axis}", 1.0)
    )


def spectral_axis(header: fits.Header) -> int | None:
    """The FITS number of the FREQ axis, if there is one."""
    return _find_axis(header, "FREQ")


def stokes_from_header(header: fits.Header) -> int | None:
    """The FITS code of the Stokes parameter of the first pixel of a STOKES axis, if there is one."""
    axis = _find_axis(header, "STOKES")
    if axis is None:
        return None
    return round(float(_axis_values(header, axis)[0]))


def frequencies_from_header(header: fits.Header) -> tuple[float, ...]:
    """The frequency of each channel of a FREQ axis, else the rest frequency, if either is known."""
    axis = spectral_axis(header)
    if axis is not None:
        return tuple(float(value) for value in _axis_values(header, axis))
    for key in ("RESTFRQ", "RESTFREQ"):
        if key in header:
            return (float(header[key]),)
    return ()


def frequency_from_header(header: fits.Header) -> float | None:
    """The frequency of the first pixel of a FREQ axis, else the rest frequency, if either is known."""
    frequencies = frequencies_from_header(header)
    return frequencies[0] if frequencies else None


def full_wcs(header: fits.Header) -> WCS:
    """The WCS of a header, without the warnings astropy raises as it tidies real headers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(header)


def celestial_wcs(header: fits.Header) -> WCS:
    """The celestial WCS of a header."""
    return full_wcs(header).celestial


def _check_axes(header: fits.Header, path: Path) -> None:
    """Raise unless an image has its celestial axes first, and no axis beyond them but FREQ is longer than one."""
    full = full_wcs(header)
    if (full.wcs.lng, full.wcs.lat) != (0, 1):
        msg = f"The first two axes of {path} must be celestial longitude and latitude"
        raise ValueError(msg)
    for axis in range(3, header["NAXIS"] + 1):
        if header[f"NAXIS{axis}"] > 1 and axis != spectral_axis(header):
            msg = (
                f"Axis {axis} ({header.get(f'CTYPE{axis}')}) of {path} has more than one "
                "pixel. Only a FREQ axis may, so each cube must hold one Stokes parameter"
            )
            raise ValueError(msg)


def channel_beams_from_hdul(
    hdul: fits.HDUList, n_channels: int
) -> tuple[BeamShape | None, ...]:
    """The restoring beam of each channel: from a BEAMS table if there is one, otherwise the header beam."""
    if "BEAMS" not in hdul:
        return (BeamShape.from_header(hdul[0].header),) * n_channels

    table = hdul["BEAMS"]
    # radio_beam reads the axes in the units of their columns, which are not always arcsec
    found = Beams.from_fits_bintable(table)
    major, minor = found.major.to_value(u.arcsec), found.minor.to_value(u.arcsec)
    pa = found.pa.to_value(u.deg)
    channels, pols = table.data["CHAN"], table.data["POL"]
    beams: list[BeamShape | None] = [NO_BEAM] * n_channels
    for row in np.flatnonzero(pols == np.min(pols)):
        if 0 <= channels[row] < n_channels:
            beams[channels[row]] = BeamShape(
                float(major[row]), float(minor[row]), float(pa[row])
            )
    return tuple(beams)


def read_image_info(
    path: str | Path,
    beam_number: int | None = None,
    frequency_hz: float | None = None,
    beam_centre: tuple[float, float] | None = None,
) -> ImageInfo:
    """Read what the mosaicking needs to know from the header of an image or cube.

    Unless given, the beam number is taken from the file name, the frequency
    from the header, and the beam centre from the reference value of the
    celestial axes.

    Args:
        path (str | Path): The image to read
        beam_number (int | None, optional): Beam number of the image. Defaults to None.
        frequency_hz (float | None, optional): Frequency of an image with no frequency in its header. Defaults to None.
        beam_centre (tuple[float, float] | None, optional): (RA, Dec) in degrees of the primary beam centre. Defaults to None.

    Returns:
        ImageInfo: The description of the image
    """
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header.copy()
        _check_axes(header=header, path=path)
        axis = spectral_axis(header)
        n_channels = 1 if axis is None else int(header[f"NAXIS{axis}"])
        channel_beams = channel_beams_from_hdul(hdul, n_channels)

    frequencies = frequencies_from_header(header) or (
        (frequency_hz,) if frequency_hz else ()
    )
    wcs = celestial_wcs(header)
    return ImageInfo(
        path=path,
        shape=(int(header["NAXIS2"]), int(header["NAXIS1"])),
        wcs=wcs,
        header=header,
        frequency_hz=frequencies[0] if frequencies else None,
        beam=channel_beams[0],
        beam_number=beam_number
        if beam_number is not None
        else beam_number_from_name(path),
        beam_centre=beam_centre or (float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])),
        stokes=stokes_from_header(header),
        n_channels=n_channels,
        frequencies_hz=frequencies,
        channel_beams=channel_beams,
    )


def data_index(
    header: fits.Header, region: Slices | None = None, channels: slice | None = None
) -> tuple[slice, ...]:
    """The index into the data of a FITS image of a region of some of its channels, keeping every axis."""
    naxis = int(header["NAXIS"])
    ys, xs = region or (slice(None), slice(None))
    spectral = spectral_axis(header)
    index = [
        (channels or slice(None)) if naxis - numpy_axis == spectral else slice(0, 1)
        for numpy_axis in range(naxis - 2)
    ]
    return (*index, ys, xs)


def read_block(
    path: str | Path,
    region: Slices | None = None,
    channels: slice | None = None,
    dtype: type[np.floating] = np.float64,
    out: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Read a region of some channels of a FITS image or cube, as (channel, y, x).

    The file is memory mapped, so only the region is read from disk.

    Args:
        path (str | Path): The image or cube to read
        region (Slices | None, optional): The (y, x) slices to read. Defaults to None, reading all of each plane.
        channels (slice | None, optional): The channels to read. Defaults to None, reading every channel.
        dtype (type[np.floating], optional): The type of the block. Defaults to np.float64.
        out (NDArray[np.floating] | None, optional): A (channel, y, x) array to read the block into, in its own type. Defaults to None, making a new one.

    Returns:
        NDArray[np.floating]: The block, with NaN wherever there is no data
    """
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0]
        data = hdu.data[data_index(hdu.header, region, channels)]
        if out is not None:
            out[...] = data.reshape(-1, *data.shape[-2:])
            return out
        # One copy, in the machine's byte order, which no longer needs the file
        block = np.array(data, dtype=dtype)
    return block.reshape(-1, *block.shape[-2:])


def read_plane(path: str | Path, region: Slices | None = None) -> NDArray[np.float64]:
    """Read the image plane of a FITS file, or a region of it.

    Args:
        path (str | Path): The image to read
        region (Slices | None, optional): The (y, x) slices to read. Defaults to None, reading all of it.

    Raises:
        ValueError: Raised if the image has more than one channel

    Returns:
        NDArray[np.float64]: The plane, with NaN wherever there is no data
    """
    block = read_block(path, region)
    if block.shape[0] > 1:
        msg = f"{path} has {block.shape[0]} channels, but one plane is read from it"
        raise ValueError(msg)
    return np.asarray(block[0])


def write_block(
    hdu: fits.PrimaryHDU,
    region: Slices,
    block: NDArray[np.floating],
    channels: slice | None = None,
) -> None:
    """Write a (channel, y, x) block into the data of a FITS image opened for update."""
    index = data_index(hdu.header, region, channels)
    target = hdu.data[index]
    target[...] = np.reshape(block, target.shape)


class DataLocation(NamedTuple):
    """Where the data of a FITS image lies in its file, to read and write whole rows of it in place"""

    path: Path
    """The file"""
    offset: int
    """The byte the data starts at"""
    shape: tuple[int, ...]
    """The shape of the data, in numpy order"""
    spectral: int | None
    """The numpy axis of the channels, if there is one"""

    @property
    def n_channels(self) -> int:
        """The number of channels of the data."""
        return 1 if self.spectral is None else self.shape[self.spectral]


def data_location(path: str | Path) -> DataLocation:
    """Find where the data of a float32 FITS image lies in its file.

    Raises:
        ValueError: Raised if its data are not plain float32, which rows are read and written as
    """
    with fits.open(path) as hdul:
        header = hdul[0].header
        offset = int(hdul.fileinfo(0)["datLoc"])
    if (
        header["BITPIX"] != -32
        or header.get("BSCALE", 1) != 1
        or header.get("BZERO", 0)
    ):
        msg = f"The data of {path} are not plain float32, so its rows cannot be written in place"
        raise ValueError(msg)
    naxis = int(header["NAXIS"])
    axis = spectral_axis(header)
    return DataLocation(
        path=Path(path),
        offset=offset,
        shape=tuple(int(header[f"NAXIS{number}"]) for number in range(naxis, 0, -1)),
        spectral=None if axis is None else naxis - axis,
    )


@functools.lru_cache(maxsize=4096)
def _mapped_location(path: str, modified_ns: int, size: int) -> DataLocation | None:  # noqa: ARG001
    """Where the data of a FITS file lies, if it can be read straight from the file: plain float32 that is not compressed and is all there.

    It is kept for the file as it was when asked, by its time of modification
    and size, so a file that is rewritten is looked at again.
    """
    with Path(path).open("rb") as file:
        # A compressed file opens as FITS too, but its bytes are not the data
        if file.read(9) != b"SIMPLE  =":
            return None
    try:
        location = data_location(path)
    except ValueError:
        return None
    if location.offset + 4 * int(np.prod(location.shape)) > size:
        return None
    return location


def readable_location(path: str | Path) -> DataLocation | None:
    """Where the data of a FITS file lies, if it can be read straight from the file: plain float32 that is not compressed and is all there, or None."""
    status = Path(path).stat()
    return _mapped_location(str(path), status.st_mtime_ns, status.st_size)


def read_window(
    path: str | Path,
    region: Slices,
    channels: slice | None,
    out: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Read a region of some channels of a FITS image or cube into ``out``, as ``read_block`` does.

    ``read_block`` opens the file with astropy each time, which parses its
    header while holding the GIL. Many windows of one file are read from
    threads here, so the header is parsed once, and each window is then copied
    from a memory map of the file, which does not hold the GIL. A file whose
    data are not plain float32 is read by ``read_block``.

    Args:
        path (str | Path): The image or cube to read
        region (Slices): The (y, x) slices to read
        channels (slice | None): The channels to read, or None for every channel
        out (NDArray[np.floating]): The (channel, y, x) array the window is read into

    Returns:
        NDArray[np.floating]: ``out``, with NaN wherever there is no data
    """
    location = readable_location(path)
    if location is None:
        return read_block(path, region, channels, out=out)
    data = np.memmap(
        path, dtype=">f4", mode="r", offset=location.offset, shape=location.shape
    )
    index = tuple(
        (channels or slice(None)) if axis == location.spectral else slice(0, 1)
        for axis in range(len(location.shape) - 2)
    )
    window = data[(*index, *region)]
    out[...] = window.reshape(-1, *window.shape[-2:])
    return out


def _rows_offset(location: DataLocation, channel: int, row: int) -> int:
    """The byte at which a row of a channel of a float32 FITS image starts."""
    index = [0] * len(location.shape)
    index[-2] = row
    if location.spectral is not None:
        index[location.spectral] = channel
    return location.offset + 4 * int(np.ravel_multi_index(index, location.shape))


def write_rows(
    location: DataLocation, rows: slice, channels: slice, block: NDArray[np.floating]
) -> None:
    """Write whole rows of some channels, a (channel, y, x) block, into a float32 FITS image in place.

    Each channel's rows are one run of bytes, written without reading or
    rewriting anything else in the file, so processes on different machines
    may write different rows of one file at once on a file system that keeps it
    consistent, such as Lustre.
    """
    n_rows, n_channels = rows.stop - rows.start, channels.stop - channels.start
    expected = (n_channels, n_rows, location.shape[-1])
    # A block of the wrong shape would run on into the bytes of other rows or channels
    if (
        block.shape != expected
        or rows.start < 0
        or rows.stop > location.shape[-2]
        or channels.start < 0
        or channels.stop > location.n_channels
    ):
        msg = (
            f"A block of shape {block.shape} does not fill rows {rows.start} to {rows.stop} "
            f"of channels {channels.start} to {channels.stop} of {location.path}, "
            f"whose data are {location.shape}"
        )
        raise ValueError(msg)
    with location.path.open("r+b") as file:
        for index, channel in enumerate(range(channels.start, channels.stop)):
            # The bytes of the rows as FITS holds them, without a second copy
            data = memoryview(np.ascontiguousarray(block[index], dtype=">f4")).cast("B")
            _write_all(file, data, _rows_offset(location, channel, rows.start))


def write_tile_rows(
    location: DataLocation,
    region: Slices,
    channels: slice,
    block: NDArray[np.floating],
) -> None:
    """Write a (channel, y, x) block over part of some rows into a float32 FITS image in place, a row at a time.

    Each row of each channel is a run of bytes of its own, so on a file system
    such as Lustre, which is slow to write many short runs, this is far slower
    than writing whole rows with ``write_rows``.
    """
    rows, columns = region
    n_rows, n_columns = rows.stop - rows.start, columns.stop - columns.start
    expected = (channels.stop - channels.start, n_rows, n_columns)
    if (
        block.shape != expected
        or rows.start < 0
        or rows.stop > location.shape[-2]
        or columns.start < 0
        or columns.stop > location.shape[-1]
        or channels.start < 0
        or channels.stop > location.n_channels
    ):
        msg = (
            f"A block of shape {block.shape} does not fill rows {rows.start} to {rows.stop}, "
            f"columns {columns.start} to {columns.stop}, of channels {channels.start} to "
            f"{channels.stop} of {location.path}, whose data are {location.shape}"
        )
        raise ValueError(msg)
    with location.path.open("r+b") as file:
        for index, channel in enumerate(range(channels.start, channels.stop)):
            data = np.ascontiguousarray(block[index], dtype=">f4")
            for row in range(n_rows):
                offset = _rows_offset(location, channel, rows.start + row)
                _write_all(
                    file, memoryview(data[row]).cast("B"), offset + 4 * columns.start
                )


def _write_all(file: BinaryIO, data: memoryview, offset: int) -> None:
    """Write all of some bytes at an offset of an open file, however many writes it takes."""
    written = 0
    while written < len(data):
        count = os.pwrite(file.fileno(), data[written:], offset + written)
        if count <= 0:
            msg = f"Could not write rows of {file.name}"
            raise OSError(msg)
        written += count


def _read_into(path: Path, offset: int, out: NDArray[Any]) -> bool:
    """Fill an array with the bytes of a file from an offset, in as few reads as the system allows.

    Returns:
        bool: False if the file ends first
    """
    buffer = memoryview(out.reshape(-1).view(np.uint8))
    with path.open("rb", buffering=0) as file:
        file.seek(offset)
        got = 0
        while got < len(buffer):
            count = file.readinto(buffer[got:])
            if not count:
                return False
            got += count
    return True


def read_rows(
    location: DataLocation, rows: slice, channel: int, step: int = 1
) -> NDArray[np.float32]:
    """Read whole rows of one channel of a float32 FITS image, as (y, x), with one read.

    With ``step`` every row is still read, as one run of bytes, but only every
    ``step``'th from the first is kept. On a file system such as Lustre a read
    of a few MB takes little longer than one of a single row, so this is far
    faster than reading the rows on their own.
    """
    data = np.empty((rows.stop - rows.start, location.shape[-1]), dtype=">f4")
    if not _read_into(location.path, _rows_offset(location, channel, rows.start), data):
        msg = f"{location.path} ends before row {rows.stop} of channel {channel}"
        raise ValueError(msg)
    return data[::step].astype(np.float32)


def read_plane_at(
    location: DataLocation, index: tuple[int, ...]
) -> NDArray[np.float32]:
    """Read one (y, x) plane of a float32 FITS image or cube, at an index of each of its other axes in numpy order, with one read.

    Raises:
        ValueError: Raised if the index is not one of the data, or the file ends before the plane does
    """
    if len(index) != len(location.shape) - 2:
        msg = f"Index {index} of {location.path} does not pick a plane of data shaped {location.shape}"
        raise ValueError(msg)
    first = np.ravel_multi_index((*index, 0, 0), location.shape)
    data = np.empty(location.shape[-2:], dtype=">f4")
    if not _read_into(location.path, location.offset + 4 * int(first), data):
        msg = f"{location.path} ends before the plane at {index}"
        raise ValueError(msg)
    return data.astype(np.float32)


def suffixed_name(path: str | Path, suffix: str) -> Path:
    """The name of an output made from an input: ``suffix`` inserted before a final ``.fits``, or added with ``.fits`` after any other name, so it never names the input itself."""
    path = Path(path)
    name = path.name
    if name.lower().endswith(".fits"):
        return path.with_name(f"{name[:-5]}.{suffix}{name[-5:]}")
    return path.with_name(f"{name}.{suffix}.fits")


def check_output_fits(
    output_fits: str | Path,
    overwrite: bool,
    input_fits: Iterable[str | Path | None] = (),
) -> None:
    """Check that an output FITS image may be written, before any work is done to
    make it. An existing output is an error unless ``overwrite`` is set, in which
    case it is replaced with a warning. An output that is also one of the inputs,
    even when reached through another path such as a symbolic link, is always an
    error: outputs are created empty before the inputs are read, so writing one
    over an input would destroy it.

    Args:
        output_fits (str | Path): The output that will be written
        overwrite (bool): Whether an existing output may be replaced
        input_fits (Iterable[str | Path | None], optional): The inputs of the operation, where None is skipped. Defaults to ().

    Raises:
        FileExistsError: Raised when the output is one of the inputs, or already exists and ``overwrite`` is False
    """
    output_fits = Path(output_fits)
    if not output_fits.exists():
        return

    if any(
        Path(path).exists() and output_fits.samefile(path)
        for path in input_fits
        if path is not None
    ):
        msg = f"{output_fits} is also an input, which writing it would destroy. Choose another output name"
        raise FileExistsError(msg)
    if not overwrite:
        msg = f"{output_fits} already exists. Set overwrite=True (--overwrite on the CLI) to replace it"
        raise FileExistsError(msg)
    logger.warning(f"Overwriting the existing {output_fits}")


def beams_header(header: fits.Header, beams: tuple[BeamShape, ...]) -> bool:
    """Describe the restoring beams of the channels in a header, returning whether a BEAMS table is needed.

    One beam for every channel is written as BMAJ, BMIN and BPA. Beams that
    differ need a BEAMS table, which the header then announces as fitscube does.
    """
    for key in ("BMAJ", "BMIN", "BPA", "CASAMBM"):
        header.remove(key, ignore_missing=True, remove_all=True)
    if not beams:
        return False
    if all(np.allclose(beam, beams[0], equal_nan=True) for beam in beams):
        if beams[0].is_usable():
            beams[0].to_header(header)
        return False

    header["CASAMBM"] = True
    header["COMMENT"] = "The PSF in each image plane varies."
    header["COMMENT"] = "Full beam information is stored in the second FITS extension."
    header["COMMENT"] = (
        f"The value '{BLANK_BEAM}' represents a NaN PSF in the beamtable."
    )
    return True


def beam_table(beams: tuple[BeamShape, ...], stokes_index: int = 0) -> fits.BinTableHDU:
    """A BEAMS table of the restoring beam of each channel, as CASA and fitscube write it."""
    values = np.array(
        [beam if beam.is_usable() else (BLANK_BEAM,) * 3 for beam in beams],
        dtype=np.float64,
    ).reshape(-1, 3)
    table = Table(
        data=[
            values[:, 0],
            values[:, 1],
            values[:, 2],
            np.arange(len(beams)),
            np.full(len(beams), stokes_index),
        ],
        names=["BMAJ", "BMIN", "BPA", "CHAN", "POL"],
        dtype=["f4", "f4", "f4", "i4", "i4"],
        units=["arcsec", "arcsec", "deg", None, None],
    )
    hdu = fits.table_to_hdu(table)
    hdu.header["EXTNAME"] = "BEAMS"
    hdu.header["NCHAN"] = len(beams)
    hdu.header["NPOL"] = 1
    return hdu


def create_output_fits(
    path: str | Path, header: fits.Header, beams: tuple[BeamShape, ...] = ()
) -> Path:
    """Create a float32 FITS file shaped by ``header`` without holding its data in memory.

    The data starts as zeros, and is filled by writing blocks into it. The
    restoring beams of the channels go into the header, or into a BEAMS table
    after the data when they differ.

    Args:
        path (str | Path): The file to create, replacing any that exists
        header (fits.Header): Header giving the shape and coordinates of the data
        beams (tuple[BeamShape, ...], optional): The restoring beam of each channel. Defaults to (), leaving the header beam as it is.

    Returns:
        Path: The created file
    """
    path = Path(path)
    naxis = int(header["NAXIS"])
    shape = [int(header[f"NAXIS{axis}"]) for axis in range(1, naxis + 1)]

    structural = {"SIMPLE", "BITPIX", "NAXIS", "EXTEND", "BSCALE", "BZERO", "BLANK"}
    # Checksums of the reference's data do not hold for the new file
    structural |= {"CHECKSUM", "DATASUM"}
    structural |= {f"NAXIS{axis}" for axis in range(1, naxis + 1)}
    out_header = fits.Header(
        [
            ("SIMPLE", True),
            ("BITPIX", -32),
            ("NAXIS", naxis),
            *[(f"NAXIS{axis}", size) for axis, size in enumerate(shape, start=1)],
            ("EXTEND", True),
        ]
    )
    for card in header.cards:
        if card.keyword not in structural:
            out_header.append(card)
    needs_table = beams_header(out_header, beams) if beams else False

    header_bytes = out_header.tostring().encode("ascii")
    data_bytes = 4 * int(np.prod(shape))
    padded_bytes = -(-data_bytes // 2880) * 2880
    with path.open("wb") as out_file:
        out_file.write(header_bytes)
        out_file.seek(len(header_bytes) + padded_bytes - 1)
        out_file.write(b"\0")
    if needs_table:
        table = beam_table(beams)
        fits.append(path, table.data, table.header)

    return path

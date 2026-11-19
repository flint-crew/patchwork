"""The output frame of a mosaic: its pixel grid, coordinates and tiles"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from capn_crunch import add_options_to_parser, create_options_from_parser

from patchwork.coordinates import east_north, offsets, ra_dec, unit_vectors
from patchwork.io import Slices, celestial_wcs, read_image_info, spectral_axis
from patchwork.logging import logger, setup_logging
from patchwork.options import FrameOptions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from patchwork.io import ImageInfo

BoundingBox = tuple[int, int, int, int]
"""A (y_low, y_high, x_low, x_high) inclusive region of the frame in pixels"""


class Frame(NamedTuple):
    """The pixel grid of a mosaic"""

    wcs: WCS
    """Celestial WCS of the frame"""
    shape: tuple[int, int]
    """The (ny, nx) shape of the frame"""

    def header(
        self, reference: fits.Header | None = None, channels: slice | None = None
    ) -> fits.Header:
        """The header of the frame, keeping the non-celestial axes of a reference header.

        Those axes are made a single pixel long, except a FREQ axis when
        ``channels`` are given, which then holds those channels.

        Args:
            reference (fits.Header | None, optional): Header whose non-celestial axes are kept. Defaults to None.
            channels (slice | None, optional): The channels of the reference FREQ axis the frame holds. Defaults to None, a single plane.

        Returns:
            fits.Header: The header
        """
        extra_axes = [] if reference is None else list(range(3, reference["NAXIS"] + 1))
        spectral = (
            None if reference is None or channels is None else spectral_axis(reference)
        )
        header = fits.Header()
        header["NAXIS"] = 2 + len(extra_axes)
        header["NAXIS1"] = self.shape[1]
        header["NAXIS2"] = self.shape[0]
        for axis in extra_axes:
            header[f"NAXIS{axis}"] = 1

        header.update(self.wcs.to_header())
        header["WCSAXES"] = header["NAXIS"]
        if reference is None:
            return header

        keys = [
            f"{key}{axis}"
            for axis in extra_axes
            for key in ("CTYPE", "CRVAL", "CDELT", "CRPIX", "CUNIT")
        ]
        for key in (*keys, "RESTFRQ", "SPECSYS"):
            if key in reference:
                header[key] = reference[key]
        if spectral is not None and channels is not None:
            start, stop, _ = channels.indices(int(reference[f"NAXIS{spectral}"]))
            header[f"NAXIS{spectral}"] = stop - start
            header[f"CRPIX{spectral}"] = reference.get(f"CRPIX{spectral}", 1.0) - start
        return header

    def crop(self, box: BoundingBox) -> Frame:
        """The part of the frame within a bounding box, on the same grid."""
        wcs = self.wcs.deepcopy()
        wcs.wcs.crpix = wcs.wcs.crpix - [box[2], box[0]]
        wcs.wcs.set()
        return Frame(wcs=wcs, shape=(box[1] - box[0] + 1, box[3] - box[2] + 1))


def union_of_boxes(boxes: list[BoundingBox | None]) -> BoundingBox | None:
    """The smallest bounding box that holds every box, or None if there are none."""
    present = [box for box in boxes if box is not None]
    if not present:
        return None
    return (
        min(box[0] for box in present),
        max(box[1] for box in present),
        min(box[2] for box in present),
        max(box[3] for box in present),
    )


def shift_box(box: BoundingBox | None, origin: BoundingBox) -> BoundingBox | None:
    """A bounding box relative to the corner of another."""
    if box is None:
        return None
    return (
        box[0] - origin[0],
        box[1] - origin[0],
        box[2] - origin[2],
        box[3] - origin[2],
    )


def boundary_pixels(shape: tuple[int, int], samples: int = 32) -> NDArray[np.float64]:
    """Zero-based (x, y) pixels along the outer edges of an image, with ``samples`` per edge."""
    ny, nx = shape
    xs = np.linspace(-0.5, nx - 0.5, samples)
    ys = np.linspace(-0.5, ny - 0.5, samples)
    x = np.concatenate([xs, xs, np.full(samples, -0.5), np.full(samples, nx - 0.5)])
    y = np.concatenate([np.full(samples, -0.5), np.full(samples, ny - 0.5), ys, ys])
    return np.array([x, y])


def footprint_world(info: ImageInfo) -> NDArray[np.float64]:
    """The (RA, Dec) in degrees along the edges of an image."""
    x, y = boundary_pixels(info.shape)
    return np.array(info.wcs.pixel_to_world_values(x, y))


def image_centre(info: ImageInfo) -> tuple[float, float]:
    """The (RA, Dec) in degrees of the central pixel of an image."""
    ny, nx = info.shape
    ra, dec = info.wcs.pixel_to_world_values(nx // 2, ny // 2)
    return float(ra), float(dec)


def central_index(infos: list[ImageInfo]) -> int:
    """The index of the input whose centre is nearest the mean centre of all inputs."""
    centres = np.array([image_centre(info) for info in infos])
    vectors = unit_vectors(centres[:, 0], centres[:, 1])
    mean = np.mean(vectors, axis=1)
    return int(np.argmax(mean @ vectors))


def new_wcs(
    centre: tuple[float, float], pixel_scale_deg: float, projection: str = "SIN"
) -> WCS:
    """A north-up celestial WCS centred on a position, with its reference pixel at the origin."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    wcs.wcs.crval = [centre[0] % 360, centre[1]]
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.set()
    return wcs


def cover_inputs(wcs: WCS, infos: list[ImageInfo]) -> Frame:
    """Shift and size a WCS so that its frame just covers every input."""
    pixels = np.concatenate(
        [np.array(wcs.world_to_pixel_values(*footprint_world(info))) for info in infos],
        axis=1,
    )
    pixels = pixels[:, np.all(np.isfinite(pixels), axis=0)]
    # Edges that land on pixel edges only through round-off do not add a pixel
    x_low, y_low = np.floor(np.min(pixels, axis=1) + 0.5 + 1e-6).astype(int)
    x_high, y_high = np.ceil(np.max(pixels, axis=1) - 0.5 - 1e-6).astype(int)

    wcs = wcs.deepcopy()
    wcs.wcs.crpix = wcs.wcs.crpix - [x_low, y_low]
    wcs.wcs.set()
    return Frame(wcs=wcs, shape=(int(y_high - y_low + 1), int(x_high - x_low + 1)))


def reference_frame(
    infos: list[ImageInfo],
    reference_index: int | None = None,
    centre: tuple[float, float] | None = None,
    projection: str = "SIN",
) -> Frame:
    """A frame on the pixel grid of one input that covers every input.

    The grid is that of ``reference_index``, or of the central input. Given a
    ``centre``, a north-up grid with the pixel scale of the reference input is
    centred there instead.

    Args:
        infos (list[ImageInfo]): The inputs to cover
        reference_index (int | None, optional): The input whose grid is used. Defaults to None, the central input.
        centre (tuple[float, float] | None, optional): (RA, Dec) in degrees of a new frame centre. Defaults to None.
        projection (str, optional): Projection used when ``centre`` is given. Defaults to "SIN".

    Returns:
        Frame: The frame of the mosaic
    """
    index = central_index(infos) if reference_index is None else reference_index
    logger.info(f"Using the grid of {infos[index].path}")
    wcs = infos[index].wcs.deepcopy()
    if centre is not None:
        scale = float(np.sqrt(proj_plane_pixel_area(wcs)))
        wcs = new_wcs(centre=centre, pixel_scale_deg=scale, projection=projection)
    return cover_inputs(wcs=wcs, infos=infos)


def _most_covered_range(
    lows: NDArray[np.float64], highs: NDArray[np.float64]
) -> tuple[float, float]:
    """The range most inputs cover along one axis: the lower and upper edges that the most inputs reach past, counting an input once for starting below an edge and once for ending above it."""
    count_low = np.sum(lows[:, None] > lows[None, :], axis=1) + np.sum(
        highs[None, :] > lows[:, None], axis=1
    )
    count_high = np.sum(highs[None, :] > highs[:, None], axis=1) + np.sum(
        highs[:, None] > lows[None, :], axis=1
    )
    return float(lows[np.argmax(count_low)]), float(highs[np.argmax(count_high)])


def fitted_centre(
    infos: list[ImageInfo], centre_type: Literal["all", "most"] = "all"
) -> tuple[float, float]:
    """The centre of a fitted frame: the middle of all inputs, or of the region most of them cover.

    Ranges of RA and Dec break down towards a pole, where an input can span
    every RA. The ranges are instead those of offsets in a plane tangent to the
    sky at the mean centre of the inputs, which behave the same everywhere on
    the sky.

    Args:
        infos (list[ImageInfo]): The inputs
        centre_type (Literal["all", "most"], optional): Centre on all inputs, or on the region most of them cover. Defaults to "all".

    Returns:
        tuple[float, float]: The (RA, Dec) of the centre in degrees
    """
    centres = unit_vectors(*np.array([image_centre(info) for info in infos]).T)
    mean = np.mean(centres, axis=1)
    tangent_ra, tangent_dec = ra_dec(mean / np.linalg.norm(mean))
    tangent_point = (float(tangent_ra), float(tangent_dec))

    footprints = [
        np.array(offsets(tangent_point, unit_vectors(*footprint_world(info))))
        for info in infos
    ]
    lows = np.array([np.min(footprint, axis=1) for footprint in footprints])
    highs = np.array([np.max(footprint, axis=1) for footprint in footprints])
    if centre_type == "all":
        east, north = (np.min(lows, axis=0) + np.max(highs, axis=0)) / 2
    else:
        east, north = (
            np.mean(_most_covered_range(lows[:, axis], highs[:, axis]))
            for axis in range(2)
        )

    east_vector, north_vector = east_north(*tangent_point)
    centre = (
        east * east_vector
        + north * north_vector
        + np.sqrt(max(1 - east**2 - north**2, 0.0)) * unit_vectors(*tangent_point)
    )
    ra, dec = ra_dec(centre)
    return float(ra), float(dec)


def fitted_frame(infos: list[ImageInfo], options: FrameOptions) -> Frame:
    """A new north-up frame fitted to every input.

    The pixel scale is the median, minimum or maximum of the input pixel scales,
    and the centre is the middle of all inputs or of the region most of them
    cover (see ``fitted_centre``). Unless an image size is given the frame is
    sized to cover every input.

    Args:
        infos (list[ImageInfo]): The inputs to cover
        options (FrameOptions): How the frame is fitted

    Returns:
        Frame: The frame of the mosaic
    """
    if options.pixel_scale_arcsec is not None:
        scale = options.pixel_scale_arcsec / 3600
    else:
        scales = [float(np.sqrt(proj_plane_pixel_area(info.wcs))) for info in infos]
        percentile = {"min": 0, "median": 50, "max": 100}[options.pixel_scale_type]
        scale = float(np.percentile(scales, percentile))

    centre = options.centre or fitted_centre(infos, options.centre_type)
    logger.info(
        f"Fitted frame centred on {centre} with {scale * 3600:.3f} arcsec pixels"
    )

    wcs = new_wcs(centre=centre, pixel_scale_deg=scale, projection=options.projection)
    if options.image_size is None:
        return cover_inputs(wcs=wcs, infos=infos)

    nx, ny = options.image_size
    wcs.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    wcs.wcs.set()
    return Frame(wcs=wcs, shape=(ny, nx))


def frame_from_header(header: fits.Header | str | Path) -> Frame:
    """The frame described by a FITS header, or by the header of a FITS file."""
    if not isinstance(header, fits.Header):
        header = fits.getheader(header)
    return Frame(
        wcs=celestial_wcs(header), shape=(int(header["NAXIS2"]), int(header["NAXIS1"]))
    )


def build_frame(infos: list[ImageInfo], options: FrameOptions) -> Frame:
    """The frame of a mosaic of the inputs."""
    if options.frame_type == "fit":
        return fitted_frame(infos=infos, options=options)
    return reference_frame(
        infos=infos,
        reference_index=options.reference_index,
        centre=options.centre,
        projection=options.projection,
    )


def world_bounding_box(
    frame: Frame, world: NDArray[np.float64], margin: int = 2
) -> BoundingBox | None:
    """The region of the frame that encloses (RA, Dec) points, grown by a margin, or None if it misses the frame."""
    pixels = np.array(frame.wcs.world_to_pixel_values(*world))
    pixels = pixels[:, np.all(np.isfinite(pixels), axis=0)]
    if pixels.size == 0:
        return None

    ny, nx = frame.shape
    x_low, y_low = np.floor(np.min(pixels, axis=1)).astype(int) - margin
    x_high, y_high = np.ceil(np.max(pixels, axis=1)).astype(int) + margin
    box = (
        max(int(y_low), 0),
        min(int(y_high), ny - 1),
        max(int(x_low), 0),
        min(int(x_high), nx - 1),
    )
    if box[0] > box[1] or box[2] > box[3]:
        return None
    return box


def input_bounding_box(
    frame: Frame, info: ImageInfo, margin: int = 2
) -> BoundingBox | None:
    """The region of the frame an input covers, grown by a margin, or None if it covers none of it."""
    return world_bounding_box(frame, footprint_world(info), margin=margin)


def intersect_boxes(
    first: BoundingBox | None, second: BoundingBox | None
) -> BoundingBox | None:
    """The region two bounding boxes share, or None if they share none."""
    if first is None or second is None:
        return None
    box = (
        max(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        min(first[3], second[3]),
    )
    if box[0] > box[1] or box[2] > box[3]:
        return None
    return box


def tiles(frame: Frame, tile_size: int = 512) -> list[Slices]:
    """Split a frame into square (y, x) tiles of up to ``tile_size`` pixels."""
    ny, nx = frame.shape
    return [
        (slice(y, min(y + tile_size, ny)), slice(x, min(x + tile_size, nx)))
        for y in range(0, ny, tile_size)
        for x in range(0, nx, tile_size)
    ]


def strips(frame: Frame, tile_size: int = 512) -> list[Slices]:
    """Split a frame into strips of whole rows, each about as large as a square tile of ``tile_size`` pixels.

    Each strip of a channel is one run of bytes in a FITS file, so tasks
    writing neighbouring strips share at most the blocks at their ends.
    """
    ny, nx = frame.shape
    rows = max(1, -(-(tile_size**2) // nx))
    return [(slice(y, min(y + rows, ny)), slice(0, nx)) for y in range(0, ny, rows)]


def tile_region(tile: Slices, box: BoundingBox | None) -> Slices | None:
    """The part of a tile within a bounding box, relative to the tile, or None if they share no pixels."""
    if box is None:
        return None
    ys, xs = tile
    y_low, y_high = max(ys.start, box[0]), min(ys.stop - 1, box[1])
    x_low, x_high = max(xs.start, box[2]), min(xs.stop - 1, box[3])
    if y_low > y_high or x_low > x_high:
        return None
    return (
        slice(y_low - ys.start, y_high - ys.start + 1),
        slice(x_low - xs.start, x_high - xs.start + 1),
    )


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """The parser of the ``frame`` command."""
    if parser is None:
        parser = ArgumentParser(
            description="Construct the output frame of a mosaic without making it"
        )
    parser.add_argument(
        "images", nargs="+", type=Path, help="The images the frame covers"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Text file the frame header is written to. If not given it is printed",
    )
    return add_options_to_parser(parser=parser, options_class=FrameOptions)


def cli(args: Namespace | None = None) -> None:
    """Run the ``frame`` command."""
    setup_logging()
    args = get_parser().parse_args() if args is None else args
    infos = [read_image_info(image) for image in args.images]
    frame = build_frame(
        infos,
        create_options_from_parser(parser_namespace=args, options_class=FrameOptions),
    )
    header = frame.header(infos[0].header)
    if args.output is None:
        logger.info(f"Frame header:\n{header.tostring(sep=chr(10)).strip()}")
    else:
        header.totextfile(args.output, overwrite=True)
        logger.info(f"Written the frame header to {args.output}")

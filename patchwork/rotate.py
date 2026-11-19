"""Rotate the Q and U of images or cubes from the frame they were imaged in to north at each pixel.

An interferometric image references the polarisation angle of every pixel to
one direction: north at its phase centre or, for ASKAP, at the boresight of the
observation, carried across the field as a fixed direction of the feeds.
``patchwork mosaic`` turns each input into the frame of the mosaic as it
resamples it. This command does the same turn on an input's own grid, writing a
new image beside it whose Q and U are referenced to north at each pixel, the IAU
convention, with POLFRAME keywords that say so. A mosaic of rotated images then
needs no frame options, and differs from one rotated in flight only by the
rounding of float32.

The angle is the same for every channel of a pixel, so each block of rows is
rotated across every channel at once, in threads if asked.
"""

from __future__ import annotations

import threading
import time
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits
from capn_crunch import add_options_to_parser, create_options_from_parser

from patchwork import __version__
from patchwork.coordinates import tile_coordinates
from patchwork.io import (
    BeamShape,
    ImageInfo,
    check_output_fits,
    create_output_fits,
    read_block,
    read_image_info,
    suffixed_name,
    write_block,
)
from patchwork.logging import logger, setup_logging
from patchwork.mosaic import add_log_level_argument
from patchwork.options import PolarisationOptions, RotateOptions
from patchwork.plan import group_inputs, input_frame_centre, rotates
from patchwork.polarisation import (
    NORTH,
    STOKES_CODES,
    frame_angle,
    rotate_linear,
    write_frame_keywords,
)
from patchwork.resample import tile_threads
from patchwork.tile import PlaneWeight

ROTATE_BLOCK_BYTES = 256 * 2**20
"""The memory the Q and U of one block of rows take while they are rotated, in float64"""


def rotated_name(path: str | Path, suffix: str = "rot") -> Path:
    """The name of a rotated image: ``suffix`` inserted before a final ``.fits``, or added with ``.fits`` after any other name, so it never names the image itself."""
    return suffixed_name(path, suffix)


def rows_per_block(info: ImageInfo, budget: int | None = None) -> int:
    """The rows of an image rotated at once, so that the Q and U of a block fit in ``budget`` bytes (``ROTATE_BLOCK_BYTES``) as float64."""
    ny, nx = info.shape
    per_row = 2 * max(info.n_channels, 1) * nx * 8
    budget = ROTATE_BLOCK_BYTES if budget is None else budget
    return max(1, min(ny, budget // per_row))


def _output_beams(info: ImageInfo) -> tuple[BeamShape, ...]:
    """The beams of the channels, to be recorded again in the output, or none to leave its header's as they are."""
    beams = [beam for beam in info.channel_beams if beam is not None]
    if len(beams) != len(info.channel_beams):
        return ()
    return tuple(beams)


def rotate_pair(
    q_path: str | Path,
    u_path: str | Path,
    centre: tuple[float, float],
    q_output: str | Path | None = None,
    u_output: str | Path | None = None,
    suffix: str = "rot",
    overwrite: bool = False,
    threads: int = 1,
) -> tuple[Path, Path]:
    """Rotate the Q and U of one input from north at ``centre`` to north at each pixel, writing new images.

    This is the rotation ``patchwork mosaic`` applies to an input when it is
    referenced to north at ``centre`` and the mosaic to north at each pixel.
    The outputs copy the headers of the inputs, record the new frame in their
    POLFRAME keywords, and are float32.

    Args:
        q_path (str | Path): The Stokes Q image or cube
        u_path (str | Path): The Stokes U image or cube, on the same grid
        centre (tuple[float, float]): The (RA, Dec) in degrees whose north the Q and U are referenced to
        q_output (str | Path | None, optional): Where to write the rotated Q. Defaults to None, ``rotated_name`` of the input.
        u_output (str | Path | None, optional): Where to write the rotated U. Defaults to None, ``rotated_name`` of the input.
        suffix (str, optional): The suffix of outputs that are not named. Defaults to "rot".
        overwrite (bool, optional): Replace existing outputs. Defaults to False.
        threads (int, optional): Threads that rotate blocks of rows at once; 0 uses every CPU. Defaults to 1.

    Raises:
        ValueError: Raised if the two images are not on one grid

    Returns:
        tuple[Path, Path]: The rotated Q and U
    """
    q_info, u_info = read_image_info(q_path), read_image_info(u_path)
    if q_info.shape != u_info.shape or q_info.n_channels != u_info.n_channels:
        msg = (
            f"{q_path} is {q_info.shape} with {q_info.n_channels} channels but {u_path} is "
            f"{u_info.shape} with {u_info.n_channels}; Q and U are rotated together, on one grid"
        )
        raise ValueError(msg)
    outputs = (
        Path(q_output) if q_output is not None else rotated_name(q_path, suffix),
        Path(u_output) if u_output is not None else rotated_name(u_path, suffix),
    )
    for output in outputs:
        check_output_fits(output, overwrite=overwrite, input_fits=[q_path, u_path])
    for info, output in zip((q_info, u_info), outputs, strict=True):
        header = info.header.copy()
        write_frame_keywords(header, NORTH)
        header["HISTORY"] = (
            f"Rotated by patchwork {__version__} from north at RA {centre[0]:.5f}, "
            f"Dec {centre[1]:.5f} to north at each pixel"
        )
        create_output_fits(output, header, _output_beams(info))

    ny, nx = q_info.shape
    rows = rows_per_block(q_info)
    blocks = [slice(start, min(start + rows, ny)) for start in range(0, ny, rows)]
    threads = tile_threads(threads)
    logger.info(
        f"Rotating {q_path} and {u_path} from north at RA {centre[0]:.5f}, Dec {centre[1]:.5f} "
        f"to north at each pixel: {q_info.n_channels} channels in {len(blocks)} blocks of rows "
        f"on {threads} threads"
    )
    with (
        fits.open(outputs[0], mode="update", memmap=True) as q_hdul,
        fits.open(outputs[1], mode="update", memmap=True) as u_hdul,
    ):
        q_hdu, u_hdu = q_hdul[0], u_hdul[0]
        # Map the data once, before any thread writes to it
        q_hdu.data  # noqa: B018
        u_hdu.data  # noqa: B018

        done = 0
        counting = threading.Lock()

        def rotate_block(block: slice) -> None:
            nonlocal done
            started = time.perf_counter()
            region = (block, slice(0, nx))
            angle = frame_angle(centre, tile_coordinates(q_info.wcs, region).vectors)
            q, u = rotate_linear(
                np.asarray(read_block(q_path, region), dtype=np.float64),
                np.asarray(read_block(u_path, region), dtype=np.float64),
                angle,
            )
            write_block(q_hdu, region, q)
            write_block(u_hdu, region, u)
            with counting:
                done += block.stop - block.start
                rows_done = done
            logger.debug(
                f"Rotated rows {block.start}-{block.stop} of {q_path} and {u_path} "
                f"in {time.perf_counter() - started:.1f} s; {rows_done} of {ny} rows "
                f"({rows_done / ny:.0%}) done"
            )

        if threads > 1 and len(blocks) > 1:
            with ThreadPoolExecutor(min(threads, len(blocks))) as pool:
                # Consumed, so that a block that fails is raised here
                list(pool.map(rotate_block, blocks))
        else:
            for block in blocks:
                rotate_block(block)
    for output in outputs:
        logger.info(f"Written {output}")
    return outputs


def rotate_cubes(
    images: list[Path],
    options: RotateOptions = RotateOptions(),  # noqa: B008
    overwrite: bool = False,
    threads: int = 1,
) -> list[Path]:
    """Rotate the Q and U of images or cubes to north at each pixel, each written beside its input.

    Images on one pixel grid are a pair, as ``patchwork mosaic`` groups them,
    and the frame each pair is in comes from its POLFRAME keywords or the
    options as it does there (``patchwork.plan.input_frame_centre``). Every
    output is checked before any is written.

    Args:
        images (list[Path]): The Q and U images or cubes, in any order
        options (RotateOptions, optional): What the inputs are referenced to, and how the outputs are named. Defaults to RotateOptions().
        overwrite (bool, optional): Replace existing outputs. Defaults to False.
        threads (int, optional): Threads that rotate blocks of rows at once; 0 uses every CPU. Defaults to 1.

    Raises:
        ValueError: Raised if an image is not of Q or U, a Q or U has no partner on its grid, or a pair is already referenced to north at each pixel

    Returns:
        list[Path]: The rotated image of each input, in the order given
    """
    infos = [read_image_info(image) for image in images]
    linear = (STOKES_CODES["Q"], STOKES_CODES["U"])
    others = [str(info.path) for info in infos if info.stokes not in linear]
    if others:
        msg = f"Only Q and U are rotated, but these are not: {', '.join(others)}"
        raise ValueError(msg)
    # Images matched by a pattern can include ones rotated before, so no output may be any input
    for image in images:
        check_output_fits(
            rotated_name(image, options.suffix), overwrite=overwrite, input_fits=images
        )

    polarisation = PolarisationOptions(
        input_frame=options.input_frame, frame_centre=options.frame_centre
    )
    pairs = []
    for item in group_inputs(infos, [PlaneWeight()] * len(infos)):
        if not rotates(item.stokes):
            have = "Q" if STOKES_CODES["Q"] in item.stokes else "U"
            msg = (
                f"{item.info.path} is Stokes {have}, but there is no "
                f"{'U' if have == 'Q' else 'Q'} on its pixel grid; both are needed to rotate"
            )
            raise ValueError(msg)
        centre = input_frame_centre(item, polarisation)
        if centre is None:
            msg = (
                f"The Q and U of {item.info.path} are already referenced to north at each "
                "pixel, so there is nothing to rotate"
            )
            raise ValueError(msg)
        q_path = item.planes[item.stokes.index(STOKES_CODES["Q"])]
        u_path = item.planes[item.stokes.index(STOKES_CODES["U"])]
        pairs.append((q_path, u_path, centre))

    outputs: dict[Path, Path] = {}
    for q_path, u_path, centre in pairs:
        outputs[q_path], outputs[u_path] = rotate_pair(
            q_path,
            u_path,
            centre,
            suffix=options.suffix,
            overwrite=overwrite,
            threads=threads,
        )
    return [outputs[info.path] for info in infos]


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """The parser of the ``rotate`` command."""
    if parser is None:
        parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="The Q and U images or cubes to rotate, in any order; those on one pixel grid are a pair",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Threads that rotate blocks of rows at once; 0 uses every CPU",
    )
    return add_log_level_argument(
        add_options_to_parser(parser=parser, options_class=RotateOptions)
    )


def cli(args: Namespace | None = None) -> None:
    """Run the ``rotate`` command."""
    args = get_parser().parse_args() if args is None else args
    setup_logging(args.log_level)
    rotate_cubes(
        images=args.images,
        options=create_options_from_parser(
            parser_namespace=args, options_class=RotateOptions
        ),
        overwrite=args.overwrite,
        threads=args.threads,
    )


if __name__ == "__main__":
    cli()

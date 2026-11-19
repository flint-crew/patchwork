"""The ``patchwork`` command line interface. Each sub-command is configured by the
``get_parser`` of its module, which is passed the sub-parser to add arguments to."""

from __future__ import annotations

from argparse import ArgumentParser

from patchwork import convolve, frame, mosaic, rotate
from patchwork.logging import logger, setup_logging


def get_parser() -> ArgumentParser:
    """The parser of the ``patchwork`` command."""
    parser = ArgumentParser(description="Linear mosaicking of radio images.")
    subparser = parser.add_subparsers(dest="mode", help="Operation mode of patchwork")

    mosaic.get_parser(
        parser=subparser.add_parser(
            "mosaic",
            help="Mosaic images onto a common frame, weighting each by its primary beam and noise.",
        )
    )
    convolve.get_parser(
        parser=subparser.add_parser(
            "convolve",
            help="Convolve images to a target beam, or to the smallest beam that encloses all of theirs.",
        )
    )
    frame.get_parser(
        parser=subparser.add_parser(
            "frame",
            help="Construct the output frame of a mosaic and write its header, without making the mosaic.",
        )
    )
    rotate.get_parser(
        parser=subparser.add_parser(
            "rotate",
            help="Rotate the Q and U of images or cubes from the frame they were imaged in to north at each pixel, recording it in their headers.",
        )
    )
    return parser


def cli() -> None:
    """Run the ``patchwork`` command."""
    setup_logging()
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == "mosaic":
        mosaic.cli(args=args)
    elif args.mode == "convolve":
        convolve.cli(args=args)
    elif args.mode == "frame":
        frame.cli(args=args)
    elif args.mode == "rotate":
        rotate.cli(args=args)
    else:
        logger.error(f"Supplied mode {args.mode} is not known. ")
        parser.print_help()

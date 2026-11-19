"""The patchwork pipeline: mosaic images or cubes of Stokes I, Q, U and V on a dask cluster.

The images are given as one list, in any order, with a weights log or weight
image for each. Each one's Stokes parameter comes from its header, and those
on one pixel grid are taken to be one beam. With --convolve the inputs are
first convolved to a common beam, one to each task. Each task then makes one
tile of the mosaic over some channels, for every Stokes parameter at once.
"""

from __future__ import annotations

import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from distributed import Client

from patchwork.logging import logger, setup_logging
from patchwork.mosaic import (
    add_log_level_argument,
    add_mosaic_arguments,
    mosaic,
    options_from_args,
)
from patchwork.options import ConvolveOptions, MosaicOptions, RunOptions
from patchwork.pipeline.cluster import LogOnWorkers, get_cluster

if TYPE_CHECKING:
    from patchwork.plan import MosaicNames

SCATTER_ATTEMPTS = 5
"""Times the plan is offered to the cluster before giving up, as workers may leave while it is sent"""


def run_pipeline(
    images: list[Path],
    output_base: str | Path,
    client: Client,
    options: MosaicOptions = MosaicOptions(),  # noqa: B008
    run_options: RunOptions = RunOptions(),  # noqa: B008
    convolve_options: ConvolveOptions = ConvolveOptions(),  # noqa: B008
    weights: list[Path] | None = None,
    stokes_i_images: list[Path] | None = None,
    overwrite: bool = False,
) -> MosaicNames:
    """Mosaic images or cubes of Stokes I, Q, U and V on a dask cluster.

    Args:
        images (list[Path]): The images or cubes, in any order
        output_base (str | Path): Base name of the output files
        client (Client): Client of the cluster to run on
        options (MosaicOptions, optional): The mosaicking options. Defaults to MosaicOptions().
        run_options (RunOptions, optional): Which channels are mosaicked, whether the inputs are convolved first, and how the mosaic is written. Defaults to RunOptions().
        convolve_options (ConvolveOptions, optional): How the common beam is found and convolved to. Defaults to ConvolveOptions().
        weights (list[Path] | None, optional): A weights log or weight image for each image, in the same order. Defaults to None, weighting by noise.
        stokes_i_images (list[Path] | None, optional): Stokes I images to remove the leakage with, but not mosaic. Defaults to None.
        overwrite (bool, optional): Replace existing outputs. Defaults to False.

    Returns:
        MosaicNames: The files the mosaic was written to
    """

    def share(plan: Any) -> Any:
        # A plan is a tuple, which scatter would otherwise split into its fields.
        # It is not broadcast: the workers fetch it from each other as their
        # first tasks need it, whereas broadcasting replicates it to a list of
        # workers that adaptive scaling may retire meanwhile, which the
        # scheduler raises as a KeyError. The same can befall the one worker
        # it is sent to, so the scatter is retried
        for attempt in range(1, SCATTER_ATTEMPTS):
            try:
                return client.scatter([plan])[0]
            except (KeyError, OSError) as error:
                logger.warning(
                    f"Sending the plan to the workers failed ({error!r}), "
                    f"retrying ({attempt} of {SCATTER_ATTEMPTS - 1})"
                )
                time.sleep(attempt)
        return client.scatter([plan])[0]

    return mosaic(
        images=images,
        output_base=output_base,
        options=options,
        run_options=run_options,
        convolve_options=convolve_options,
        weights=weights,
        stokes_i_images=stokes_i_images,
        overwrite=overwrite,
        # The tasks write files, so are not pure: each call is a task of its own
        executor=client.get_executor(pure=False),
        share=share,
        capacity=lambda: sum(client.nthreads().values()),
    )


def get_parser() -> ArgumentParser:
    """The parser of the ``patchwork-pipeline`` command."""
    parser = add_log_level_argument(
        add_mosaic_arguments(ArgumentParser(description=__doc__))
    )
    parser.add_argument(
        "--cluster-config",
        type=Path,
        default=None,
        help="YAML specification of the dask cluster to run on. If not given a local cluster is started",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of dask workers, when the cluster does not adapt",
    )
    return parser


def cli(args: Namespace | None = None) -> None:
    """Run the ``patchwork-pipeline`` command."""
    setup_logging()
    args = get_parser().parse_args() if args is None else args
    setup_logging(args.log_level)
    options, run_options, convolve_options = options_from_args(args)
    with (
        get_cluster(args.cluster_config, n_workers=args.n_workers) as cluster,
        Client(cluster) as client,
    ):
        client.register_plugin(LogOnWorkers(args.log_level))
        run_pipeline(
            images=args.images,
            output_base=args.output_base,
            client=client,
            options=options,
            run_options=run_options,
            convolve_options=convolve_options,
            weights=args.weights,
            stokes_i_images=args.stokes_i_images,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    cli()

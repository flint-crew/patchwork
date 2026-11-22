"""Linear mosaicking of images and cubes of Stokes I, Q, U and V onto a common frame.

Images and cubes are given as one list, with a weight for each. Each one's
Stokes parameter is read from its STOKES axis, and those on one pixel grid are
the Stokes parameters of one beam. Each input is resampled onto the frame,
weighted by its primary beam and noise, and the inputs are
combined tile by tile, every channel and Stokes parameter in one pass: rotating
Q and U into the frame of the mosaic, and removing the leakage of Stokes I. The
inputs can first be convolved to a common beam.

The work is handed to a ``concurrent.futures.Executor``: threads or processes
here, or dask workers through ``patchwork.pipeline.flow``.
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import multiprocessing
import pickle
import tempfile
import time
import uuid
from argparse import ArgumentParser, Namespace
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from capn_crunch import add_options_to_parser, create_options_from_parser

from patchwork.convolve import (
    ConvolveBackend,
    channel_target_beams,
    convolve_cube,
    convolved_name,
)
from patchwork.io import BeamShape, check_output_fits, read_image_info
from patchwork.logging import logger, setup_logging
from patchwork.options import (
    ConvolveOptions,
    MosaicOptions,
    RunOptions,
    add_mosaic_options_to_parser,
    create_mosaic_options_from_parser,
)
from patchwork.plan import (
    STOKES_NAMES,
    MosaicNames,
    check_noise,
    check_spectral_axes,
    group_inputs,
    input_frame_centre,
    needs_image_weight,
    output_names,
    plan_mosaic,
    read_weight,
    rotates,
)
from patchwork.polarisation import STOKES_CODES
from patchwork.primary_beam import beam_model
from patchwork.resample import available_cpus, tile_threads
from patchwork.tile import PlaneWeight
from patchwork.weights import measure_channel_weights
from patchwork.write import (
    _duration,
    bounded_results,
    create_outputs,
    logged_progress,
    run_tiles,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


class _PlanFile:
    """A mosaic's plan in a file. A process it is sent to reads the file once, however many of its tasks are sent it"""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def write(cls, directory: Path, plan: Any) -> _PlanFile:
        """Write a plan to a new file in a directory."""
        path = directory / f"plan-{uuid.uuid4().hex}.pickle"
        with path.open("wb") as file:
            pickle.dump(plan, file, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(path)

    def __reduce__(self) -> tuple[Callable[[str], Any], tuple[str]]:
        # Unpickled in a process, it becomes the plan itself, so tasks need not know it was sent this way
        return _read_plan, (str(self.path),)


@functools.lru_cache(maxsize=1)
def _read_plan(path: str) -> Any:
    """The plan in a file, which each process reads once."""
    with Path(path).open("rb") as file:
        return pickle.load(file)


class _Relay(logging.Handler):
    """Hands the messages of patchwork's processes to the logger they were logged to in this process, as though logged here"""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


def _start_process(level: int, messages: Any) -> None:
    """Send patchwork's messages from a new process to a queue the process that started it reads."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(logging.handlers.QueueHandler(messages))
    logger.setLevel(level)
    logger.propagate = False


@contextmanager
def _own_pool(
    max_workers: int, processes: bool
) -> Iterator[tuple[Executor, Callable[[Any], Any] | None]]:
    """A pool of threads, or of processes along with a way of sending the plan to them once.

    What a process logs is passed to this process, so it is printed, or reaches
    an application's own handlers, as what a thread logs would be.

    Raises:
        ValueError: Raised if processes are asked for in a daemonic process, which cannot start them
    """
    if not processes:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            yield pool, None
        return
    if multiprocessing.current_process().daemon:
        msg = (
            "patchwork cannot start processes from a daemonic process, as a dask worker is. "
            "Use threads, or run patchwork in a subprocess"
        )
        raise ValueError(msg)
    # Spawned, not forked, so a process never inherits a lock some thread of this one held
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    listener = logging.handlers.QueueListener(messages, _Relay())
    listener.start()
    try:
        with (
            tempfile.TemporaryDirectory(prefix="patchwork-plan-") as directory,
            ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=context,
                initializer=_start_process,
                initargs=(logger.getEffectiveLevel(), messages),
            ) as pool,
        ):
            yield pool, functools.partial(_PlanFile.write, Path(directory))
    finally:
        # Every process has stopped, so this passes on what is left and ends
        listener.stop()
        messages.close()
        messages.join_thread()


def mosaic(
    images: list[Path],
    output_base: str | Path,
    options: MosaicOptions = MosaicOptions(),  # noqa: B008
    run_options: RunOptions = RunOptions(),  # noqa: B008
    convolve_options: ConvolveOptions = ConvolveOptions(),  # noqa: B008
    weights: list[Path] | None = None,
    stokes_i_images: list[Path] | None = None,
    beam: BeamShape | None = None,
    beam_numbers: list[int] | None = None,
    overwrite: bool = False,
    executor: Executor | None = None,
    share: Callable[[Any], Any] | None = None,
    max_workers: int = 1,
    capacity: Callable[[], int] | None = None,
    processes: bool = False,
) -> MosaicNames:
    """Mosaic images or cubes of Stokes I, Q, U and V, optionally convolving them to a common beam first.

    Args:
        images (list[Path]): The images or cubes, in any order: their Stokes parameters come from their headers, and their beams from their grids
        output_base (str | Path): Base name of the output files
        options (MosaicOptions, optional): The mosaicking options. Defaults to MosaicOptions().
        run_options (RunOptions, optional): Which channels are mosaicked, whether the inputs are convolved first, and how the mosaic is written. Defaults to RunOptions().
        convolve_options (ConvolveOptions, optional): How the common beam is found and convolved to. Defaults to ConvolveOptions().
        weights (list[Path] | None, optional): A weights log (``.txt``) or weight image for each image, in the same order. Defaults to None, weighting by the noise of each channel.
        stokes_i_images (list[Path] | None, optional): Stokes I images, matched to the beams by their grid, to remove the leakage with but not mosaic. Defaults to None, using the beams' own.
        beam (BeamShape | None, optional): Restoring beam recorded for every channel of the mosaic. Defaults to None, the beams convolved to, or else those of the first input.
        beam_numbers (list[int] | None, optional): The holography beam of each image. Defaults to None, read from their names.
        overwrite (bool, optional): Replace existing outputs. Defaults to False.
        executor (Executor | None, optional): Runs the tasks. Defaults to None, a pool of ``max_workers`` threads, or processes with ``processes``.
        share (Callable[[Any], Any] | None, optional): Turns what every task needs into what is sent to it, such as dask's ``scatter``. Defaults to None, sending it as it is to threads, and once to each process.
        max_workers (int, optional): Threads or processes of the default executor, or how many tasks a given executor runs at once. Defaults to 1.
        capacity (Callable[[], int] | None, optional): How many tasks the executor runs at once, asked again as tiles are made, such as the threads of a dask cluster as it scales. Defaults to None, ``max_workers``.
        processes (bool, optional): Make the default executor a pool of processes rather than threads. Python code holding the GIL then holds up only its own process, which matters on machines with many cores. What they log is passed to this process. The processes are spawned, so a script that calls this must do so under ``if __name__ == "__main__":``. Defaults to False.

    Raises:
        ValueError: Raised if the weights or beam numbers do not match the images, the channels are not among theirs, or ``processes`` is given with an executor or in a daemonic process

    Returns:
        MosaicNames: The files the mosaic was written to
    """
    for name, given in (("weights", weights), ("beam numbers", beam_numbers)):
        if given is not None and len(given) != len(images):
            msg = f"Got {len(given)} {name} for {len(images)} images"
            raise ValueError(msg)
    if processes and executor is not None:
        msg = "processes makes patchwork's own pool of processes, so cannot be given with an executor"
        raise ValueError(msg)
    if executor is None:
        if options.resample.tile_threads == 0:
            # The threads or processes of the pool each make a tile at once, on this machine
            shared = max(1, available_cpus() // max(max_workers, 1))
            options = options._replace(
                resample=options.resample.model_copy(update={"tile_threads": shared})
            )
        with _own_pool(max_workers, processes) as (pool, pool_share):
            return mosaic(
                images,
                output_base,
                options,
                run_options,
                convolve_options,
                weights,
                stokes_i_images,
                beam,
                beam_numbers,
                overwrite,
                executor=pool,
                share=share or pool_share,
                max_workers=max_workers,
                capacity=capacity,
            )

    infos = [
        read_image_info(
            image, beam_number=None if beam_numbers is None else beam_numbers[index]
        )
        for index, image in enumerate(images)
    ]
    stokes_i = [read_image_info(image) for image in stokes_i_images or []]
    check_spectral_axes(infos + stokes_i)
    n_channels = infos[0].n_channels
    start, stop = run_options.channels or (0, n_channels)
    if not 0 <= start < stop <= n_channels:
        msg = f"Channels {start} to {stop} are not a range within the {n_channels} channels of the inputs"
        raise ValueError(msg)
    channels = slice(start, stop)
    given_beams = None if beam is None else (beam,) * n_channels
    plane_weights = (
        None
        if weights is None
        else [
            read_weight(weight, info)
            for weight, info in zip(weights, infos, strict=True)
        ]
    )
    remove_leakage = options.polarisation.remove_leakage
    # Check the images form inputs, whose Q and U frames are known, and nothing will be written over them, before any work
    for item in group_inputs(
        infos, plane_weights or [PlaneWeight()] * len(infos), remove_leakage, stokes_i
    ):
        if rotates(item.stokes):
            input_frame_centre(item, options.polarisation)
    # Build each input's primary beam now, so a model that cannot be built fails before any convolving
    leaky = remove_leakage and any(info.stokes != STOKES_CODES["I"] for info in infos)
    if options.weight.weight_type != "weight_image" or leaky:
        for info in infos:
            beam_model(options.primary_beam, info, leakage=leaky)
    names = output_names(
        output_base,
        tuple(sorted({info.stokes or 0 for info in infos})),
        cube=n_channels > 1,
        output_format=run_options.output_format,
        noise=options.weight.noise_image,
    )
    check_noise(options.weight, weighted_by_images=weights is not None)
    originals = [*images, *(weights or []), *(stokes_i_images or [])]
    for output in (*names.images, *names.weights, *names.noises):
        check_output_fits(output, overwrite=overwrite, input_fits=originals)

    stages: list[tuple[str, float]] = []
    stage_started = time.monotonic()
    beams = None
    convolved: list[Path] = []
    if run_options.convolve != "none":
        beams = channel_target_beams(
            infos + stokes_i, run_options.convolve, convolve_options, channels
        )
        # Runs over different channels each convolve to their own files
        suffix = convolve_options.suffix + (
            f".ch{start}-{stop}" if run_options.channels else ""
        )
        to_convolve = infos + stokes_i
        outputs = [convolved_name(info.path, suffix) for info in to_convolve]
        for output in outputs:
            check_output_fits(output, overwrite=overwrite, input_fits=originals)
        logger.info(
            f"Convolving {len(to_convolve)} inputs to {len(set(beams[channels]))} beams"
        )
        # The outputs are known, so the cubes are taken as they finish, and counted
        calls = [
            (
                _convolve_cube,
                (
                    info.path,
                    beams,
                    output,
                    convolve_options.beam_cutoff_arcsec,
                    overwrite,
                    channels,
                    options.resample.tile_threads,
                    convolve_options.convolve_backend,
                ),
            )
            for info, output in zip(to_convolve, outputs, strict=True)
        ]
        list(
            logged_progress(
                bounded_results(executor, calls, lambda: len(calls)),
                len(calls),
                "convolved cubes",
            )
        )
        convolved = list(outputs)
        stages.append(("convolving", time.monotonic() - stage_started))
        stage_started = time.monotonic()
        n_images = len(infos)
        infos = [
            read_image_info(path, beam_number=info.beam_number)
            for path, info in zip(convolved[:n_images], infos, strict=True)
        ]
        stokes_i = [read_image_info(path) for path in convolved[n_images:]]

    if plane_weights is None:
        measure = needs_image_weight(options.weight, None)
        plane_weights = [
            PlaneWeight(values=values)
            for values in _results_in_order(
                executor.submit(
                    _measure_weights,
                    info.path,
                    mode=options.weight.image_weight if measure else "none",
                    stride=options.weight.weight_stride,
                    channels=channels,
                    threads=options.resample.tile_threads,
                )
                for info in infos
            )
        ]

        stages.append(("measuring noise", time.monotonic() - stage_started))
        stage_started = time.monotonic()

    inputs = group_inputs(infos, plane_weights, remove_leakage, stokes_i)
    logger.info(
        f"Found {len(inputs)} inputs of Stokes {', '.join(STOKES_NAMES.get(code, '?') for code in inputs[0].stokes)}"
    )
    plan = plan_mosaic(
        inputs,
        names,
        options,
        run_options,
        beams=beams if given_beams is None else given_beams,
    )
    create_outputs(plan, run_options)
    run_tiles(
        plan,
        options,
        run_options,
        executor,
        share,
        capacity=capacity or (lambda: max_workers),
    )

    stages.append(("making the mosaic", time.monotonic() - stage_started))

    if convolved and not run_options.keep_convolved:
        for path in convolved:
            path.unlink()
        logger.info(f"Removed the {len(convolved)} convolved inputs")
    logger.info(
        f"Took {_duration(sum(seconds for _, seconds in stages))}: "
        + ", ".join(f"{name} {_duration(seconds)}" for name, seconds in stages)
    )
    logger.info(
        f"Written {', '.join(map(str, (*names.images, *names.weights, *names.noises)))}"
    )
    return names


def _convolve_cube(
    path: Path,
    beams: tuple[BeamShape, ...],
    output: Path,
    cutoff_arcsec: float | None,
    overwrite: bool,
    channels: slice | None,
    threads: int,
    backend: ConvolveBackend,
) -> Path:
    """Convolve a cube in the threads a tile would be made with, resolved where it runs, as ``_measure_weights`` does."""
    return convolve_cube(
        path,
        beams,
        output,
        cutoff_arcsec,
        overwrite,
        channels,
        threads=tile_threads(threads),
        backend=backend,
    )


def _measure_weights(
    path: Path,
    mode: Literal["mad", "std", "none"],
    stride: int,
    channels: slice | None,
    threads: int,
) -> tuple[float, ...]:
    """Measure the weights of a cube's channels in the threads a tile would be made with, with whichever backend.

    Runs where the tiles are made, so ``threads`` of 0 is resolved there: on a
    dask worker, to its CPUs over its threads.
    """
    return measure_channel_weights(
        path,
        mode=mode,
        stride=stride,
        channels=channels,
        threads=tile_threads(threads),
    )


def _results_in_order(futures: Iterable[Future[Any]]) -> list[Any]:
    """The results of futures, in the order they were submitted, cancelling those still waiting should one fail."""
    submitted = list(futures)
    try:
        return [future.result() for future in submitted]
    finally:
        for future in submitted:
            future.cancel()


def add_mosaic_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add the inputs, outputs and options of a mosaic to a parser."""
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="The images or cubes, in any order. Their Stokes parameters are read from their headers, and those on one pixel grid are taken to be one beam",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=Path,
        default=None,
        help="A weights log (.txt) or weight image for each image, in the same order. If not given each channel is weighted by its noise",
    )
    parser.add_argument(
        "--stokes-i-images",
        nargs="+",
        type=Path,
        default=None,
        help="Stokes I images to remove the leakage with, but not mosaic, matched to the beams by their grid",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("mosaic"),
        help="Base name of the outputs: <base>.<stokes>.mosaic.fits and <base>.<stokes>.weight.fits, or <base>.<stokes>.cube.fits and <base>.<stokes>.weight.cube.fits for cubes (or .zarr)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing outputs"
    )
    add_mosaic_options_to_parser(parser)
    add_options_to_parser(parser=parser, options_class=RunOptions)
    return add_options_to_parser(parser=parser, options_class=ConvolveOptions)


def options_from_args(
    args: Namespace,
) -> tuple[MosaicOptions, RunOptions, ConvolveOptions]:
    """The options of a mosaic from parsed arguments."""
    return (
        create_mosaic_options_from_parser(args),
        create_options_from_parser(parser_namespace=args, options_class=RunOptions),
        create_options_from_parser(
            parser_namespace=args, options_class=ConvolveOptions
        ),
    )


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """The parser of the ``mosaic`` command."""
    if parser is None:
        parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of threads, or processes with --processes, that convolve, measure noise and make tiles",
    )
    parser.add_argument(
        "--processes",
        action="store_true",
        help="Work in --max-workers processes rather than threads. Threads share Python's lock (the GIL), so on a machine with many cores they leave some idle",
    )
    return add_log_level_argument(add_mosaic_arguments(parser))


def add_log_level_argument(parser: ArgumentParser) -> ArgumentParser:
    """Add ``--log-level`` to a command's parser."""
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="The lowest level of message printed. DEBUG logs where the time of each tile went",
    )
    return parser


def cli(args: Namespace | None = None) -> None:
    """Run the ``mosaic`` command."""
    setup_logging()
    args = get_parser().parse_args() if args is None else args
    setup_logging(args.log_level)
    options, run_options, convolve_options = options_from_args(args)
    mosaic(
        images=args.images,
        output_base=args.output_base,
        options=options,
        run_options=run_options,
        convolve_options=convolve_options,
        weights=args.weights,
        stokes_i_images=args.stokes_i_images,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        processes=args.processes,
    )


if __name__ == "__main__":
    cli()

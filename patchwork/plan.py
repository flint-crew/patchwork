"""Planning a mosaic: grouping its images into inputs, and building its frame and the inputs on it.

Images and cubes are given as one list. Each image's Stokes parameter is read
from its STOKES axis, and images on one pixel grid are taken to be the Stokes
parameters of one beam. An image is a cube of one channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from patchwork import __version__
from patchwork.coordinates import offset_vectors, ra_dec, unit_vectors
from patchwork.frame import (
    BoundingBox,
    Frame,
    build_frame,
    input_bounding_box,
    intersect_boxes,
    shift_box,
    strips,
    tiles,
    union_of_boxes,
    world_bounding_box,
)
from patchwork.io import NO_BEAM, BeamShape, ImageInfo, Slices, read_image_info
from patchwork.logging import logger
from patchwork.polarisation import (
    NORTH,
    STOKES_CODES,
    LinearFrame,
    frame_angle,
    header_frame,
    remove_frame_keywords,
    write_frame_keywords,
)
from patchwork.primary_beam import BeamModel, beam_model
from patchwork.resample import kernel_width
from patchwork.tile import (
    MosaicInput,
    PlaneWeight,
    group_members,
    weight_groups,
)
from patchwork.weights import combined_weight, read_weights_log

if TYPE_CHECKING:
    from astropy.io import fits

    from patchwork.options import (
        MosaicOptions,
        PolarisationOptions,
        RunOptions,
        WeightOptions,
    )


STOKES_NAMES = {code: name for name, code in STOKES_CODES.items()}


def _grid_key(info: ImageInfo) -> tuple[Any, ...]:
    """What makes two cubes share a pixel grid."""
    wcs = info.wcs.wcs
    return (
        info.shape,
        tuple(wcs.ctype),
        *(
            tuple(np.round(values, decimals))
            for values, decimals in ((wcs.crval, 9), (wcs.crpix, 6), (wcs.cdelt, 12))
        ),
    )


def _output_beams(
    infos: list[ImageInfo], channels: slice, convolved: tuple[BeamShape, ...] | None
) -> tuple[BeamShape, ...]:
    """The restoring beam of each channel of the mosaic: the beam convolved to, else that of the first cube."""
    if convolved is not None:
        return convolved[channels]
    beams = tuple(beam or NO_BEAM for beam in infos[0].channel_beams[channels])
    for info in infos[1:]:
        others = tuple(beam or NO_BEAM for beam in info.channel_beams[channels])
        if not np.allclose(beams, others, rtol=0.01, equal_nan=True):
            logger.warning(
                f"The restoring beams of {info.path} differ from those of {infos[0].path}, "
                "whose beams the mosaic is given. Convolve the cubes to a common beam to avoid this"
            )
            break
    return beams


class MosaicNames(NamedTuple):
    """The files a mosaic is written to"""

    images: tuple[Path, ...]
    """The mosaic images, one for each image of the inputs"""
    weights: tuple[Path, ...]
    """The weight of the mosaic images, one for each group of images that share a weight"""
    noises: tuple[Path, ...] = ()
    """The noise of each weight, the inverse of its square root, when it is written"""

    @property
    def image(self) -> Path:
        """The first, and usually only, mosaic image."""
        return self.images[0]

    @property
    def weight(self) -> Path:
        """The first, and usually only, weight image."""
        return self.weights[0]


class InputImages(NamedTuple):
    """The images of one input, before they are placed on the frame"""

    info: ImageInfo
    """The description of the first image, which gives the grid, frequencies and beams"""
    planes: tuple[Path, ...]
    """The images of the input, which share its grid"""
    stokes: tuple[int, ...]
    """The FITS Stokes code of each image, or 0 where it is unknown"""
    weights: tuple[PlaneWeight, ...]
    """The weight of each image"""
    stokes_i_path: Path | None = None
    """Stokes I image of the input, whose leakage is removed from its other images"""
    frame: LinearFrame | None = None
    """What the POLFRAME keywords of its images say its Q and U are referenced to, or None if they say nothing"""


class MosaicPlan(NamedTuple):
    """Everything needed to make and write the tiles of a mosaic"""

    names: MosaicNames
    """The files the mosaic is written to"""
    frame: Frame
    """The frame of the mosaic"""
    inputs: list[MosaicInput]
    """The inputs of the mosaic"""
    tiles: list[Slices]
    """The tiles the frame is split into"""
    channels: slice = slice(0, 1)
    """The channels of the inputs that are mosaicked"""
    groups: tuple[int, ...] = (0,)
    """The weight group of each mosaic image"""
    headers: tuple[fits.Header, ...] = ()
    """The header of each output: the images, then the weights, then the noises"""
    beams: tuple[BeamShape, ...] = ()
    """The restoring beam of each channel of the mosaic"""


def read_weight(path: str | Path, info: ImageInfo) -> PlaneWeight:
    """The weight of an image or cube: a linmos weights log (``.txt``), with a weight for each of its channels, or a weight image or cube on its grid.

    Raises:
        ValueError: Raised if the weights do not cover the channels or grid of the image
    """
    path = Path(path)
    if path.suffix == ".txt":
        values = read_weights_log(path)
        if len(values) != info.n_channels:
            msg = f"{path} has {len(values)} channels, but {info.path} has {info.n_channels}"
            raise ValueError(msg)
        return PlaneWeight(values=values)

    weight_info = read_image_info(path)
    if weight_info.shape != info.shape or weight_info.n_channels not in (
        1,
        info.n_channels,
    ):
        msg = f"The weight image {path} is not the shape of {info.path}"
        raise ValueError(msg)
    return PlaneWeight(values=(), paths=(path,))


def combine_weights(weights: list[PlaneWeight]) -> PlaneWeight:
    """The one weight of images that are weighted together, such as Q and U: the inverse of their mean variance.

    Raises:
        ValueError: Raised if some of the images are weighted by weight images and others are not
    """
    paths = tuple(path for weight in weights for path in weight.paths)
    if paths and not all(weight.paths for weight in weights):
        msg = "Images weighted together must all have weight images, or none"
        raise ValueError(msg)
    if paths:
        return PlaneWeight(values=(), paths=paths)
    return PlaneWeight(
        values=tuple(
            combined_weight(list(channel))
            for channel in zip(*(weight.values for weight in weights), strict=True)
        )
    )


def check_spectral_axes(infos: list[ImageInfo]) -> None:
    """Raise unless every cube has the channels of the first, to within 1% of a channel.

    Images of a single channel may be at any frequency, as each input's
    primary beam is found at its own.

    Raises:
        ValueError: Raised if the cubes differ in their channels
    """
    if all(info.n_channels == 1 for info in infos):
        return
    reference = np.asarray(infos[0].frequencies_hz)
    width = np.min(np.abs(np.diff(reference))) if len(reference) > 1 else 1.0
    for info in infos[1:]:
        frequencies = np.asarray(info.frequencies_hz)
        if frequencies.shape != reference.shape or np.any(
            np.abs(frequencies - reference) > 0.01 * width
        ):
            msg = (
                f"{info.path} has {len(frequencies)} channels that do not match the "
                f"{len(reference)} of {infos[0].path}. Every cube must share its channels"
            )
            raise ValueError(msg)


def group_inputs(
    infos: list[ImageInfo],
    weights: list[PlaneWeight],
    remove_leakage: bool = False,
    stokes_i: list[ImageInfo] | None = None,
) -> list[InputImages]:
    """Group images or cubes into the Stokes parameters of each beam, by the pixel grid they share.

    Every beam must hold the same Stokes parameters. Their images are ordered
    I, Q, U, V, and Q and U share a weight, the inverse of their mean
    variance. Images of a single Stokes parameter that share a grid, such as
    epochs of one pointing, are each an input of their own. Q without U cannot
    be rotated into the frame of the mosaic, so it is mosaicked as it is.

    Args:
        infos (list[ImageInfo]): The images or cubes
        weights (list[PlaneWeight]): The weight of each
        remove_leakage (bool, optional): Remove the leakage of each beam's Stokes I from its other images. Defaults to False.
        stokes_i (list[ImageInfo] | None, optional): Stokes I images, matched to beams by their grid, that the leakage is removed with but are not mosaicked. Defaults to None, using the beams' own.

    Raises:
        ValueError: Raised if beams hold different Stokes parameters, images of one Stokes parameter share a grid with other Stokes parameters, or the leakage is to be removed from an image of unknown Stokes parameter, or from a beam with no Stokes I image

    Returns:
        list[InputImages]: The images of each input
    """
    beams: dict[tuple[Any, ...], dict[int, list[tuple[ImageInfo, PlaneWeight]]]] = {}
    for info, weight in zip(infos, weights, strict=True):
        beam = beams.setdefault(_grid_key(info), {})
        beam.setdefault(info.stokes or 0, []).append((info, weight))

    stokes_sets = {tuple(sorted(beam)) for beam in beams.values()}
    if len(stokes_sets) > 1:
        msg = f"Every beam must hold the same Stokes parameters, but they hold {sorted(stokes_sets)}"
        raise ValueError(msg)
    codes = stokes_sets.pop()
    repeated = [
        members
        for beam in beams.values()
        for members in beam.values()
        if len(members) > 1
    ]
    if repeated and len(codes) > 1:
        msg = (
            f"{repeated[0][0][0].path} and {repeated[0][1][0].path} are both Stokes code "
            f"{repeated[0][0][0].stokes or 0} on one pixel grid, so cannot be paired with the other Stokes parameters of their beam"
        )
        raise ValueError(msg)
    if (STOKES_CODES["Q"] in codes) != (STOKES_CODES["U"] in codes):
        logger.warning(
            "Stokes Q or U images are mosaicked without the other, so cannot be "
            "rotated into the frame of the mosaic. Mosaic Q with U for that"
        )

    references: dict[tuple[Any, ...], Path] = {}
    for info in stokes_i or []:
        if references.setdefault(_grid_key(info), info.path) != info.path:
            msg = f"{info.path} and {references[_grid_key(info)]} are Stokes I images on one pixel grid"
            raise ValueError(msg)
    leaky = remove_leakage and any(code != STOKES_CODES["I"] for code in codes)
    unknown = [info.path for info in infos if not info.stokes]
    if leaky and unknown:
        msg = f"{unknown[0]} has no STOKES axis, so the leakage of Stokes I cannot be removed from it"
        raise ValueError(msg)

    images = []
    for key, beam in beams.items():
        # Each image of a single Stokes parameter sharing a grid is an input of its own
        inputs = (
            [{codes[0]: member} for member in beam[codes[0]]]
            if repeated
            else [{code: beam[code][0] for code in codes}]
        )
        for members in inputs:
            first = members[codes[0]][0]
            stokes_i_path = None
            if leaky:
                stokes_i_path = (
                    members[STOKES_CODES["I"]][0].path
                    if STOKES_CODES["I"] in members
                    else references.get(key)
                )
                if stokes_i_path is None:
                    msg = f"Removing the leakage needs a Stokes I image of every beam, but {first.path} has none"
                    raise ValueError(msg)
            plane_weights = {code: members[code][1] for code in codes}
            linear = [
                code for code in (STOKES_CODES["Q"], STOKES_CODES["U"]) if code in codes
            ]
            if linear:
                shared = combine_weights([members[code][1] for code in linear])
                for code in linear:
                    plane_weights[code] = shared
            images.append(
                InputImages(
                    info=first,
                    planes=tuple(members[code][0].path for code in codes),
                    stokes=codes,
                    weights=tuple(plane_weights[code] for code in codes),
                    stokes_i_path=stokes_i_path,
                    frame=recorded_frame([members[code][0] for code in codes], codes),
                )
            )
    return images


def rotates(stokes: tuple[int, ...]) -> bool:
    """Whether the images of an input are rotated into the frame of the mosaic, which needs both Q and U."""
    return STOKES_CODES["Q"] in stokes and STOKES_CODES["U"] in stokes


def recorded_frame(
    infos: list[ImageInfo], stokes: tuple[int, ...]
) -> LinearFrame | None:
    """The frame the POLFRAME keywords of the images of one input give, or None if they give none.

    For an input of Q and U, the frame is read from those two images, which
    must both have the keywords, or neither, and agree. The keywords of its
    other images are not used.

    Any other input is not rotated, and its frame only orients its holography
    beam. Keywords that cannot be read, or that disagree between its images,
    are then ignored with a warning rather than stopping the mosaic.

    Args:
        infos (list[ImageInfo]): The images of the input
        stokes (tuple[int, ...]): The Stokes code of each image

    Raises:
        ValueError: Raised if the keywords of Q and U cannot be read, are on only one of them, or disagree

    Returns:
        LinearFrame | None: The frame, or None if the keywords give none
    """
    if rotates(stokes):
        linear = [
            info
            for info, code in zip(infos, stokes, strict=True)
            if code in (STOKES_CODES["Q"], STOKES_CODES["U"])
        ]
        frames = [header_frame(info.header, info.path) for info in linear]
        if (frames[0] is None) != (frames[1] is None):
            tagged, untagged = linear if frames[0] is not None else linear[::-1]
            msg = (
                f"{tagged.path} has POLFRAME keywords, but {untagged.path}, the other of "
                "its Q and U, does not. Write them into both, or neither"
            )
            raise ValueError(msg)
        if (
            frames[0] is not None
            and frames[1] is not None
            and not frames[0].matches(frames[1])
        ):
            msg = (
                f"{linear[0].path} and {linear[1].path} are the Q and U of one beam, but "
                f"their POLFRAME keywords disagree: {frames[0].describe()}, and "
                f"{frames[1].describe()}"
            )
            raise ValueError(msg)
        return frames[0]

    found: list[tuple[ImageInfo, LinearFrame]] = []
    for info in infos:
        try:
            frame = header_frame(info.header, info.path)
        except ValueError as error:
            logger.warning(
                f"{error}. The keywords only orient its holography beam, so they are ignored"
            )
            continue
        if frame is not None:
            found.append((info, frame))
    if any(not frame.matches(found[0][1]) for _, frame in found[1:]):
        logger.warning(
            f"The POLFRAME keywords of the images on the grid of {infos[0].path} disagree. "
            "They only orient its holography beam, so they are ignored"
        )
        return None
    return found[0][1] if found else None


def output_names(
    output_base: str | Path,
    stokes: tuple[int, ...],
    cube: bool = False,
    output_format: str = "fits",
    noise: bool = False,
) -> MosaicNames:
    """The files of a mosaic: an image per Stokes parameter, and a weight (and noise) per group that shares a weight.

    Each is labelled by the Stokes parameters it holds, where they are known:
    ``<base>.i.mosaic.fits`` and ``<base>.qu.weight.fits`` for images, or
    ``<base>.i.cube.fits`` and ``<base>.qu.weight.cube.fits`` for cubes.

    Args:
        output_base (str | Path): Base name of the files
        stokes (tuple[int, ...]): The FITS Stokes code of each image, or 0 where it is unknown
        cube (bool, optional): Whether the mosaic is a cube. Defaults to False.
        output_format (str, optional): ``fits``, or ``zarr`` stores. Defaults to "fits".
        noise (bool, optional): Whether noise images are written. Defaults to False.

    Returns:
        MosaicNames: The files of the mosaic
    """
    suffix = "zarr" if output_format == "zarr" else "fits"
    kind = ".cube" if cube else ""

    def label(codes: list[int]) -> str:
        names = "".join(
            STOKES_NAMES[code].lower() for code in codes if code in STOKES_NAMES
        )
        return f".{names}" if names else ""

    groups = [
        [stokes[index] for index in members]
        for members in group_members(weight_groups(stokes))
    ]
    return MosaicNames(
        images=tuple(
            Path(f"{output_base}{label([code])}{kind or '.mosaic'}.{suffix}")
            for code in stokes
        ),
        weights=tuple(
            Path(f"{output_base}{label(codes)}.weight{kind}.{suffix}")
            for codes in groups
        ),
        noises=tuple(
            Path(f"{output_base}{label(codes)}.noise{kind}.{suffix}")
            for codes in groups
        )
        if noise
        else (),
    )


def plan_mosaic(
    images: list[InputImages],
    names: MosaicNames,
    options: MosaicOptions,
    run_options: RunOptions,
    beams: tuple[BeamShape, ...] | None = None,
) -> MosaicPlan:
    """Build the frame and inputs of a mosaic, and the headers of its outputs.

    Args:
        images (list[InputImages]): The images of each input
        names (MosaicNames): The files the mosaic is written to
        options (MosaicOptions): The mosaicking options
        run_options (RunOptions): Which channels are mosaicked, and how the mosaic is written
        beams (tuple[BeamShape, ...] | None, optional): The restoring beam of every channel of the inputs, such as the beams they were convolved to. Defaults to None, those of the first input.

    Returns:
        MosaicPlan: The plan of the mosaic
    """
    reference = images[options.frame.reference_index or 0].info
    channels = slice(*(run_options.channels or (0, reference.n_channels)))
    stokes = images[0].stokes
    frame, prepared = plan_inputs(images, options)
    check_holography_planes(prepared, channels)

    linear = (STOKES_CODES["Q"], STOKES_CODES["U"])
    # Q or U alone is not rotated, so its frame is whatever the inputs' was
    output_frame = None
    if rotates(stokes):
        output_frame = (
            NORTH
            if options.polarisation.output_frame == "north"
            else LinearFrame(
                (float(frame.wcs.wcs.crval[0]), float(frame.wcs.wcs.crval[1]))
            )
        )
    image_headers = []
    for code in stokes:
        header = output_header(frame, reference, stokes=code or None, channels=channels)
        if output_frame is not None and code in linear:
            write_frame_keywords(header, output_frame)
        image_headers.append(header)
    groups = weight_groups(stokes)
    firsts = [groups.index(group) for group in sorted(set(groups))]
    weight_headers = [weight_header(image_headers[first]) for first in firsts]
    noise_headers = []
    for first, header in zip(firsts[: len(names.noises)], weight_headers, strict=False):
        noise_header = header.copy()
        noise_header["BUNIT"] = image_headers[first].get("BUNIT", "")
        noise_headers.append(noise_header)

    if run_options.output_format == "zarr":
        parts = tiles(frame, tile_size=run_options.zarr_chunk)
    elif run_options.fits_writer == "workers":
        # Each task writes a run of bytes of its own, not a piece of every row
        parts = strips(frame, tile_size=options.resample.tile_size)
    else:
        parts = tiles(frame, tile_size=options.resample.tile_size)
    return MosaicPlan(
        names=names,
        frame=frame,
        inputs=prepared,
        tiles=parts,
        channels=channels,
        groups=groups,
        headers=(*image_headers, *weight_headers, *noise_headers),
        beams=_output_beams([item.info for item in images], channels, beams),
    )


def check_holography_planes(
    inputs: list[MosaicInput], channels: slice, tolerance_hz: float = 2e6
) -> None:
    """Warn of channels whose holography beam is read from a plane more than ``tolerance_hz`` from their frequency, as happens amid flagged planes.

    The plane is stretched to the channel's frequency, which is only a fair
    model of the beam close to the plane. One warning covers every input, as
    the beams of a cube are often flagged alike.
    """
    distant = []
    for item in inputs:
        model = item.primary_beam
        planes = None if model is None else model.plane_frequencies_hz()
        if model is None or planes is None:
            continue
        offsets = np.zeros(len(planes))
        offsets[channels] = np.abs(planes - np.asarray(model.frequencies_hz))[channels]
        if item.usable_channels:
            offsets[~np.array(item.usable_channels)] = 0.0
        far = offsets > tolerance_hz
        if np.any(far):
            logger.debug(
                f"{np.sum(far)} channels of {item.info.path} are up to "
                f"{np.max(offsets) / 1e6:.1f} MHz from a usable plane of holography beam {model.beam_number}"
            )
            distant.append(
                (float(np.max(offsets)), int(np.sum(far)), len(far), item.info.path)
            )
    if distant:
        offset, n_far, n_channels, path = max(distant)
        logger.warning(
            f"{len(distant)} of {len(inputs)} inputs have channels more than "
            f"{tolerance_hz / 1e6:g} MHz from the nearest usable plane of the holography cube, "
            f"so their beams are stretched from planes further away. The furthest is "
            f"{offset / 1e6:.1f} MHz away, for {n_far} of {n_channels} channels of {path}"
        )


def check_noise(options: WeightOptions, weighted_by_images: bool) -> None:
    """Warn if a noise image is asked for of inputs whose weights cannot be the inverse of their variance."""
    if options.noise_image and (
        options.weight_type == "primary_beam"
        or (not weighted_by_images and options.image_weight == "none")
    ):
        logger.warning(
            "The noise image is only relative, as the inputs are not weighted by the "
            "inverse of their variance. Weight them by their noise or weight images for one in Jy/beam"
        )


def needs_image_weight(options: WeightOptions, weight_paths: list[Path] | None) -> bool:
    """Whether the inputs are weighted by their measured noise."""
    return (
        options.weight_type != "primary_beam"
        and weight_paths is None
        and options.image_weight != "none"
    )


def input_frame_centre(
    images: InputImages, options: PolarisationOptions
) -> tuple[float, float] | None:
    """The position the Q and U of an input are referenced to north at, or None if they are referenced to north at each pixel.

    The frame comes from the POLFRAME keywords of the input's images, or from
    the options: ``frame_centre``, or ``input_frame`` of ``crval`` or
    ``north``. If both give one, they must agree. The keywords record the
    frame an image is in, such as that of a mosaic patchwork made, so rotating
    it from another frame would turn it twice.

    An input with no frame from either is refused if it holds Q and U.
    Otherwise it is not rotated, and its frame only orients its holography
    beam: options that disagree with its keywords are used with a warning,
    and with neither its frame is north at its own reference position.

    Args:
        images (InputImages): The images of the input
        options (PolarisationOptions): How polarisation is mosaicked

    Raises:
        ValueError: Raised if the keywords and options of an input of Q and U disagree, ``frame_centre`` is given with an ``input_frame`` it does not apply to, or an input of Q and U has no frame

    Returns:
        tuple[float, float] | None: The (RA, Dec) in degrees, or None for north at each pixel
    """
    info = images.info
    crval = (float(info.wcs.wcs.crval[0]), float(info.wcs.wcs.crval[1]))
    if options.frame_centre is not None and options.input_frame != "reference":
        msg = (
            "frame_centre is the position Q and U are referenced to, so it does not go "
            f"with input_frame={options.input_frame!r}"
        )
        raise ValueError(msg)
    given = {
        "north": NORTH,
        "crval": LinearFrame(crval),
        "reference": None
        if options.frame_centre is None
        else LinearFrame(options.frame_centre),
    }[options.input_frame]
    recorded = images.frame
    if recorded is not None and given is not None and not recorded.matches(given):
        if rotates(images.stokes):
            msg = (
                f"The POLFRAME keywords of {info.path} say its Q and U are referenced to "
                f"{recorded.describe()}, but the options say {given.describe()}. The keywords "
                "record the frame the image is in, so rotating it from another would turn it "
                "twice. Leave out the frame options, or correct the keywords"
            )
            raise ValueError(msg)
        logger.warning(
            f"The POLFRAME keywords of {info.path} say {recorded.describe()}, but the "
            f"options say {given.describe()}. It is not rotated, so the options are used "
            "to orient its holography beam"
        )
        recorded = None
    frame = recorded if recorded is not None else given
    if frame is not None:
        return frame.centre
    if rotates(images.stokes):
        msg = (
            f"{info.path} has Q and U, but nothing says what their angles are measured from. "
            "For the beams of an ASKAP observation, give the boresight with --frame-centre RA DEC "
            "(frame_centre); FixMS keeps it as the PHASE_DIR of the FIELD_OLD table of each "
            "measurement set. If Q and U are already referenced to north at each pixel, give "
            "--input-frame north. If each image is referenced to north at its own reference "
            "position, as for a single pointing, give --input-frame crval"
        )
        raise ValueError(msg)
    return crval


def contributing_box(
    frame: Frame,
    info: ImageInfo,
    primary_beam: BeamModel | None,
    cutoff: float,
    margin: int,
) -> BoundingBox | None:
    """The region of the frame an input contributes to: its footprint, within where its beam reaches ``cutoff`` in any channel."""
    box = input_bounding_box(frame, info, margin=margin)
    reach = None if primary_beam is None else primary_beam.reach(cutoff)
    if reach is None:
        return box
    world = np.array(ra_dec(offset_vectors(info.beam_centre, *reach)))
    return intersect_boxes(box, world_bounding_box(frame, world, margin=margin))


def prepare_inputs(
    images: list[InputImages], frame: Frame, options: MosaicOptions
) -> list[MosaicInput]:
    """Find the primary beam and footprint of each input on the frame.

    Args:
        images (list[InputImages]): The images of each input
        frame (Frame): The frame of the mosaic
        options (MosaicOptions): The mosaicking options

    Returns:
        list[MosaicInput]: The prepared inputs
    """
    weight_options, polarisation = options.weight, options.polarisation
    margin = kernel_width(options.resample.kernel) // 2 + 1
    weighted_by_beam = weight_options.weight_type != "weight_image"
    inputs = []
    for item in images:
        info = item.info
        for indices in group_members(weight_groups(item.stokes)):
            if any(
                item.weights[index] != item.weights[indices[0]] for index in indices
            ):
                msg = f"The images of {info.path} that share a weight (Q and U) are given different weights"
                raise ValueError(msg)
        frame_centre = input_frame_centre(item, polarisation)
        frame_rotation = (
            0.0
            if frame_centre is None
            else float(frame_angle(frame_centre, unit_vectors(*info.beam_centre)))
        )
        leakage = item.stokes_i_path is not None
        primary_beam = (
            beam_model(
                options.primary_beam,
                info,
                leakage=leakage,
                normalise_leakage=not polarisation.keep_centre_leakage,
                frame_rotation_rad=frame_rotation,
            )
            if weighted_by_beam or leakage
            else None
        )
        inputs.append(
            MosaicInput(
                info=info,
                planes=item.planes,
                stokes=item.stokes,
                weights=item.weights,
                primary_beam=primary_beam,
                bounding_box=contributing_box(
                    frame,
                    info,
                    primary_beam if weighted_by_beam else None,
                    cutoff=options.primary_beam.pb_cutoff,
                    margin=margin,
                ),
                stokes_i_path=item.stokes_i_path,
                frame_centre=frame_centre,
                usable_channels=tuple(
                    beam is None or beam.is_usable() for beam in info.channel_beams
                ),
            )
        )
        logger.info(
            f"Prepared {info.path} with weights {[weight.values[:3] for weight in item.weights]}"
        )
    return inputs


def trim_to_inputs(
    frame: Frame, inputs: list[MosaicInput]
) -> tuple[Frame, list[MosaicInput]]:
    """Trim a frame to where its inputs contribute, moving their footprints onto the trimmed frame."""
    union = union_of_boxes([item.bounding_box for item in inputs])
    if union is None:
        return frame, inputs
    trimmed = frame.crop(union)
    if trimmed.shape != frame.shape:
        logger.info(
            f"Trimmed the frame from {frame.shape[1]}x{frame.shape[0]} to {trimmed.shape[1]}x{trimmed.shape[0]} pixels"
        )
    return trimmed, [
        item._replace(bounding_box=shift_box(item.bounding_box, union))
        for item in inputs
    ]


def output_header(
    frame: Frame,
    reference: ImageInfo,
    beam: BeamShape | None = None,
    stokes: int | None = None,
    channels: slice | None = None,
) -> fits.Header:
    """The header of a mosaic, with keywords and the restoring beam of a reference input, and its Stokes parameter.

    Args:
        frame (Frame): The frame of the mosaic
        reference (ImageInfo): The input whose keywords and axes are kept
        beam (BeamShape | None, optional): Restoring beam of the mosaic. Defaults to None, the beam of the reference input.
        stokes (int | None, optional): FITS Stokes code of the mosaic. Defaults to None, that of the reference input.
        channels (slice | None, optional): The channels of the reference input that the mosaic holds. Defaults to None, a single plane.

    Returns:
        fits.Header: The header
    """
    header = frame.header(reference.header, channels=channels)
    for key in ("TELESCOP", "PROJECT", "SBID", "DATE-OBS", "DURATION", "BUNIT"):
        if key in reference.header:
            header[key] = reference.header[key]

    if stokes is not None:
        for axis in range(3, header["NAXIS"] + 1):
            if str(header.get(f"CTYPE{axis}", "")).upper() == "STOKES":
                header[f"CRVAL{axis}"] = stokes
                header[f"CRPIX{axis}"] = 1

    beam = beam or reference.beam
    if beam is not None and beam.is_usable():
        beam.to_header(header)
    header["HISTORY"] = f"Mosaicked by patchwork {__version__}"
    return header


def weight_header(image_header: fits.Header) -> fits.Header:
    """The header of a weight image, from that of the first mosaic image of its group."""
    header = image_header.copy()
    header.remove("BUNIT", ignore_missing=True)
    remove_frame_keywords(header)
    return header


def plan_inputs(
    images: list[InputImages], options: MosaicOptions
) -> tuple[Frame, list[MosaicInput]]:
    """Build the frame of a mosaic and prepare its inputs on it, trimming it to where they contribute."""
    frame = build_frame([item.info for item in images], options.frame)
    inputs = prepare_inputs(images, frame, options)
    if not options.frame.full_frame and options.frame.image_size is None:
        frame, inputs = trim_to_inputs(frame, inputs)
    logger.info(f"Mosaic frame is {frame.shape[1]}x{frame.shape[0]} pixels")
    return frame, inputs

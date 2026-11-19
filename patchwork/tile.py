"""Making one tile of a mosaic: resampling, weighting and combining every input that covers it.

An input holds one or more images on one grid (I, Q, U or V, each a single
plane or a cube of channels), which are made together: Stokes Q and U so that
each input can be rotated into the frame the mosaic references its polarisation
to, and Stokes I so that its leakage into the others can be removed. Tiles are
independent, so they can be made in any order or in parallel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from patchwork.combine import TileSums, accumulate, empty_sums, finalise
from patchwork.coordinates import (
    TileCoordinates,
    map_to_input,
    offsets,
    tile_coordinates,
)
from patchwork.frame import tile_region
from patchwork.io import read_block
from patchwork.polarisation import (
    STOKES_CODES,
    frame_angle,
    remove_leakage,
    rotate_linear,
)
from patchwork.resample import (
    Backend,
    Kernel,
    input_region,
    interpolate_stack,
    resolve_backend,
)
from patchwork.weights import combined_weights, weight_from_map, weighting

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from patchwork.frame import BoundingBox, Frame
    from patchwork.io import ImageInfo, Slices
    from patchwork.options import MosaicOptions, PolarisationOptions, WeightOptions
    from patchwork.primary_beam import BeamModel


class PlaneWeight(NamedTuple):
    """The weight of one image of an input: a value for each channel, or weight images"""

    values: tuple[float, ...] = (1.0,)
    """The weight of each channel, or one weight for every channel, used when there are no weight images"""
    paths: tuple[Path, ...] = ()
    """Weight images or cubes, combined as ``combined_weights`` does when there are several"""


class MosaicInput(NamedTuple):
    """An input, ready to be added to the tiles it covers"""

    info: ImageInfo
    """The description of the first image of the input"""
    planes: tuple[Path, ...]
    """The images of the input, which share its grid"""
    stokes: tuple[int, ...]
    """The FITS Stokes code of each image, or 0 where it is unknown"""
    weights: tuple[PlaneWeight, ...]
    """The weight of each image"""
    primary_beam: BeamModel | None
    """Primary beam of the input, if it is weighted by one or has its leakage removed"""
    bounding_box: BoundingBox | None
    """The region of the frame the input contributes to"""
    stokes_i_path: Path | None = None
    """Stokes I image of the input, whose leakage is removed from its other images"""
    frame_centre: tuple[float, float] | None = None
    """The (RA, Dec) its Q and U are referenced to north at, or None if they are referenced to north at every pixel"""
    usable_channels: tuple[bool, ...] = ()
    """Which channels hold an image, by their restoring beam, where empty means every channel"""


class TileResult(NamedTuple):
    """One finished tile of a mosaic, over some channels"""

    slices: Slices
    """The (y, x) region of the frame the tile covers"""
    channels: slice
    """The channels of the inputs the tile covers"""
    images: tuple[NDArray[np.float32], ...]
    """The (channel, y, x) mosaic of each image of the inputs"""
    weights: tuple[NDArray[np.float32], ...]
    """The (channel, y, x) weight of each group of images that share a weight"""


def weight_groups(stokes: tuple[int, ...]) -> tuple[int, ...]:
    """The group of each image that shares its weight: Q and U share one, and every other image has its own."""
    keys = [
        "qu" if code in (STOKES_CODES["Q"], STOKES_CODES["U"]) else str(code)
        for code in stokes
    ]
    order = list(dict.fromkeys(keys))
    return tuple(order.index(key) for key in keys)


def group_members(groups: tuple[int, ...]) -> list[list[int]]:
    """The indices of the images in each weight group, in the order of the groups."""
    return [
        [index for index, member in enumerate(groups) if member == group]
        for group in sorted(set(groups))
    ]


def _rotation_into_mosaic(
    item: MosaicInput,
    options: PolarisationOptions,
    frame: Frame,
    vectors: NDArray[np.float64],
) -> NDArray[np.float64] | float:
    """The angle that turns polarisation from the frame of an input into the frame of the mosaic."""
    rotation: NDArray[np.float64] | float = (
        0.0 if item.frame_centre is None else frame_angle(item.frame_centre, vectors)
    )
    if options.output_frame == "centre":
        centre = (float(frame.wcs.wcs.crval[0]), float(frame.wcs.wcs.crval[1]))
        rotation = rotation - frame_angle(centre, vectors)
    return rotation


class _Sampler(NamedTuple):
    """Interpolates the channels of images on one input grid at the positions of a region of a tile"""

    window: Slices
    channels: slice
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    kernel: Kernel
    backend: Backend

    def image(self, path: Path) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
        """The interpolated channels of an image, and which are valid."""
        # float32 halves the memory of the block, and interpolation sums in float64 either way
        block = read_block(path, self.window, self.channels, dtype=np.float32)
        return interpolate_stack(block, self.x, self.y, self.kernel, self.backend)

    def weight(
        self, path: Path, map_type: Literal["weight", "variance", "rms"]
    ) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
        """The interpolated channels of a weight image, as weights, and which are valid."""
        block = weight_from_map(read_block(path, self.window, self.channels), map_type)
        return interpolate_stack(block, self.x, self.y, self.kernel, self.backend)


def mosaic_tile(
    tile: Slices,
    frame: Frame,
    inputs: list[MosaicInput],
    options: MosaicOptions,
    channels: slice | None = None,
) -> TileResult:
    """Resample, weight and combine every input that covers one tile of the frame, over some channels.

    Each input is only worked on over the part of the tile it contributes to.
    Positions in it are found once and used for each of its images and channels.

    Args:
        tile (Slices): The (y, x) region of the frame to make
        frame (Frame): The frame of the mosaic
        inputs (list[MosaicInput]): The inputs of the mosaic
        options (MosaicOptions): The mosaicking options
        channels (slice | None, optional): The channels of the inputs to make. Defaults to None, the first channel.

    Returns:
        TileResult: The finished tile
    """
    if resolve_backend(options.resample.backend) == "fused":
        from patchwork.fused import mosaic_tile_fused  # noqa: PLC0415

        return mosaic_tile_fused(tile, frame, inputs, options, channels)
    combine_type = options.combine.combine_type
    channels = channels or slice(0, 1)
    channel_range = range(channels.start or 0, channels.stop)

    coordinates = tile_coordinates(
        frame.wcs, tile, step=options.resample.coordinate_step
    )
    shape = (len(channel_range), *coordinates.vectors.shape[1:])
    stokes = inputs[0].stokes if inputs else (0,)
    members = group_members(weight_groups(stokes))
    sums = [empty_sums(shape, len(indices), combine_type) for indices in members]
    for item in inputs:
        _add_input(item, tile, frame, coordinates, sums, members, options, channels)

    images: list[NDArray[np.float32]] = [np.empty(0, dtype=np.float32)] * len(stokes)
    group_weights = []
    for group_sums, indices in zip(sums, members, strict=True):
        group_images, group_weight = finalise(group_sums, combine_type)
        for index, image in zip(indices, group_images, strict=True):
            images[index] = image
        group_weights.append(group_weight)
    return TileResult(
        slices=tile,
        channels=slice(channel_range.start, channel_range.stop),
        images=tuple(images),
        weights=tuple(group_weights),
    )


def _add_input(
    item: MosaicInput,
    tile: Slices,
    frame: Frame,
    coordinates: TileCoordinates,
    sums: list[TileSums],
    members: list[list[int]],
    options: MosaicOptions,
    channels: slice,
) -> None:
    """Resample, weight and add one input to the sums of a tile, over the part of the tile it contributes to.

    Its arrays are let go on return, so a tile only ever holds those of one input.
    """
    weight_options, kernel = options.weight, options.resample.kernel
    backend = resolve_backend(options.resample.backend)
    weighted_by_beam = weight_options.weight_type != "weight_image"
    combine_type = options.combine.combine_type
    channel_range = range(channels.start or 0, channels.stop)

    region = tile_region(tile, item.bounding_box)
    if region is None:
        return
    usable = np.array(
        [
            item.usable_channels[index] if item.usable_channels else True
            for index in channel_range
        ]
    )
    if not np.any(usable):
        return
    x, y = map_to_input(coordinates, item.info.wcs, region)
    window = input_region(item.info.shape, x, y, kernel)
    if window is None:
        return
    x, y = x - window[1].start, y - window[0].start

    sampler = _Sampler(window, channels, x, y, kernel, backend)
    interpolated = [sampler.image(path) for path in item.planes]
    planes = [values for values, _ in interpolated]
    valid = [is_valid & usable[:, None, None] for _, is_valid in interpolated]

    vectors = coordinates.vectors[:, region[0], region[1]]
    east, north = offsets(item.info.beam_centre, vectors)
    primary_beam = None
    cutoff: NDArray[np.float64] | float = options.primary_beam.pb_cutoff
    if item.primary_beam is not None and weighted_by_beam:
        primary_beam = item.primary_beam.responses(channel_range, east, north)
        cutoff = cutoff * item.primary_beam.cutoff_scales(channel_range).reshape(
            -1, 1, 1
        )

    if item.stokes_i_path is not None and item.primary_beam is not None:
        if item.stokes_i_path in item.planes:
            position = item.planes.index(item.stokes_i_path)
            stokes_i, stokes_i_valid = planes[position], valid[position]
        else:
            stokes_i, stokes_i_valid = sampler.image(item.stokes_i_path)
        leaky = [
            index for index, code in enumerate(item.stokes) if code != STOKES_CODES["I"]
        ]
        corrected = remove_leakage(
            tuple(planes[index] for index in leaky),
            tuple(item.stokes[index] for index in leaky),
            stokes_i,
            np.moveaxis(
                item.primary_beam.leakage_responses(
                    channel_range,
                    east,
                    north,
                    include_v=STOKES_CODES["V"] in item.stokes,
                ),
                1,
                0,
            ),
        )
        for index, plane in zip(leaky, corrected, strict=True):
            planes[index] = plane
            valid[index] = valid[index] & stokes_i_valid

    if STOKES_CODES["Q"] in item.stokes and STOKES_CODES["U"] in item.stokes:
        q_index, u_index = (
            item.stokes.index(STOKES_CODES["Q"]),
            item.stokes.index(STOKES_CODES["U"]),
        )
        planes[q_index], planes[u_index] = rotate_linear(
            planes[q_index],
            planes[u_index],
            _rotation_into_mosaic(item, options.polarisation, frame, vectors),
        )
        valid[q_index] = valid[u_index] = valid[q_index] & valid[u_index]

    # The images of a group share a weight, so it is found once for them all
    for group_sums, indices in zip(sums, members, strict=True):
        weight, contributes = _group_weight(
            item.weights[indices[0]], sampler, channels, weight_options
        )
        for index in indices:
            contributes = contributes & valid[index]
        if primary_beam is not None:
            contributes &= primary_beam >= cutoff
        factor, weight = weighting(
            primary_beam=primary_beam,
            weight=weight,
            weight_type=weight_options.weight_type,
            weight_state=weight_options.weight_state,
        )
        accumulate(
            group_sums.region((slice(None), *region)),
            [planes[index] for index in indices],
            factor,
            np.where(contributes, weight, 0.0),
            combine_type,
        )


def _group_weight(
    plane_weight: PlaneWeight,
    sampler: _Sampler,
    channels: slice,
    options: WeightOptions,
) -> tuple[NDArray[np.float64], NDArray[np.bool] | bool]:
    """The weight of a group of an input's images over a region of a tile, and where it is valid."""
    if not plane_weight.paths:
        values = np.asarray(plane_weight.values, dtype=np.float64)
        values = values if len(values) == 1 else values[channels]
        return values.reshape(-1, 1, 1), True
    maps = [
        sampler.weight(path, options.weight_map_type) for path in plane_weight.paths
    ]
    weight = (
        maps[0][0]
        if len(maps) == 1
        else combined_weights([values for values, _ in maps])
    )
    return weight, np.logical_and.reduce([is_valid for _, is_valid in maps])

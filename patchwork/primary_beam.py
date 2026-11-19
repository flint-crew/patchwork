"""Primary beam models that weight each input"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from astropy.constants import c as speed_of_light
from astropy.io import fits
from scipy.ndimage import map_coordinates

from patchwork.coordinates import offsets, unit_vectors
from patchwork.logging import logger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from patchwork.io import ImageInfo
    from patchwork.options import PrimaryBeamOptions


def direction_cosines(
    ra0_deg: float, dec0_deg: float, ra_deg: ArrayLike, dec_deg: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The offsets (l to the east, m to the north) of sky positions from a centre, as direction cosines."""
    return offsets((ra0_deg, dec0_deg), unit_vectors(ra_deg, dec_deg))


class GaussianBeam(NamedTuple):
    """A circular Gaussian primary beam"""

    fwhm_rad: float
    """Full width at half maximum in radians"""

    def response(
        self, east: NDArray[np.float64], north: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The response at offsets from the beam centre, given as direction cosines."""
        offset = np.arcsin(np.clip(np.hypot(east, north), 0.0, 1.0))
        return np.asarray(np.exp(-4 * np.log(2) * offset**2 / self.fwhm_rad**2))

    def support_radius(self, cutoff: float) -> float | None:
        """The offset in radians beyond which the response is below ``cutoff``, or None if it never is."""
        if cutoff <= 0:
            return None
        return float(
            self.fwhm_rad * np.sqrt(np.log(1 / min(cutoff, 1.0)) / (4 * np.log(2)))
        )


class PatternTaps(NamedTuple):
    """Where positions fall on the grid of a holography cube: the four pixels around each, and their weights in linear interpolation"""

    indices: NDArray[np.intp]
    """Flat indices into the grid of the four pixels around each position, shaped (4, n)"""
    weights: NDArray[np.float64]
    """The weight of each of the four pixels, shaped (4, n), which is 0 for positions off the grid"""
    shape: tuple[int, ...]
    """The shape of the positions"""

    def sample(self, plane: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate a plane of the grid at the positions."""
        flat = plane.ravel()
        values = self.weights[0] * flat[self.indices[0]]
        for tap in range(1, 4):
            values += self.weights[tap] * flat[self.indices[tap]]
        return np.asarray(values.reshape(self.shape), dtype=np.float64)


def bilinear_taps(
    pixels: NDArray[np.float64], grid_shape: tuple[int, ...], shape: tuple[int, ...]
) -> PatternTaps:
    """The taps that interpolate a grid linearly at (y, x) pixels, as ``scipy.ndimage.map_coordinates`` does with ``order=1`` and ``mode="constant"``.

    Args:
        pixels (NDArray[np.float64]): Zero-based (y, x) positions, shaped (2, n)
        grid_shape (tuple[int, ...]): The (ny, nx) shape of the grid
        shape (tuple[int, ...]): The shape the n positions take

    Returns:
        PatternTaps: The taps, which give 0 for positions off the grid
    """
    ny, nx = grid_shape
    y, x = pixels
    on_grid = (y >= 0) & (y <= ny - 1) & (x >= 0) & (x <= nx - 1)
    y, x = np.where(on_grid, y, 0.0), np.where(on_grid, x, 0.0)
    # The upper corner is the last pixel's own, so a position on the last row or column uses it
    y_low = np.minimum(np.floor(y), max(ny - 2, 0)).astype(np.intp)
    x_low = np.minimum(np.floor(x), max(nx - 2, 0)).astype(np.intp)
    y_fraction, x_fraction = y - y_low, x - x_low
    low = y_low * nx + x_low
    y_step, x_step = min(1, ny - 1) * nx, min(1, nx - 1)
    return PatternTaps(
        indices=np.array([low, low + x_step, low + y_step, low + y_step + x_step]),
        weights=np.array(
            [
                (1 - y_fraction) * (1 - x_fraction),
                (1 - y_fraction) * x_fraction,
                y_fraction * (1 - x_fraction),
                y_fraction * x_fraction,
            ]
        )
        * on_grid,
        shape=shape,
    )


class HolographyBeam(NamedTuple):
    """One beam of a holography cube at one frequency"""

    plane: NDArray[np.float64]
    """The (y, x) response, normalised to 1 at the beam centre unless used as the cube stores it"""
    pixel_scale_rad: tuple[float, float]
    """Signed pixel size along x and y in radians (CDELT1, CDELT2)"""
    centre_pixel: tuple[float, float]
    """Zero-based (x, y) pixel of the beam centre"""
    alpha_rad: float
    """Rotation of the feeds from the frame the cube was measured in"""
    scale: float
    """Frequency of the image over the frequency of the plane, by which offsets are scaled as the beam widens with wavelength"""
    leakage: NDArray[np.float64] | None = None
    """The fractional leakage of Stokes I into (Q, U, V), shaped (3, y, x), if it was read"""
    frame_rotation_rad: float = 0.0
    """Position angle, east of north at the beam centre, of the direction the input's polarisation is referenced to"""
    cutoff_scale: float = 1.0
    """The response a cutoff is a fraction of: 1, the beam centre, or the peak of the plane when the beam is used as the cube stores it"""

    def _pattern_pixels(
        self, east: NDArray[np.float64], north: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The (y, x) pixels of the cube at offsets from the beam centre."""
        rotation = self.alpha_rad + self.frame_rotation_rad
        cos_rotation, sin_rotation = np.cos(rotation), np.sin(rotation)
        x = east * cos_rotation - north * sin_rotation
        y = east * sin_rotation + north * cos_rotation

        radius = np.hypot(x, y)
        offset = np.arcsin(np.clip(radius, 0.0, 1.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            stretch = np.where(
                radius > 0, np.sin(self.scale * offset) / radius, self.scale
            )

        x_pixel = x * stretch / self.pixel_scale_rad[0] + self.centre_pixel[0]
        y_pixel = y * stretch / self.pixel_scale_rad[1] + self.centre_pixel[1]
        return np.array([np.ravel(y_pixel), np.ravel(x_pixel)])

    def taps(
        self, east: NDArray[np.float64], north: NDArray[np.float64]
    ) -> PatternTaps:
        """Where offsets from the beam centre, given as direction cosines, fall on the grid of the cube."""
        return bilinear_taps(
            self._pattern_pixels(east, north), self.plane.shape, np.shape(east)
        )

    def response(
        self,
        east: NDArray[np.float64],
        north: NDArray[np.float64],
        taps: PatternTaps | None = None,
    ) -> NDArray[np.float64]:
        """The response at offsets from the beam centre, given as direction cosines, or at ``taps`` found for them. It is 0 off the cube."""
        return (self.taps(east, north) if taps is None else taps).sample(self.plane)

    def leakage_response(
        self,
        east: NDArray[np.float64],
        north: NDArray[np.float64],
        taps: PatternTaps | None = None,
        include_v: bool = True,
    ) -> NDArray[np.float64]:
        """The fractional (Q, U, V) leakage at offsets from the beam centre, with Q and U in the frame of the input.

        The leakage is rotated with the feeds, by ``alpha_rad``.
        Without ``include_v`` only Q and U are given.

        Raises:
            ValueError: Raised if the leakage was not read from the cube
        """
        if self.leakage is None:
            msg = "The leakage of this holography beam was not read"
            raise ValueError(msg)
        taps = self.taps(east, north) if taps is None else taps
        q, u = taps.sample(self.leakage[0]), taps.sample(self.leakage[1])
        cos_2alpha, sin_2alpha = np.cos(2 * self.alpha_rad), np.sin(2 * self.alpha_rad)
        leakage = [q * cos_2alpha - u * sin_2alpha, q * sin_2alpha + u * cos_2alpha]
        if include_v:
            leakage.append(taps.sample(self.leakage[2]))
        return np.array(leakage)

    def reaches(self, cutoff: float) -> NDArray[np.bool_]:
        """The pixels of the plane at or above ``cutoff`` of ``cutoff_scale``, or above 0 when ``cutoff`` is not positive."""
        if cutoff > 0:
            return np.asarray(self.plane >= cutoff * self.cutoff_scale)
        return np.asarray(self.plane > 0)

    def pixel_offsets(
        self, y: NDArray[np.float64], x: NDArray[np.float64], scale: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """The offsets (east, north) from the beam centre, as direction cosines, that ``taps`` places on (y, x) pixels at a frequency scale, or None if some lie a quarter of the sky or more away."""
        grid_x = (x - self.centre_pixel[0]) * self.pixel_scale_rad[0]
        grid_y = (y - self.centre_pixel[1]) * self.pixel_scale_rad[1]
        radius = np.hypot(grid_x, grid_y)
        offset = np.arcsin(np.clip(radius, 0.0, 1.0)) / scale
        if np.any(offset >= np.pi / 2):
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            shrink = np.where(radius > 0, np.sin(offset) / radius, 0.0)
        rotation = self.alpha_rad + self.frame_rotation_rad
        cos_rotation, sin_rotation = np.cos(rotation), np.sin(rotation)
        x, y = grid_x * shrink, grid_y * shrink
        return x * cos_rotation + y * sin_rotation, y * cos_rotation - x * sin_rotation


PrimaryBeam = GaussianBeam | HolographyBeam


def gaussian_beam(
    frequency_hz: float, aperture_m: float = 12.0, fwhm_scaling: float = 1.09
) -> GaussianBeam:
    """The Gaussian beam of a dish, with a FWHM of ``fwhm_scaling`` wavelengths over the dish diameter."""
    wavelength_m = speed_of_light.value / frequency_hz
    return GaussianBeam(fwhm_rad=fwhm_scaling * wavelength_m / aperture_m)


@functools.lru_cache(maxsize=128)
def usable_planes(
    holography_cube: Path, beam_number: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """The frequency of every plane of a beam of a holography cube, which planes respond at the beam centre, and that response.

    Only the pixels around the centre of every plane are read. The result is
    kept, as every channel of every input that uses the beam needs it.

    Raises:
        ValueError: Raised if the beam centre is near the cube edge, or the beam has no usable planes
    """
    with fits.open(holography_cube, memmap=True) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        beam = data[beam_number] if data.ndim == 5 else data[beam_number][None]
        n_freq, ny, nx = beam.shape[1:]

        x0, y0 = header["CRPIX1"] - 1, header["CRPIX2"] - 1
        if min(x0, nx - 1 - x0, y0, ny - 1 - y0) < 2:
            msg = (
                f"The beam centre (CRPIX) of {holography_cube} is at pixel ({x0 + 1}, {y0 + 1}) "
                f"of a {nx}x{ny} grid, too close to its edge"
            )
            raise ValueError(msg)

        x_low, y_low = int(np.floor(x0)), int(np.floor(y0))
        around = np.nan_to_num(
            np.asarray(
                beam[0, :, y_low : y_low + 2, x_low : x_low + 2], dtype=np.float64
            )
        )
        freqs_hz = (
            header["CRVAL3"]
            + (np.arange(n_freq) + 1 - header["CRPIX3"]) * header["CDELT3"]
        )

    centre = map_coordinates(
        around,
        np.array(
            [
                np.arange(n_freq),
                np.full(n_freq, y0 - y_low),
                np.full(n_freq, x0 - x_low),
            ]
        ),
        order=1,
    )
    usable = np.flatnonzero(centre > 0)
    if usable.size == 0:
        msg = f"Beam {beam_number} of {holography_cube} has no usable response"
        raise ValueError(msg)
    return np.asarray(freqs_hz, dtype=np.float64), usable, centre


def nearest_planes(
    holography_cube: Path, beam_number: int, frequencies_hz: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """The usable plane of a beam of a holography cube nearest each frequency, and the frequency of that plane."""
    freqs_hz, usable, _ = usable_planes(Path(holography_cube), beam_number)
    wanted = np.reshape(np.asarray(frequencies_hz, dtype=np.float64), (-1, 1))
    nearest = usable[np.argmin(np.abs(freqs_hz[usable] - wanted), axis=1)]
    return nearest, freqs_hz[nearest]


def load_holography_beam(
    holography_cube: Path,
    beam_number: int,
    frequency_hz: float,
    alpha_rad: float = 0.0,
    leakage: bool = False,
    normalise_leakage: bool = True,
    frame_rotation_rad: float = 0.0,
    normalise: bool = True,
) -> HolographyBeam:
    """Read one beam of a holography cube at the plane nearest a frequency.

    The spatial axes are offsets about the beam centre, so only CRPIX and CDELT
    are used. NaNs count as no response, planes with no response
    at the beam centre (such as flagged channels) are skipped, and the beam is
    scaled by the ratio of the image and plane frequencies, as the beam widens
    with wavelength.

    Args:
        holography_cube (Path): Cube with axes (beam, stokes, freq, dec, ra), or (beam, freq, dec, ra)
        beam_number (int): Zero-based beam to read
        frequency_hz (float): Frequency of the image the beam weights
        alpha_rad (float, optional): Rotation of the feeds from the frame the cube was measured in. Defaults to 0.0.
        leakage (bool, optional): Also read the fractional leakage of Stokes I into Q, U and V. Defaults to False.
        normalise_leakage (bool, optional): Remove the leakage at the beam centre, as calibration does. Defaults to True.
        frame_rotation_rad (float, optional): Position angle at the beam centre of the direction the input's polarisation is referenced to. Defaults to 0.0.
        normalise (bool, optional): Scale the beam to 1 at its centre, where calibration sets the gain. Otherwise it is used as the cube stores it, and a cutoff is measured from its peak. Defaults to True.

    Raises:
        ValueError: Raised if the beam centre is near the cube edge, the beam has no usable planes, or leakage is asked of a cube without Stokes planes

    Returns:
        HolographyBeam: The beam
    """
    freqs_hz, _, centre = usable_planes(Path(holography_cube), beam_number)
    nearest = int(nearest_planes(holography_cube, beam_number, frequency_hz)[0][0])
    with fits.open(holography_cube, memmap=True) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        beam = data[beam_number] if data.ndim == 5 else data[beam_number][None]
        if leakage and beam.shape[0] < 4:
            msg = f"{holography_cube} has no Stokes planes to read the leakage from"
            raise ValueError(msg)
        x0, y0 = header["CRPIX1"] - 1, header["CRPIX2"] - 1
        plane = np.nan_to_num(np.asarray(beam[0, nearest], dtype=np.float64))
        leakage_planes = None
        if leakage:
            leakage_planes = np.nan_to_num(
                np.asarray(beam[1:4, nearest], dtype=np.float64)
            )
            if normalise_leakage:
                at_centre = map_coordinates(
                    leakage_planes,
                    np.array([np.arange(3), np.full(3, y0), np.full(3, x0)]),
                    order=1,
                )
                leakage_planes -= at_centre[:, None, None]

    logger.debug(
        f"Using plane {nearest} at {freqs_hz[nearest] / 1e6:.2f} MHz of beam {beam_number}"
    )
    return HolographyBeam(
        plane=plane / centre[nearest] if normalise else plane,
        pixel_scale_rad=(
            float(np.radians(header["CDELT1"])),
            float(np.radians(header["CDELT2"])),
        ),
        centre_pixel=(float(x0), float(y0)),
        alpha_rad=alpha_rad,
        scale=float(frequency_hz / freqs_hz[nearest]),
        leakage=leakage_planes,
        frame_rotation_rad=frame_rotation_rad,
        cutoff_scale=1.0 if normalise else float(np.max(plane)),
    )


@functools.lru_cache(maxsize=512)
def _cached_holography_beam(
    holography_cube: Path,
    beam_number: int,
    frequency_hz: float,
    alpha_rad: float,
    leakage: bool,
    normalise_leakage: bool,
    frame_rotation_rad: float,
    normalise: bool,
) -> HolographyBeam:
    """A holography beam, kept so that each worker reads a plane once."""
    return load_holography_beam(
        holography_cube,
        beam_number,
        frequency_hz,
        alpha_rad=alpha_rad,
        leakage=leakage,
        normalise_leakage=normalise_leakage,
        frame_rotation_rad=frame_rotation_rad,
        normalise=normalise,
    )


class BeamModel(NamedTuple):
    """The primary beam of an input at the frequency of each of its channels.

    It is small enough to send to every worker, which build the beam of a
    channel when it is needed, reading holography planes as they go.
    """

    options: PrimaryBeamOptions
    """The beam model"""
    frequencies_hz: tuple[float, ...]
    """The frequency of each channel"""
    beam_number: int | None = None
    """The beam of the holography cube"""
    leakage: bool = False
    """Whether the polarisation leakage is read"""
    normalise_leakage: bool = True
    """Whether the leakage at the beam centre is removed"""
    frame_rotation_rad: float = 0.0
    """Position angle at the beam centre of the direction the input's polarisation is referenced to"""

    def channel(self, index: int) -> PrimaryBeam:
        """The primary beam of one channel."""
        frequency = self.frequencies_hz[index]
        if self.options.pb_model == "gaussian":
            return gaussian_beam(
                frequency, self.options.aperture_m, self.options.fwhm_scaling
            )
        return _cached_holography_beam(
            Path(str(self.options.holography_cube)),
            int(self.beam_number or 0),
            frequency,
            self.options.holography_alpha_rad,
            self.leakage,
            self.normalise_leakage,
            self.frame_rotation_rad,
            self.options.holography_scale == "centre",
        )

    def responses(
        self, channels: range, east: NDArray[np.float64], north: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The response of each channel at offsets from the beam centre, shaped ``(channel, *east.shape)``."""
        if self.options.pb_model == "gaussian":
            offset = np.arcsin(np.clip(np.hypot(east, north), 0.0, 1.0)) ** 2
            fwhms = np.array([self.channel(index).fwhm_rad for index in channels])  # type: ignore[union-attr]
            return np.asarray(
                np.exp(
                    -4 * np.log(2) * offset / fwhms.reshape(-1, *[1] * offset.ndim) ** 2
                )
            )
        beams = self._holography_beams(channels)
        return np.array(
            [
                beam.response(east, north, taps)
                for beam, taps in zip(
                    beams, _shared_taps(beams, east, north), strict=True
                )
            ]
        )

    def leakage_responses(
        self,
        channels: range,
        east: NDArray[np.float64],
        north: NDArray[np.float64],
        include_v: bool = True,
    ) -> NDArray[np.float64]:
        """The fractional (Q, U, V) leakage of each channel, shaped ``(channel, 3, *east.shape)``, or without V ``(channel, 2, *east.shape)``."""
        beams = self._holography_beams(channels)
        return np.array(
            [
                beam.leakage_response(east, north, taps, include_v=include_v)
                for beam, taps in zip(
                    beams, _shared_taps(beams, east, north), strict=True
                )
            ]
        )

    def cutoff_scales(self, channels: range) -> NDArray[np.float64]:
        """The response a cutoff is a fraction of in each channel: 1, or the peak of a holography beam used as the cube stores it."""
        if self.options.pb_model != "holography":
            return np.ones(len(channels))
        return np.array(
            [beam.cutoff_scale for beam in self._holography_beams(channels)]
        )

    def plane_frequencies_hz(self) -> NDArray[np.float64] | None:
        """The frequency of the holography plane each channel's beam is read from, or None if it is not a holography beam."""
        if self.options.pb_model != "holography":
            return None
        return nearest_planes(
            Path(str(self.options.holography_cube)),
            int(self.beam_number or 0),
            self.frequencies_hz,
        )[1]

    def _holography_beams(self, channels: range) -> list[HolographyBeam]:
        """The holography beam of each channel."""
        beams = [self.channel(index) for index in channels]
        holography = [beam for beam in beams if isinstance(beam, HolographyBeam)]
        if len(holography) != len(beams):
            msg = "Only holography beams carry polarisation leakage"
            raise ValueError(msg)
        return holography

    def reach(
        self, cutoff: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Offsets (east, north) from the beam centre, as direction cosines, whose bounding box on a frame holds every position where some channel reaches ``cutoff``.

        A Gaussian beam reaches a circle. A holography beam reaches the pixels of
        its planes at or above the cutoff, which differ from channel to channel
        when the beam fills its grid. No offsets are given if no channel reaches
        the cutoff, and None if the response never falls below it.
        """
        channels = range(len(self.frequencies_hz))
        if self.options.pb_model == "gaussian":
            radii = [self.channel(index).support_radius(cutoff) for index in channels]  # type: ignore[union-attr]
            if any(radius is None for radius in radii):
                return None
            # Points on a circle bound a polygon inside it, which falls short of the
            # circle by many pixels for a wide beam. On a circle larger by
            # 1 / cos(pi / samples) the polygon's edges touch the true circle
            samples = 64
            radius = max(radius for radius in radii if radius is not None) / np.cos(
                np.pi / samples
            )
            if radius >= np.pi / 2:
                return None
            angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
            return np.sin(radius) * np.sin(angles), np.sin(radius) * np.cos(angles)

        groups: dict[tuple[Any, ...], list[HolographyBeam]] = {}
        # The reach needs only the Stokes I planes, not the leakage
        for beam in self._replace(leakage=False)._holography_beams(channels):
            rotation = beam.alpha_rad + beam.frame_rotation_rad
            key = (beam.plane.shape, beam.pixel_scale_rad, beam.centre_pixel, rotation)
            groups.setdefault(key, []).append(beam)
        east, north = [], []
        for group in groups.values():
            y, x = np.nonzero(np.any([beam.reaches(cutoff) for beam in group], axis=0))
            # A response interpolated between pixels reaches the cutoff only within a
            # pixel of one that does, and each pixel lies furthest from the centre,
            # in the same direction, at the smallest scale
            corners = group[0].pixel_offsets(
                np.ravel(y[:, None] + np.array([-1.0, -1.0, 1.0, 1.0])),
                np.ravel(x[:, None] + np.array([-1.0, 1.0, -1.0, 1.0])),
                scale=min(beam.scale for beam in group),
            )
            if corners is None:
                return None
            east.append(corners[0])
            north.append(corners[1])
        if not any(len(values) for values in east):
            return np.zeros(0), np.zeros(0)
        # The centre too, so the box spans every direction out from it
        return np.concatenate([[0.0], *east]), np.concatenate([[0.0], *north])


def _shared_taps(
    beams: list[HolographyBeam],
    east: NDArray[np.float64],
    north: NDArray[np.float64],
) -> list[PatternTaps]:
    """The taps of each beam of one input at offsets, found once for all the channels that share a frequency scale.

    The beams of an input differ only in their planes and scale, so channels at
    the frequencies of the planes, as ASKAP cubes have, all share one set.
    """
    found: dict[float, PatternTaps] = {}
    taps = []
    for beam in beams:
        # Scales a part in 10^9 apart move a position by far less than a pixel
        key = round(beam.scale, 9)
        if key not in found:
            found[key] = beam.taps(east, north)
        taps.append(found[key])
    return taps


def beam_model(
    options: PrimaryBeamOptions,
    info: ImageInfo,
    leakage: bool = False,
    normalise_leakage: bool = True,
    frame_rotation_rad: float = 0.0,
) -> BeamModel | None:
    """The primary beam model of an input, or None when no beam model is used.

    Args:
        options (PrimaryBeamOptions): The beam model to build
        info (ImageInfo): The input the beam belongs to
        leakage (bool, optional): Also read the polarisation leakage, which needs a holography beam. Defaults to False.
        normalise_leakage (bool, optional): Remove the leakage at the beam centre. Defaults to True.
        frame_rotation_rad (float, optional): Position angle at the beam centre of the direction the input's polarisation is referenced to. Defaults to 0.0.

    Raises:
        ValueError: Raised if the input lacks the frequency or beam number the model needs, or leakage is asked of a beam without it

    Returns:
        BeamModel | None: The primary beam model of the input
    """
    if leakage and options.pb_model != "holography":
        msg = "Removing polarisation leakage needs the holography primary beam"
        raise ValueError(msg)
    if options.pb_model == "none":
        return None
    if not info.frequencies_hz:
        msg = f"The frequency of {info.path} is unknown, and the primary beam needs it"
        raise ValueError(msg)
    if options.pb_model == "holography" and (
        options.holography_cube is None or info.beam_number is None
    ):
        msg = f"The holography beam of {info.path} needs a holography cube and a beam number"
        raise ValueError(msg)

    model = BeamModel(
        options=options,
        frequencies_hz=info.frequencies_hz,
        beam_number=info.beam_number,
        leakage=leakage,
        normalise_leakage=normalise_leakage,
        frame_rotation_rad=frame_rotation_rad,
    )
    model.channel(0)  # Fail now, rather than on a worker, if the beam cannot be built
    return model


def make_primary_beam(
    options: PrimaryBeamOptions,
    info: ImageInfo,
    leakage: bool = False,
    normalise_leakage: bool = True,
    frame_rotation_rad: float = 0.0,
) -> PrimaryBeam | None:
    """The primary beam of the first channel of an input, or None when no beam model is used."""
    model = beam_model(options, info, leakage, normalise_leakage, frame_rotation_rad)
    return None if model is None else model.channel(0)


def pb_rotation_from_pol_axis(pol_axis_rad: float) -> float:
    """The holography beam rotation of a field, from the polarisation axis of its measurement set."""
    # Holography cubes are measured with the feeds at -45 degrees
    return pol_axis_rad + np.pi / 4

"""Convolution of images to a common restoring beam.

The convolution is done in the Fourier domain by the ratio of the transforms of
the target and current beams, as the racs-tools ``robust`` mode does, so no
kernel image is ever formed. Images in Jy/beam are rescaled by the ratio of the
beam areas.
"""

from __future__ import annotations

import importlib
import os
from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import scipy.fft
from astropy.io import fits
from capn_crunch import add_options_to_parser, create_options_from_parser
from radio_beam import Beams
from radio_beam.utils import BeamError

from patchwork.io import (
    NO_BEAM,
    BeamShape,
    ImageInfo,
    celestial_wcs,
    check_output_fits,
    create_output_fits,
    read_block,
    read_image_info,
    suffixed_name,
    write_block,
)
from patchwork.logging import logger, setup_logging
from patchwork.options import ConvolveOptions

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from numpy.typing import NDArray

ConvolveBackend = Literal["scipy", "rust"]

NAN_FRACTION = 0.5
"""A convolved pixel with more than this share of its kernel on missing data is blanked"""
MISSING_ROUND_OFF = 1e-8
"""Shares of a kernel below this are taken as none, as the kernel is cut at the same level by its reach"""


class NoUsableBeamError(ValueError):
    """None of the beams a common beam is wanted for is usable."""


def convolve_rs() -> ModuleType:
    """The convolve-rs module, or an error that says how to install it."""
    try:
        return importlib.import_module("convolve_rs")
    except ImportError as error:
        msg = "The rust backend needs convolve-rs, which is installed with `pip install askap-patchwork[rust]`"
        raise ImportError(msg) from error


def check_backend(backend: str, threads: int | None = None) -> None:
    """Check that a convolution backend is known, and for ``rust`` that convolve-rs is installed.

    convolve-rs convolves on one pool of threads in each process, made at its
    first convolution there. ``threads`` sets the size of that pool, through
    ``RAYON_NUM_THREADS``, unless that is set already: by the user, or by the
    first call that gave ``threads``.

    Args:
        backend (str): The backend, ``scipy`` or ``rust``
        threads (int | None, optional): The threads convolve-rs convolves with in this process. Defaults to None, as many as it can see CPUs, unless set already.

    Raises:
        ValueError: Raised if the backend is not known
    """
    if backend not in get_args(ConvolveBackend):
        msg = f"Unknown convolution backend {backend!r}, which must be one of {', '.join(get_args(ConvolveBackend))}"
        raise ValueError(msg)
    if backend == "rust":
        if threads is not None:
            os.environ.setdefault("RAYON_NUM_THREADS", str(threads))
        convolve_rs()


def _rust_beam(cr: ModuleType, beam: BeamShape) -> Any:
    """A convolve-rs beam, which needs its major axis to be the larger, as a ``BeamShape``'s need not be."""
    if beam.bmin_arcsec > beam.bmaj_arcsec:
        beam = BeamShape(beam.bmin_arcsec, beam.bmaj_arcsec, beam.bpa_deg + 90)
    return cr.Beam.from_arcsec(*beam)


def beam_covariance(beam: BeamShape) -> NDArray[np.float64]:
    """The covariance in degrees squared of a beam, along (east, north)."""
    pa = np.radians(beam.bpa_deg)
    major = np.array([np.sin(pa), np.cos(pa)])
    minor = np.array([np.cos(pa), -np.sin(pa)])
    fwhm_to_sigma = 1 / np.sqrt(8 * np.log(2))
    sigma_major = beam.bmaj_arcsec / 3600 * fwhm_to_sigma
    sigma_minor = beam.bmin_arcsec / 3600 * fwhm_to_sigma
    return np.asarray(
        sigma_major**2 * np.outer(major, major)
        + sigma_minor**2 * np.outer(minor, minor)
    )


def beam_from_covariance(covariance: NDArray[np.float64]) -> BeamShape:
    """The beam with a covariance in degrees squared along (east, north), as ``beam_covariance`` gives."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sigma_to_fwhm = np.sqrt(8 * np.log(2)) * 3600
    east, north = eigenvectors[:, 1]
    return BeamShape(
        bmaj_arcsec=float(np.sqrt(eigenvalues[1]) * sigma_to_fwhm),
        bmin_arcsec=float(np.sqrt(max(eigenvalues[0], 0.0)) * sigma_to_fwhm),
        bpa_deg=float(np.degrees(np.arctan2(east, north))),
    )


def _round_up(value: float, decimals: int) -> float:
    """Round up to a number of decimal places."""
    return float(np.ceil(value * 10**decimals) / 10**decimals)


def common_beam(
    beams: list[BeamShape | None],
    cutoff_arcsec: float | None = None,
    circularise: bool = False,
    tolerance: float = 1e-4,
    nsamps: int = 200,
    epsilon: float = 5e-4,
    name: str | None = None,
    backend: ConvolveBackend = "scipy",
) -> BeamShape:
    """The smallest beam that encloses every usable beam, rounded up as racs-tools does.

    Missing, zero and NaN beams, and beams whose major axis is larger than the
    cutoff, are left out. The axes are rounded up to 0.1 arcsec and the position
    angle up to 0.01 degrees.

    Args:
        beams (list[BeamShape | None]): The beams to enclose
        cutoff_arcsec (float | None, optional): Leave out beams with a major axis larger than this. Defaults to None.
        circularise (bool, optional): Make the common beam circular. Defaults to False.
        tolerance (float, optional): Tolerance of the common beam search. Defaults to 1e-4.
        nsamps (int, optional): Number of samples of the common beam search. Defaults to 200.
        epsilon (float, optional): Epsilon of the common beam search. Defaults to 5e-4.
        name (str | None, optional): What the beams are of, such as a channel, for the log. Defaults to None.
        backend (ConvolveBackend, optional): Find the beam with radio_beam (scipy) or with convolve-rs (rust). Defaults to "scipy".

    Raises:
        NoUsableBeamError: Raised when no beam is usable
        BeamError: Raised when the search fails, even at a tenth of the tolerance

    Returns:
        BeamShape: The common beam
    """
    check_backend(backend)
    usable = [
        beam
        for beam in beams
        if beam is not None
        and beam.is_usable()
        and (cutoff_arcsec is None or beam.bmaj_arcsec <= cutoff_arcsec)
    ]
    if not usable:
        msg = f"None of the {len(beams)} beams is usable"
        raise NoUsableBeamError(msg)

    if backend == "rust":
        found_shape = _rust_common_beam(usable, tolerance, nsamps, epsilon)
    else:
        radio_beams = Beams(beams=[beam.to_radio_beam() for beam in usable])
        try:
            found = radio_beams.common_beam(
                tolerance=tolerance, nsamps=nsamps, epsilon=epsilon
            )
        except BeamError:
            logger.warning(
                f"Retrying the common beam search with {tolerance=:.1g} / 10"
            )
            found = radio_beams.common_beam(
                tolerance=tolerance * 0.1, nsamps=nsamps, epsilon=epsilon
            )
        found_shape = BeamShape.from_radio_beam(found)
    bmaj = _round_up(found_shape.bmaj_arcsec, decimals=1)
    shape = BeamShape(
        bmaj_arcsec=bmaj,
        bmin_arcsec=bmaj
        if circularise
        else _round_up(found_shape.bmin_arcsec, decimals=1),
        bpa_deg=0.0 if circularise else _round_up(found_shape.bpa_deg, decimals=2),
    )
    n_missing = sum(beam is None or not beam.is_usable() for beam in beams)
    n_large = len(beams) - len(usable) - n_missing
    left_out = ([f"{n_missing} with no beam"] if n_missing else []) + (
        [f'{n_large} larger than {cutoff_arcsec:g}"'] if n_large else []
    )
    logger.info(
        f"Common beam{f' of {name}' if name else ''} is {shape}, "
        f"from {len(usable)} of {len(beams)} beams"
        + (f" ({', '.join(left_out)})" if left_out else "")
    )
    return shape


def _rust_common_beam(
    beams: list[BeamShape], tolerance: float, nsamps: int, epsilon: float
) -> BeamShape:
    """The unrounded common beam convolve-rs finds, retried as radio_beam's is, and failing as it does."""
    cr = convolve_rs()
    rust_beams = [_rust_beam(cr, beam) for beam in beams]
    try:
        found = cr.common_beam(rust_beams, tolerance, nsamps, epsilon)
    except ValueError:
        logger.warning(f"Retrying the common beam search with {tolerance=:.1g} / 10")
        try:
            found = cr.common_beam(rust_beams, tolerance * 0.1, nsamps, epsilon)
        except ValueError as error:
            raise BeamError(str(error)) from error
    return BeamShape(found.major_arcsec, found.minor_arcsec, found.pa_deg)


def convolve_to_beam(
    image: NDArray[np.floating],
    old_beam: BeamShape,
    new_beam: BeamShape,
    pixel_scale_matrix: NDArray[np.float64],
    cutoff_arcsec: float | None = None,
    backend: ConvolveBackend = "scipy",
) -> NDArray[np.floating]:
    """Convolve an image in Jy/beam from its restoring beam to a larger one.

    The sky beyond the image's edges and under its NaN or infinite pixels is
    missing. Each pixel is divided by the share of its kernel that fell on data,
    and blanked if more than half fell on missing data. The image's own missing
    pixels stay blank.

    Args:
        image (NDArray[np.floating]): The (y, x) image
        old_beam (BeamShape): The restoring beam of the image
        new_beam (BeamShape): The beam to convolve to
        pixel_scale_matrix (NDArray[np.float64]): Degrees (east, north) per (x, y) pixel, e.g. ``WCS.pixel_scale_matrix``
        cutoff_arcsec (float | None, optional): Blank the whole image if its beam major axis is larger than this. Defaults to None.
        backend (ConvolveBackend, optional): Convolve with scipy or with convolve-rs (rust). Defaults to "scipy".

    Raises:
        ValueError: Raised if the new beam does not enclose the old beam

    Returns:
        NDArray[np.floating]: The convolved image, in the precision of the input
    """
    check_backend(backend)
    if cutoff_arcsec is not None and old_beam.bmaj_arcsec > cutoff_arcsec:
        logger.warning(
            f'Blanking an image whose beam, {old_beam}, is larger than {cutoff_arcsec:g}"'
        )
        return np.full_like(image, np.nan)
    if np.allclose(old_beam, new_beam, rtol=1e-9, atol=0):
        return image.copy()

    # The kernel is the difference of the beam covariances, in pixels. radio_beam's
    # deconvolve is not used, as it fails when a common beam falls a hair inside an
    # input, which rounding its position angle up can do, and its kernels ignore
    # pixels that are not aligned with the sky
    to_pixels = np.linalg.inv(pixel_scale_matrix)
    difference = (
        to_pixels
        @ (beam_covariance(new_beam) - beam_covariance(old_beam))
        @ to_pixels.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(difference)
    old_size = np.max(
        np.linalg.eigvalsh(to_pixels @ beam_covariance(old_beam) @ to_pixels.T)
    )
    if np.min(eigenvalues) < -1e-3 * old_size:
        msg = f"Cannot convolve {old_beam} to {new_beam}, which does not enclose it"
        raise ValueError(msg)
    clipped = np.min(eigenvalues) < 0
    eigenvalues = np.clip(eigenvalues, 0, None)
    difference = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    real_dtype: type[np.floating] = (
        np.float32 if image.dtype.itemsize <= 4 else np.float64
    )
    area_ratio = (new_beam.bmaj_arcsec * new_beam.bmin_arcsec) / (
        old_beam.bmaj_arcsec * old_beam.bmin_arcsec
    )
    finite = np.isfinite(image)
    has_missing = not np.all(finite)

    if backend == "rust":
        # The kernel clipped above is passed as the beam it gives the image
        target = (
            beam_from_covariance(
                beam_covariance(old_beam)
                + pixel_scale_matrix @ difference @ pixel_scale_matrix.T
            )
            if clipped
            else new_beam
        )
        data = np.asarray(image, dtype=real_dtype)
        if has_missing:
            # convolve-rs takes NaNs as missing, but not infinities
            data = np.where(finite, data, np.nan).astype(real_dtype, copy=False)
        cr = convolve_rs()
        convolved = cr.smooth(
            data,
            _rust_beam(cr, old_beam),
            _rust_beam(cr, target),
            pixel_scale_matrix=np.asarray(pixel_scale_matrix).tolist(),
            # Jy/beam scales by the areas of the beams passed, so a clipped
            # target is scaled here, by the area of the beam asked for
            bunit="K" if clipped else "Jy/beam",
            nan_fraction=NAN_FRACTION,
        )
        if clipped:
            convolved *= area_ratio
    else:
        ny, nx = image.shape
        # Zeros out to where the kernel holds under 1e-8 of its weight, 6.1 of
        # its sigma, make the circular transforms a linear convolution
        reach = int(np.ceil(6.1 * np.sqrt(np.max(eigenvalues)))) + 1
        padded = (
            scipy.fft.next_fast_len(ny + reach, real=True),
            scipy.fft.next_fast_len(nx + reach, real=True),
        )
        # Images can be large, so each plane-sized array is made once and worked
        # on in place
        taper = _taper(difference, padded, real_dtype)
        convolved = _tapered(image, taper, padded, where=finite)
        convolved *= area_ratio
        for block, missing in _missing_shares(
            finite, has_missing, difference, reach, taper, padded
        ):
            part = convolved[block]
            if missing is None:
                part[...] = np.nan
                continue
            blank = missing > NAN_FRACTION
            # A kernel under a pixel wide rings, so the share can dip below zero
            divide = ~blank & (np.abs(missing) > MISSING_ROUND_OFF)
            np.divide(part, 1 - missing, out=part, where=divide)
            part[blank] = np.nan

    if has_missing:
        convolved[~finite] = np.nan
        logger.info(
            f"Blanked {np.mean(np.isnan(convolved)):.1%} of the convolved image"
        )
    return np.asarray(convolved, dtype=image.dtype)


def _taper(
    difference: NDArray[np.float64],
    shape: tuple[int, int],
    real_dtype: type[np.floating],
) -> NDArray[np.floating]:
    """The transform of a Gaussian kernel of unit gain, from its covariance in pixels, on the ``rfft2`` grid of an array of ``shape``.

    The Nyquist bin of an even axis holds +1/2 and -1/2 cycles per pixel at
    once, so its cross term is the average of the cross terms at both, a cosh.
    Where both axes are even, the corner bin holds all four, and its average is
    the same cosh, so it too takes it once.
    """
    fy = scipy.fft.fftfreq(shape[0]).astype(real_dtype)[:, None]
    fx = scipy.fft.rfftfreq(shape[1]).astype(real_dtype)[None, :]
    exponent: NDArray[np.floating] = (-4 * np.pi**2 * float(difference[0, 1]) * fy) * fx
    row = shape[0] // 2 if shape[0] % 2 == 0 else None
    crosses = [] if row is None else [exponent[row, :]]
    if shape[1] % 2 == 0:
        column = exponent[:, -1]
        crosses += [column] if row is None else [column[:row], column[row + 1 :]]
    for cross in crosses:
        cross[...] = np.logaddexp(cross, -cross) - np.log(2)
    exponent -= 2 * np.pi**2 * float(difference[0, 0]) * fx**2
    exponent -= 2 * np.pi**2 * float(difference[1, 1]) * fy**2
    np.exp(exponent, out=exponent)
    return exponent


def _tapered(
    plane: NDArray[np.generic],
    taper: NDArray[np.floating],
    shape: tuple[int, int],
    fill: float = 0.0,
    where: NDArray[np.bool_] | bool = True,
) -> NDArray[np.floating]:
    """``plane`` in the corner of an array of ``shape`` filled with ``fill`` (and zero where not ``where``), multiplied by ``taper`` in the Fourier domain, and cropped back to the shape of ``plane``."""
    ny, nx = plane.shape
    padded_plane = np.full(shape, fill, dtype=taper.dtype)
    padded_plane[:ny, :nx] = 0
    np.copyto(padded_plane[:ny, :nx], plane, where=where)
    spectrum = scipy.fft.rfft2(padded_plane, overwrite_x=True)
    del padded_plane
    spectrum *= taper
    return np.asarray(scipy.fft.irfft2(spectrum, s=shape, overwrite_x=True)[:ny, :nx])


def _missing_shares(
    finite: NDArray[np.bool_],
    has_missing: bool,
    difference: NDArray[np.float64],
    reach: int,
    taper: NDArray[np.floating],
    padded: tuple[int, int],
    tiled: bool | None = None,
) -> Iterator[tuple[tuple[slice, slice], NDArray[np.floating] | None]]:
    """The share of each pixel's kernel that falls on missing data, in the blocks of an image that have any.

    Missing data is the sky beyond the image, and its pixels that are not
    finite. As in convolve-rs, only the parts of the image within the kernel's
    reach of it are convolved, in square tiles with a halo of that reach. The
    share is none away from the blocks given. The whole image is convolved
    instead where the tiles would not be much smaller than it, or where the
    kernel is too narrow for its transform to have fallen to nothing at the
    highest frequencies: such a kernel rings beyond its reach, differently in a
    tile's transform than in the image's own.

    Args:
        finite (NDArray[np.bool]): Where the image has data
        has_missing (bool): Whether any pixel of the image is not finite
        difference (NDArray[np.float64]): The covariance of the kernel in pixels
        reach (int): The pixels beyond which the kernel's weight is negligible
        taper (NDArray[np.floating]): The kernel's transform for the whole image, padded to ``padded``
        padded (tuple[int, int]): The shape the whole image is padded to
        tiled (bool | None, optional): Whether to tile the image. Defaults to None, tiling where the tiles are much smaller than the image and the kernel is not too narrow.

    Yields:
        tuple[tuple[slice, slice], NDArray[np.floating] | None]: A block of the image, and the share of each of its pixels, or None where it is all missing data
    """
    ny, nx = finite.shape
    size = scipy.fft.next_fast_len(max(6 * reach, 128), real=True)
    if tiled is None:
        highest = max(np.max(taper[padded[0] // 2]), np.max(taper[:, -1]))
        tiled = 8 * size <= 2 * min(ny, nx) and highest < MISSING_ROUND_OFF
    if not tiled:
        yield (slice(None), slice(None)), _tapered(~finite, taper, padded, fill=1.0)
        return

    inner = size - 2 * reach
    tile_taper = _taper(difference, (size, size), taper.dtype.type)
    for r0 in range(0, ny, inner):
        for c0 in range(0, nx, inner):
            # The tile's halo, and the part of it on the image
            ra, ca = r0 - reach, c0 - reach
            rows = slice(max(ra, 0), min(ra + size, ny))
            cols = slice(max(ca, 0), min(ca + size, nx))
            on_image = finite[rows, cols]
            if on_image.shape == (size, size) and (not has_missing or on_image.all()):
                continue
            block = (slice(r0, min(r0 + inner, ny)), slice(c0, min(c0 + inner, nx)))
            if has_missing and not on_image.any():
                yield block, None
                continue
            mask = np.ones((size, size), dtype=taper.dtype)
            mask[
                rows.start - ra : rows.stop - ra, cols.start - ca : cols.stop - ca
            ] = ~on_image
            share = _tapered(mask, tile_taper, (size, size))[
                reach : reach + block[0].stop - r0, reach : reach + block[1].stop - c0
            ]
            if np.any(np.abs(share) > MISSING_ROUND_OFF):
                yield block, share


def convolved_name(path: str | Path, suffix: str = "conv") -> Path:
    """The name of a convolved image: ``suffix`` inserted before a final ``.fits``, or added with ``.fits`` after any other name, so it never names the image itself."""
    return suffixed_name(path, suffix)


def convolve_fits(
    path: str | Path,
    beam: BeamShape,
    output_path: str | Path | None = None,
    suffix: str = "conv",
    cutoff_arcsec: float | None = None,
    overwrite: bool = False,
    backend: ConvolveBackend = "scipy",
) -> Path:
    """Convolve a FITS image to a beam, writing a float32 image with the new beam in its header.

    An image blanked by the cutoff is given a zero beam, as racs-tools does.

    Args:
        path (str | Path): The image to convolve
        beam (BeamShape): The beam to convolve to
        output_path (str | Path | None, optional): Where to write the convolved image. Defaults to None, the image name with ``suffix``.
        suffix (str, optional): Suffix of the default output name. Defaults to "conv".
        cutoff_arcsec (float | None, optional): Blank the image if its beam major axis is larger than this. Defaults to None.
        overwrite (bool, optional): Replace an existing output. Defaults to False.
        backend (ConvolveBackend, optional): Convolve with scipy or with convolve-rs (rust). Defaults to "scipy".

    Raises:
        ValueError: Raised if the image has no usable restoring beam

    Returns:
        Path: The convolved image
    """
    path = Path(path)
    output_path = (
        convolved_name(path, suffix) if output_path is None else Path(output_path)
    )
    check_output_fits(output_path, overwrite=overwrite, input_fits=[path])

    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32)

    old_beam = BeamShape.from_header(header)
    if old_beam is None or not old_beam.is_usable():
        msg = f"{path} has no usable restoring beam"
        raise ValueError(msg)

    plane = data.reshape(data.shape[-2:])
    convolved = convolve_to_beam(
        image=plane,
        old_beam=old_beam,
        new_beam=beam,
        pixel_scale_matrix=celestial_wcs(header).pixel_scale_matrix,
        cutoff_arcsec=cutoff_arcsec,
        backend=backend,
    )
    blanked = cutoff_arcsec is not None and old_beam.bmaj_arcsec > cutoff_arcsec
    (BeamShape(0.0, 0.0, 0.0) if blanked else beam).to_header(header)
    header["HISTORY"] = f"Convolved from {old_beam} to {beam}"

    fits.writeto(
        output_path, convolved.reshape(data.shape), header=header, overwrite=True
    )
    logger.info(f"Written {output_path}")
    return output_path


def convolve_images(
    images: list[Path],
    options: ConvolveOptions = ConvolveOptions(),  # noqa: B008
    overwrite: bool = False,
) -> tuple[BeamShape, list[Path]]:
    """Convolve images to a target beam, or to their common beam, one after another.

    Args:
        images (list[Path]): The images to convolve
        options (ConvolveOptions, optional): How the images are convolved. Defaults to ConvolveOptions().
        overwrite (bool, optional): Replace existing outputs. Defaults to False.

    Returns:
        tuple[BeamShape, list[Path]]: The beam convolved to, and the convolved images
    """
    check_backend(options.convolve_backend)
    # Images matched by a pattern can include ones convolved before, so no output may be any input
    for image in images:
        check_output_fits(
            convolved_name(image, options.suffix),
            overwrite=overwrite,
            input_fits=images,
        )
    beam = target_beam(images, options)
    outputs = [
        convolve_fits(
            image,
            beam=beam,
            suffix=options.suffix,
            cutoff_arcsec=options.beam_cutoff_arcsec,
            overwrite=overwrite,
            backend=options.convolve_backend,
        )
        for image in images
    ]
    return beam, outputs


def target_beam(images: list[Path], options: ConvolveOptions) -> BeamShape:
    """The beam images are convolved to: the target beam if given, otherwise their common beam."""
    if options.target_beam is not None:
        return BeamShape(*options.target_beam)
    return common_beam(
        beams=[BeamShape.from_header(fits.getheader(image)) for image in images],
        cutoff_arcsec=options.beam_cutoff_arcsec,
        circularise=options.circularise,
        tolerance=options.tolerance,
        nsamps=options.nsamps,
        epsilon=options.epsilon,
        backend=options.convolve_backend,
    )


def channel_target_beams(
    infos: list[ImageInfo],
    mode: str,
    options: ConvolveOptions,
    channels: slice,
) -> tuple[BeamShape, ...]:
    """The beam each channel is convolved to, or ``NO_BEAM`` for a channel left blank.

    Channels outside ``channels`` are left blank.

    Args:
        infos (list[ImageInfo]): Every cube
        mode (str): ``natural``, a common beam for each channel, or ``total``, one beam for every channel
        options (ConvolveOptions): How the common beam is found
        channels (slice): The channels that are convolved

    Returns:
        tuple[BeamShape, ...]: The beam of each channel
    """
    n_channels = infos[0].n_channels
    selected = range(*channels.indices(n_channels))
    if options.target_beam is not None:
        target = BeamShape(*options.target_beam)
        return tuple(
            target if channel in selected else NO_BEAM for channel in range(n_channels)
        )

    def common(beams: list[BeamShape | None], name: str | None) -> BeamShape:
        try:
            return common_beam(
                beams,
                cutoff_arcsec=options.beam_cutoff_arcsec,
                circularise=options.circularise,
                tolerance=options.tolerance,
                nsamps=options.nsamps,
                epsilon=options.epsilon,
                name=name,
                backend=options.convolve_backend,
            )
        except NoUsableBeamError as error:
            logger.info(f"Leaving {name or 'the images'} blank: {error}")
            return NO_BEAM

    def channel_name(channel: int) -> str | None:
        if n_channels == 1:
            return None
        frequencies = infos[0].frequencies_hz
        if channel < len(frequencies):
            return f"channel {channel} ({frequencies[channel] / 1e6:.3f} MHz)"
        return f"channel {channel}"

    if mode == "total":
        total = common(
            [info.channel_beams[channel] for info in infos for channel in selected],
            None if n_channels == 1 else f"channels {selected[0]} to {selected[-1]}",
        )
        return tuple(
            total if channel in selected else NO_BEAM for channel in range(n_channels)
        )
    return tuple(
        common([info.channel_beams[channel] for info in infos], channel_name(channel))
        if channel in selected
        else NO_BEAM
        for channel in range(n_channels)
    )


def convolve_cube(
    path: str | Path,
    beams: tuple[BeamShape, ...],
    output_path: str | Path,
    cutoff_arcsec: float | None = None,
    overwrite: bool = False,
    channels: slice | None = None,
    threads: int = 1,
    backend: ConvolveBackend = "scipy",
) -> Path:
    """Convolve some channels of a cube to their beams, a channel at a time in each of ``threads`` threads.

    A channel whose target beam is ``NO_BEAM``, or whose own beam is not usable
    or is larger than the cutoff, is blanked and given no beam. Channels outside
    ``channels`` are given no beam and are not written at all, so they cost no
    time, and read as zeros.

    Each thread holds about three planes of the padded image at once (its
    padded plane, its spectrum and its result), so 16 threads on 10000² planes
    hold some 20 GB.

    Args:
        path (str | Path): The cube to convolve
        beams (tuple[BeamShape, ...]): The beam to convolve each channel to
        output_path (str | Path): Where to write the convolved cube
        cutoff_arcsec (float | None, optional): Blank channels whose beam major axis is larger than this. Defaults to None.
        overwrite (bool, optional): Replace an existing output. Defaults to False.
        channels (slice | None, optional): The channels to convolve. Defaults to None, every channel.
        threads (int, optional): Threads that each convolve a channel at a time. With ``rust``, also the size of convolve-rs's pool of threads in this process, which the channels share, unless that is set already (see ``check_backend``). Defaults to 1.
        backend (ConvolveBackend, optional): Convolve with scipy or with convolve-rs (rust). Defaults to "scipy".

    Returns:
        Path: The convolved cube
    """
    check_backend(backend, threads)
    info = read_image_info(path)
    output_path = Path(output_path)
    check_output_fits(output_path, overwrite=overwrite, input_fits=[path])
    selected = range(*(channels or slice(None)).indices(info.n_channels))

    new_beams = []
    for channel, (old, new) in enumerate(zip(info.channel_beams, beams, strict=True)):
        usable = (
            channel in selected
            and old is not None
            and old.is_usable()
            and new.is_usable()
            and (cutoff_arcsec is None or old.bmaj_arcsec <= cutoff_arcsec)
        )
        new_beams.append(new if usable else NO_BEAM)
        if not usable and channel in selected:
            logger.debug(f"Leaving channel {channel} of {path} blank")

    header = info.header.copy()
    header["HISTORY"] = "Convolved to a common beam by patchwork"
    create_output_fits(output_path, header, tuple(new_beams))
    pixel_scale_matrix = celestial_wcs(info.header).pixel_scale_matrix
    with fits.open(output_path, mode="update", memmap=True) as hdul:
        hdu = hdul[0]
        hdu.data  # noqa: B018  # Map the data once, before any thread writes to it

        def convolve_channel(channel: int) -> None:
            old, new = info.channel_beams[channel], new_beams[channel]
            channel_slice = slice(channel, channel + 1)
            if not new.is_usable() or old is None:
                write_block(
                    hdu,
                    (slice(None), slice(None)),
                    np.full((1, *info.shape), np.nan),
                    channel_slice,
                )
                return
            plane = read_block(path, channels=channel_slice, dtype=np.float32)[0]
            convolved = convolve_to_beam(
                plane, old, new, pixel_scale_matrix, backend=backend
            )
            write_block(hdu, (slice(None), slice(None)), convolved[None], channel_slice)

        if threads > 1 and len(selected) > 1:
            # The FFTs and the copies release the GIL, and each channel is its own
            # part of the file
            with ThreadPoolExecutor(min(threads, len(selected))) as pool:
                list(pool.map(convolve_channel, selected))
        else:
            for channel in selected:
                convolve_channel(channel)
    logger.info(f"Written {output_path}")
    return output_path


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    """The parser of the ``convolve`` command."""
    if parser is None:
        parser = ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="The images to convolve")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files"
    )
    return add_options_to_parser(parser=parser, options_class=ConvolveOptions)


def cli(args: Namespace | None = None) -> None:
    """Run the ``convolve`` command."""
    setup_logging()
    args = get_parser().parse_args() if args is None else args
    convolve_images(
        images=args.images,
        options=create_options_from_parser(
            parser_namespace=args, options_class=ConvolveOptions
        ),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    cli()

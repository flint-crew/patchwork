"""Convolution of images to a common restoring beam.

The convolution is done in the Fourier domain by the ratio of the transforms of
the target and current beams, as the racs-tools ``robust`` mode does, so no
kernel image is ever formed. Images in Jy/beam are rescaled by the ratio of the
beam areas.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

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
    from numpy.typing import NDArray


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

    Raises:
        ValueError: Raised when no beam is usable

    Returns:
        BeamShape: The common beam
    """
    usable = [
        beam
        for beam in beams
        if beam is not None
        and beam.is_usable()
        and (cutoff_arcsec is None or beam.bmaj_arcsec <= cutoff_arcsec)
    ]
    if not usable:
        msg = f"None of the {len(beams)} beams is usable"
        raise ValueError(msg)

    radio_beams = Beams(beams=[beam.to_radio_beam() for beam in usable])
    try:
        found = radio_beams.common_beam(
            tolerance=tolerance, nsamps=nsamps, epsilon=epsilon
        )
    except BeamError:
        logger.warning(f"Retrying the common beam search with {tolerance=:.1g} / 10")
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


def convolve_to_beam(
    image: NDArray[np.floating],
    old_beam: BeamShape,
    new_beam: BeamShape,
    pixel_scale_matrix: NDArray[np.float64],
    cutoff_arcsec: float | None = None,
) -> NDArray[np.floating]:
    """Convolve an image in Jy/beam from its restoring beam to a larger one.

    NaNs are filled with zeros for the convolution and then blanked again, along
    with every pixel where more than half of the convolved signal came from them.

    Args:
        image (NDArray[np.floating]): The (y, x) image
        old_beam (BeamShape): The restoring beam of the image
        new_beam (BeamShape): The beam to convolve to
        pixel_scale_matrix (NDArray[np.float64]): Degrees (east, north) per (x, y) pixel, e.g. ``WCS.pixel_scale_matrix``
        cutoff_arcsec (float | None, optional): Blank the whole image if its beam major axis is larger than this. Defaults to None.

    Raises:
        ValueError: Raised if the new beam does not enclose the old beam

    Returns:
        NDArray[np.floating]: The convolved image, in the precision of the input
    """
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
    eigenvalues = np.clip(eigenvalues, 0, None)
    difference = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    real_dtype: type[np.floating] = (
        np.float32 if image.dtype.itemsize <= 4 else np.float64
    )
    ny, nx = image.shape
    # The transforms are circular, so zeros as wide as the kernel reaches (five of
    # its sigma) keep flux from wrapping between opposite edges
    reach = int(np.ceil(5 * np.sqrt(np.max(eigenvalues))))
    padded = (
        scipy.fft.next_fast_len(ny + reach, real=True),
        scipy.fft.next_fast_len(nx + reach, real=True),
    )
    area_ratio = (new_beam.bmaj_arcsec * new_beam.bmin_arcsec) / (
        old_beam.bmaj_arcsec * old_beam.bmin_arcsec
    )

    # Images can be large, so each plane-sized array is made once and worked on in
    # place. The taper carries the ratio of the beam areas
    fy = scipy.fft.fftfreq(padded[0]).astype(real_dtype)[:, None]
    fx = scipy.fft.rfftfreq(padded[1]).astype(real_dtype)[None, :]
    taper = (2 * float(difference[0, 1]) * fy) * fx
    taper += float(difference[0, 0]) * fx**2
    taper += float(difference[1, 1]) * fy**2
    taper *= -2 * np.pi**2
    np.exp(taper, out=taper)
    taper *= area_ratio

    def tapered(
        plane: NDArray[np.generic], where: NDArray[np.bool_] | bool = True
    ) -> NDArray[np.floating]:
        """``plane``, zero where not ``where``, padded, multiplied by the taper in the Fourier domain, and cropped."""
        padded_plane = np.zeros(padded, dtype=real_dtype)
        np.copyto(padded_plane[:ny, :nx], plane, where=where)
        spectrum = scipy.fft.rfft2(padded_plane, overwrite_x=True)
        del padded_plane
        spectrum *= taper
        return np.asarray(
            scipy.fft.irfft2(spectrum, s=padded, overwrite_x=True)[:ny, :nx]
        )

    # The NaNs are spread first, so only a mask of where they reach is held while
    # the image is convolved
    finite = np.isfinite(image)
    blank = None
    if not np.all(finite):
        blank = ~finite
        blank |= tapered(blank) > 0.5 * area_ratio

    convolved = tapered(image, where=finite)
    if blank is not None:
        convolved[blank] = np.nan
        logger.info(f"Blanked {np.mean(blank):.1%} of the convolved image")

    return np.asarray(convolved, dtype=image.dtype)


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
            )
        except ValueError as error:
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
        threads (int, optional): Threads that each convolve a channel at a time. Defaults to 1.

    Returns:
        Path: The convolved cube
    """
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
            convolved = convolve_to_beam(plane, old, new, pixel_scale_matrix)
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

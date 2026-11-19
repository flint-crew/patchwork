"""Fixtures shared across the test modules"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from numpy.typing import NDArray

from patchwork.io import celestial_wcs
from patchwork.primary_beam import direction_cosines, gaussian_beam
from patchwork.resample import available_cpus

# numba starts a thread for each CPU the process may run on, and the tests of
# the fused backend make tiles with up to 3. On a smaller machine, such as a CI
# runner with 2 CPUs, numba is told to start 4. Nothing above imports numba,
# which reads this when it is imported.
if available_cpus() < 4:
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")

FREQUENCY_HZ = 9.4e8

SkyModel = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
ImageFactory = Callable[..., Path]


def image_header(
    ra: float,
    dec: float,
    size: int = 200,
    cdelt: float = 0.004,
    frequency_hz: float = FREQUENCY_HZ,
) -> fits.Header:
    """A 4D ASKAP-like image header with degenerate FREQ and STOKES axes."""
    header = fits.Header()
    header["NAXIS"] = 4
    axes = [
        ("RA---SIN", ra, -cdelt, size / 2 + 1, size),
        ("DEC--SIN", dec, cdelt, size / 2 + 1, size),
        ("FREQ", frequency_hz, 1e6, 1, 1),
        ("STOKES", 1, 1, 1, 1),
    ]
    for axis, (ctype, crval, cdelt_axis, crpix, naxis) in enumerate(axes, start=1):
        header[f"NAXIS{axis}"] = naxis
        header[f"CTYPE{axis}"] = ctype
        header[f"CRVAL{axis}"] = crval
        header[f"CDELT{axis}"] = cdelt_axis
        header[f"CRPIX{axis}"] = crpix
    header["BMAJ"] = header["BMIN"] = 0.01
    header["BPA"] = 0.0
    header["BUNIT"] = "Jy/beam"
    header["TELESCOP"] = "ASKAP"
    return header


@pytest.fixture
def sky_model() -> SkyModel:
    """A smooth sky of a few extended Gaussian sources, as a function of (RA, Dec) in degrees."""
    sources = [
        (150.0, -30.0, 1.0),
        (150.6, -30.2, 0.5),
        (149.5, -29.7, 2.0),
        (150.3, -29.5, 1.0),
    ]
    fwhm_deg = 0.05

    def model(ra: NDArray[np.float64], dec: NDArray[np.float64]) -> NDArray[np.float64]:
        sky = np.zeros(np.shape(ra))
        for source_ra, source_dec, flux in sources:
            east, north = direction_cosines(source_ra, source_dec, ra, dec)
            offset = np.degrees(np.hypot(east, north))
            sky += flux * np.exp(-4 * np.log(2) * (offset / fwhm_deg) ** 2)
        return sky

    return model


@pytest.fixture
def make_image(tmp_path: Path, sky_model: SkyModel) -> ImageFactory:
    """A factory of images of the sky model, attenuated by a Gaussian primary beam unless ``corrected``."""

    def factory(
        name: str,
        ra: float,
        dec: float,
        size: int = 200,
        cdelt: float = 0.004,
        corrected: bool = False,
        noise: float = 0.0,
        seed: int = 0,
    ) -> Path:
        header = image_header(ra=ra, dec=dec, size=size, cdelt=cdelt)
        wcs = celestial_wcs(header)
        y, x = np.mgrid[:size, :size]
        sky_ra, sky_dec = wcs.pixel_to_world_values(x, y)
        data = sky_model(sky_ra, sky_dec)
        if not corrected:
            east, north = direction_cosines(ra, dec, sky_ra, sky_dec)
            data *= gaussian_beam(FREQUENCY_HZ).response(east, north)
        data += np.random.default_rng(seed).normal(0, noise, data.shape) if noise else 0
        path = tmp_path / name
        fits.writeto(path, data[None, None].astype(np.float32), header)
        return path

    return factory

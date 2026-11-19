"""The frames that linear polarisation is referenced to, and the leakage of Stokes I into it.

Stokes Q and U describe a polarisation angle, measured east of a reference
direction. In a mosaic (by the IAU convention) that direction is north at each
pixel. An interferometric image instead references every pixel to one direction:
north at its phase centre, or for ASKAP at the boresight of the dish, carried
across the field as a fixed direction of the feeds. Away from that position
local north turns away from it, by roughly the offset times ``tan(Dec)``, so the
two frames part ways quickly towards a celestial pole. Each input is therefore
rotated from its own frame into the frame of the mosaic before it is combined.

The frame of an image can be recorded in its header, as patchwork records the
frame of its mosaics, with the keywords ``FRAME_KEYWORDS``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from patchwork.coordinates import east_north, unit_vectors

if TYPE_CHECKING:
    from astropy.io import fits
    from numpy.typing import NDArray

STOKES_CODES = {"I": 1, "Q": 2, "U": 3, "V": 4}
"""FITS codes of the Stokes parameters, as found on a STOKES axis"""

FRAME_KEYWORDS = ("POLFRAME", "POLRA", "POLDEC")
"""The header keywords that record what the Q and U of an image are referenced to.
POLFRAME is NORTH (north at each pixel) or FIXED (north at POLRA, POLDEC, in
degrees, carried across the image)"""


class LinearFrame(NamedTuple):
    """What the polarisation angles of an image are measured from"""

    centre: tuple[float, float] | None = None
    """The (RA, Dec) in degrees whose north the angles are measured from everywhere, or None if they are measured from north at each pixel"""

    def describe(self) -> str:
        """The frame in words."""
        if self.centre is None:
            return "north at each pixel"
        return f"north at RA {self.centre[0]:.5f}, Dec {self.centre[1]:.5f}"

    def matches(self, other: LinearFrame, tolerance_deg: float = 1 / 3600) -> bool:
        """Whether two frames are the same, their positions no more than ``tolerance_deg`` apart."""
        if self.centre is None or other.centre is None:
            return self.centre is None and other.centre is None
        chord = float(
            np.linalg.norm(unit_vectors(*self.centre) - unit_vectors(*other.centre))
        )
        return bool(np.degrees(2 * np.arcsin(min(chord / 2, 1.0))) <= tolerance_deg)


NORTH = LinearFrame()
"""Angles measured from north at each pixel, the IAU convention"""


def header_frame(
    header: fits.Header, path: str | Path | None = None
) -> LinearFrame | None:
    """The frame the keywords of a header give the Q and U of its image, or None if it has none of them.

    Args:
        header (fits.Header): The header
        path (str | Path | None, optional): The file the header is from, named in errors. Defaults to None.

    Raises:
        ValueError: Raised if the keywords do not describe a frame

    Returns:
        LinearFrame | None: The frame, or None if the header does not record one
    """
    if not any(key in header for key in FRAME_KEYWORDS):
        return None
    where = "" if path is None else f" of {Path(path)}"
    kind = str(header.get("POLFRAME", "")).strip().upper()
    if kind == "NORTH":
        return NORTH
    if kind != "FIXED":
        msg = (
            f"The header{where} has POLFRAME {header.get('POLFRAME')!r}, but it must be NORTH or FIXED"
            if "POLFRAME" in header
            else f"The header{where} has POLRA or POLDEC, but no POLFRAME to say what they mean"
        )
        raise ValueError(msg)
    try:
        ra, dec = float(header["POLRA"]), float(header["POLDEC"])
    except (KeyError, TypeError, ValueError):
        ra = dec = np.nan
    if not (np.isfinite(ra) and np.isfinite(dec) and -90 <= dec <= 90):
        msg = f"The header{where} has POLFRAME FIXED, but its POLRA and POLDEC are not a position"
        raise ValueError(msg)
    return LinearFrame((ra % 360, dec))


def write_frame_keywords(header: fits.Header, frame: LinearFrame) -> None:
    """Record in a header what the Q and U of its image are referenced to, in place of any record there."""
    remove_frame_keywords(header)
    if frame.centre is None:
        header["POLFRAME"] = ("NORTH", "Q,U angles from north at each pixel")
        return
    header["POLFRAME"] = ("FIXED", "Q,U angles from north at POLRA,POLDEC")
    header["POLRA"] = (frame.centre[0], "[deg] RA of the north Q,U are from")
    header["POLDEC"] = (frame.centre[1], "[deg] Dec of the north Q,U are from")


def remove_frame_keywords(header: fits.Header) -> None:
    """Remove any record of the frame of Q and U from a header."""
    for key in FRAME_KEYWORDS:
        header.remove(key, ignore_missing=True, remove_all=True)


def frame_angle(
    reference: tuple[float, float], vectors: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The position angle, east of local north, at each position of the direction that is north at ``reference``.

    The direction is the projection of north at the reference position onto the
    sky at each position, which is how a fixed feed direction appears across a
    field. It is 0 at the reference position, and its value at a pole is that of
    the RA the pole is approached along.

    Args:
        reference (tuple[float, float]): The (RA, Dec) in degrees the frame is north at
        vectors (NDArray[np.float64]): Unit vectors of the positions, along the leading axis

    Returns:
        NDArray[np.float64]: Position angles in radians
    """
    _, north = east_north(*reference)
    x, y, z = vectors
    # Unnormalised east (-y, x, 0) and north (-z x, -z y, x^2 + y^2) at each position
    along_east = -north[0] * y + north[1] * x
    along_north = -z * (north[0] * x + north[1] * y) + north[2] * (x**2 + y**2)
    return np.asarray(np.arctan2(along_east, along_north))


def rotate_linear(
    q: NDArray[np.float64], u: NDArray[np.float64], angle: NDArray[np.float64] | float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Turn polarisation angles by ``angle`` radians east, which turns (Q, U) by twice it."""
    cos_2angle, sin_2angle = np.cos(2 * angle), np.sin(2 * angle)
    return q * cos_2angle - u * sin_2angle, q * sin_2angle + u * cos_2angle


def remove_leakage(
    planes: tuple[NDArray[np.float64], ...],
    stokes: tuple[int, ...],
    stokes_i: NDArray[np.float64],
    leakage: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Subtract the leakage of Stokes I from images of Q, U or V.

    Args:
        planes (tuple[NDArray[np.float64], ...]): The images, all in the frame of the input
        stokes (tuple[int, ...]): The FITS Stokes code of each image
        stokes_i (NDArray[np.float64]): The Stokes I image, still carrying the primary beam
        leakage (NDArray[np.float64]): Fractional leakage into (Q, U, V), in the frame of the input

    Raises:
        ValueError: Raised if an image is not of Q, U or V

    Returns:
        tuple[NDArray[np.float64], ...]: The images with the leakage removed
    """
    if any(code not in (2, 3, 4) for code in stokes):
        msg = f"Leakage is only removed from Q, U and V, not from Stokes codes {stokes}"
        raise ValueError(msg)
    return tuple(
        plane - leakage[code - 2] * stokes_i
        for plane, code in zip(planes, stokes, strict=True)
    )

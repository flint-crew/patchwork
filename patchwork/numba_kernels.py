"""Numba versions of the resampling loops, used when numba is installed.

They follow ``patchwork.resample``, so both give the same values to round-off.
A stack of images on one grid, such as the channels of a cube, is interpolated
with the taps of each position found once. Each runs on one thread and releases the GIL, so tiles can be made
in parallel by threads or dask workers without oversubscribing the cores.

The compiled kernels are cached in ``cache_directory()``.
"""

from __future__ import annotations

import functools
import os
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from patchwork.logging import logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@functools.cache
def cache_directory() -> str:
    """The directory numba caches these kernels in: ``NUMBA_CACHE_DIR`` if it is set, otherwise one of this user's own in the temporary directory.

    numba's default is a ``__pycache__`` directory beside this module. When
    several machines share one installation, as the workers of a cluster
    often do, they all compile the kernels at once into that directory. One
    can then read the cache's index while another replaces it, which on NFS
    fails with "Stale file handle" and takes the task with it. The temporary
    directory (``TMPDIR``, usually ``/tmp``) is on each machine instead.

    numba loads whatever it finds in its cache, so the directory must belong
    to this user and be closed to everyone else. If the usual one does not,
    perhaps because someone else made it first, a new one is made with
    ``tempfile.mkdtemp`` and used instead, and the kernels compile again in
    each process. On Windows, which has no user ids, the temporary directory
    already belongs to each user, so the directory is made there and used.

    Returns:
        str: The directory
    """
    if numba.config.CACHE_DIR:
        return str(numba.config.CACHE_DIR)
    getuid = getattr(os, "getuid", None)
    user = None if getuid is None else getuid()
    name = "patchwork-numba" if user is None else f"patchwork-numba-{user}"
    path = Path(tempfile.gettempdir()) / name
    try:
        path.mkdir(mode=0o700, exist_ok=True)
        # Not followed, so a link someone else left in its place is not used
        found = path.lstat()
        private = stat.S_ISDIR(found.st_mode) and (
            user is None or (found.st_uid == user and not found.st_mode & 0o077)
        )
    except OSError:
        private = False
    if not private:
        path = Path(tempfile.mkdtemp(prefix="patchwork-numba-"))
    logger.debug(f"Caching the numba kernels in {path}")
    return str(path)


def _kernel(function: Callable[..., Any], **options: Any) -> Any:
    """Compile a function with numba, releasing the GIL, caching it in ``cache_directory()``.

    numba chooses where a function is cached when it is decorated, so its
    setting is changed only while each kernel is, and anything else in the
    process is cached where numba would cache it. ``options`` are passed to
    ``numba.njit``.
    """
    setting = numba.config.CACHE_DIR
    numba.config.CACHE_DIR = cache_directory()
    try:
        return numba.njit(cache=True, nogil=True, **options)(function)
    finally:
        numba.config.CACHE_DIR = setting


@_kernel
def _taps(
    position: float,
    code: int,
    width: int,
    weights: NDArray[np.float64],
    shift_trig: NDArray[np.float64],
) -> int:
    """Fill ``weights`` with the taps that interpolate a position, returning the index of the first.

    ``shift_trig`` holds the cosine and sine of ``pi k / a`` for each tap ``k`` of
    a Lanczos kernel of half-width ``a``.
    """
    if code == 0:
        weights[0] = 1.0
        return int(np.floor(position + 0.5))

    base = np.floor(position)
    fraction = position - base
    first = width // 2 - 1
    if code == 3:
        # sin(pi (f - k)) is +-sin(pi f), and sin(pi (f - k) / a) follows by angle
        # addition, so a Lanczos kernel needs three trigonometric calls, not 2 per tap
        a = width // 2
        sin_f = np.sin(np.pi * fraction)
        sin_fa, cos_fa = np.sin(np.pi * fraction / a), np.cos(np.pi * fraction / a)
        total = 0.0
        for k in range(width):
            shift = k - first
            distance = fraction - shift
            if abs(distance) < 1e-12:
                weight = 1.0
            else:
                sin_d = -sin_f if shift & 1 else sin_f
                sin_da = sin_fa * shift_trig[0, k] - cos_fa * shift_trig[1, k]
                weight = a * sin_d * sin_da / (np.pi * distance) ** 2
            weights[k] = weight
            total += weight
        for k in range(width):
            weights[k] /= total
        return int(base) - first

    for k in range(width):
        distance = fraction - (k - first)
        if code == 1:
            weights[k] = max(1.0 - abs(distance), 0.0)
        else:
            t = abs(distance)
            if t <= 1:
                weights[k] = (1.5 * t - 2.5) * t**2 + 1
            elif t < 2:
                weights[k] = ((-0.5 * t + 2.5) * t - 4) * t + 2
            else:
                weights[k] = 0.0
    return int(base) - first


@_kernel
def interpolate_stack(
    windows: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    code: int,
    width: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    """Interpolate a (y, x, image) stack at pixel positions, as ``patchwork.resample.interpolate_stack`` does.

    The taps of each position are found once and used for every image. Values
    and validity are returned shaped ``(*x.shape, image)``.
    """
    ny, nx, n_images = windows.shape
    x_flat = x.ravel()
    y_flat = y.ravel()
    values = np.zeros((x_flat.size, n_images))
    valid = np.zeros((x_flat.size, n_images), dtype=np.bool_)
    x_weights = np.empty(width)
    y_weights = np.empty(width)
    shifts = np.pi * (np.arange(width) - (width // 2 - 1)) / max(width // 2, 1)
    shift_trig = np.vstack((np.cos(shifts), np.sin(shifts)))
    total = np.empty(n_images)
    is_valid = np.empty(n_images, dtype=np.bool_)

    for pixel in range(x_flat.size):
        x_position, y_position = x_flat[pixel], y_flat[pixel]
        if not (np.isfinite(x_position) and np.isfinite(y_position)):
            continue
        x_first = _taps(x_position, code, width, x_weights, shift_trig)
        y_first = _taps(y_position, code, width, y_weights, shift_trig)

        total[:] = 0.0
        is_valid[:] = True
        for j in range(width):
            row = y_first + j
            row_outside = row < 0 or row >= ny
            row = min(max(row, 0), ny - 1)
            for i in range(width):
                column = x_first + i
                outside = row_outside or column < 0 or column >= nx
                column = min(max(column, 0), nx - 1)

                tap_weight = y_weights[j] * x_weights[i]
                carries_weight = abs(tap_weight) > 1e-9
                for image in range(n_images):
                    value = windows[row, column, image]
                    blank = not np.isfinite(value)
                    if carries_weight and (outside or blank):
                        is_valid[image] = False
                    total[image] += tap_weight * (0.0 if blank else value)

        for image in range(n_images):
            if is_valid[image]:
                values[pixel, image] = total[image]
                valid[pixel, image] = True

    return values.reshape((*x.shape, n_images)), valid.reshape((*x.shape, n_images))

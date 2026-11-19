"""Resampling inputs onto the output frame with separable interpolation kernels"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from patchwork.io import Slices

Kernel = Literal["nearest", "linear", "cubic", "lanczos2", "lanczos3", "lanczos4"]
Backend = Literal["auto", "numpy", "numba", "fused"]


def kernel_width(kernel: Kernel) -> int:
    """The number of taps of a kernel along each axis."""
    if kernel.startswith("lanczos"):
        return 2 * int(kernel[-1])
    return {"nearest": 1, "linear": 2, "cubic": 4}[kernel]


def resolve_backend(backend: Backend) -> Literal["numpy", "numba", "fused"]:
    """The backend that is used: the one asked for or, with ``auto``, numba when it is installed and numpy when not."""
    has_numba = importlib.util.find_spec("numba") is not None
    if backend == "auto":
        return "numba" if has_numba else "numpy"
    if backend != "numpy" and not has_numba:
        msg = f"The {backend} backend needs numba, which is installed with `pip install askap-patchwork[numba]`"
        raise ImportError(msg)
    return backend


def available_cpus() -> int:
    """The CPUs this process may run on: those a job scheduler such as SLURM has given it, where the system can say, and otherwise all of the machine's."""
    affinity = getattr(os, "sched_getaffinity", None)
    count = len(affinity(0)) if affinity is not None else os.cpu_count()
    return max(count or 1, 1)


def _worker_threads() -> int:
    """The threads of the dask worker this is a task of, which run that many tasks at once, or 1 outside a dask worker."""
    # A task of a dask worker has distributed imported, so it is not imported here
    distributed = sys.modules.get("distributed")
    if distributed is None:
        return 1
    try:
        return max(int(distributed.get_worker().state.nthreads), 1)
    except ValueError:
        return 1


def tile_threads(requested: int) -> int:
    """The threads a task uses: the fused backend makes a tile with them, and every backend measures the noise of an image with them.

    With 0, every CPU the process may run on (``available_cpus``) is shared
    between the tiles it makes at once. A dask worker makes as many as it has
    threads, so one with a single thread and 32 CPUs makes each tile with 32
    threads, and one with a thread for each CPU makes each tile with 1.
    ``patchwork.mosaic.mosaic`` shares the CPUs between its own
    ``max_workers`` before any tile is made.

    Worker processes that share the CPUs of one job cannot see each other, so
    there each would use them all: give the number of threads instead.

    Args:
        requested (int): The threads asked for, ``ResampleOptions.tile_threads``, where 0 is as many as there are CPUs to use

    Raises:
        ValueError: Raised if a negative number is asked for

    Returns:
        int: The number of threads, which is at least 1
    """
    if requested < 0:
        msg = f"tile_threads is {requested}, but must be 0 or more"
        raise ValueError(msg)
    return max(1, requested or available_cpus() // _worker_threads())


def _keys_cubic(distance: NDArray[np.float64]) -> NDArray[np.float64]:
    """The Keys cubic convolution kernel with a = -0.5."""
    t = np.abs(distance)
    near = (1.5 * t - 2.5) * t**2 + 1
    far = ((-0.5 * t + 2.5) * t - 4) * t + 2
    return np.where(t <= 1, near, np.where(t < 2, far, 0.0))


def kernel_taps(
    position: NDArray[np.float64], kernel: Kernel
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """The pixel indices and weights of the taps that interpolate each position.

    Positions are zero-based pixel coordinates along one axis, with pixel centres
    at integers. Both outputs have a trailing axis over the taps, and the weights
    of each position sum to 1.

    Args:
        position (NDArray[np.float64]): Positions to interpolate at
        kernel (Kernel): The interpolation kernel

    Returns:
        tuple[NDArray[np.int64], NDArray[np.float64]]: Index and weight of each tap
    """
    if kernel == "nearest":
        index = np.floor(position + 0.5).astype(np.int64)[..., None]
        return index, np.ones(index.shape)

    width = kernel_width(kernel)
    base = np.floor(position)
    offsets = np.arange(width) - (width // 2 - 1)
    index = base[..., None].astype(np.int64) + offsets
    distance = (position - base)[..., None] - offsets

    if kernel == "linear":
        weights = np.clip(1 - np.abs(distance), 0.0, None)
    elif kernel == "cubic":
        weights = _keys_cubic(distance)
    else:
        a = width // 2
        weights = np.sinc(distance) * np.sinc(distance / a)
        weights /= np.sum(weights, axis=-1, keepdims=True)

    return index, weights


def interpolate(
    window: NDArray[np.floating],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    kernel: Kernel,
    backend: Backend = "numpy",
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    """Interpolate an image at pixel positions with a separable kernel.

    A position is invalid when any tap that carries weight falls on a
    NaN or off the image, so blanked pixels grow by the kernel half-width rather
    than biasing their neighbours.

    Args:
        window (NDArray[np.floating]): The (y, x) image, possibly a region of a larger one
        x (NDArray[np.float64]): Zero-based x position in the window of each output pixel
        y (NDArray[np.float64]): Zero-based y position in the window of each output pixel
        kernel (Kernel): The interpolation kernel
        backend (Backend, optional): Interpolate with numpy or with numba. Defaults to "numpy".

    Returns:
        tuple[NDArray[np.float64], NDArray[np.bool]]: The interpolated values (0 where invalid), and which are valid
    """
    values, valid = interpolate_stack(window[None], x, y, kernel, backend)
    return values[0], valid[0]


def interpolate_stack(
    windows: NDArray[np.floating],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    kernel: Kernel,
    backend: Backend = "numpy",
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    """Interpolate a stack of images that share a grid, such as the channels of a cube, at pixel positions.

    The taps are found once for every position and used for every image, as
    ``interpolate`` would use them for each image alone.

    Args:
        windows (NDArray[np.floating]): The (image, y, x) stack, possibly a region of a larger one
        x (NDArray[np.float64]): Zero-based x position in the window of each output pixel
        y (NDArray[np.float64]): Zero-based y position in the window of each output pixel
        kernel (Kernel): The interpolation kernel
        backend (Backend, optional): Interpolate with numpy or with numba. Defaults to "numpy".

    Returns:
        tuple[NDArray[np.float64], NDArray[np.bool]]: The interpolated values (0 where invalid) and which are valid, both shaped ``(image, *x.shape)``
    """
    if resolve_backend(backend) != "numpy":
        from patchwork import numba_kernels  # noqa: PLC0415

        code = {"nearest": 0, "linear": 1, "cubic": 2}.get(kernel, 3)
        dtype = (
            windows.dtype if windows.dtype in (np.float32, np.float64) else np.float64
        )
        values, valid = numba_kernels.interpolate_stack(
            # Each pixel's taps then read every image from one place in memory
            np.ascontiguousarray(np.moveaxis(windows, 0, -1), dtype=dtype),
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
            code,
            kernel_width(kernel),
        )
        return (
            np.moveaxis(np.asarray(values, dtype=np.float64), -1, 0),
            np.moveaxis(np.asarray(valid, dtype=bool), -1, 0),
        )

    n_images, ny, nx = windows.shape
    finite_position = np.isfinite(x) & np.isfinite(y)
    x_index, x_weights = kernel_taps(np.where(finite_position, x, 0.0), kernel)
    y_index, y_weights = kernel_taps(np.where(finite_position, y, 0.0), kernel)

    blank = ~np.isfinite(windows)
    has_blank = bool(np.any(blank))
    flat = np.reshape(
        np.where(blank, 0.0, windows) if has_blank else windows, (n_images, -1)
    )
    flat_blank = np.reshape(blank, (n_images, -1))
    # Taps that land off the window only need clipping where some of them do
    clip_x = bool(np.min(x_index) < 0 or np.max(x_index) >= nx)
    clip_y = bool(np.min(y_index) < 0 or np.max(y_index) >= ny)

    values = np.zeros((n_images, *x.shape))
    invalid = np.broadcast_to(~finite_position, values.shape).copy()
    for j in range(y_index.shape[-1]):
        row = y_index[..., j]
        row_outside = (row < 0) | (row >= ny) if clip_y else None
        row_start = (np.clip(row, 0, ny - 1) if clip_y else row) * nx
        for i in range(x_index.shape[-1]):
            column = x_index[..., i]
            outside = row_outside
            if clip_x:
                column_outside = (column < 0) | (column >= nx)
                outside = (
                    column_outside if outside is None else outside | column_outside
                )
                column = np.clip(column, 0, nx - 1)
            index = row_start + column

            tap_weight = y_weights[..., j] * x_weights[..., i]
            if outside is not None or has_blank:
                bad = (
                    flat_blank[:, index]
                    if has_blank
                    else np.zeros(values.shape, dtype=bool)
                )
                if outside is not None:
                    bad |= outside
                invalid |= bad & (np.abs(tap_weight) > 1e-9)
            values += tap_weight * flat[:, index]

    return np.where(invalid, 0.0, values), ~invalid


def input_region(
    shape: tuple[int, int],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    kernel: Kernel,
) -> Slices | None:
    """The (y, x) region of an input that interpolating at the positions reads, or None if it reads none of it."""
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return None

    margin = kernel_width(kernel) // 2 + 1
    ny, nx = shape
    x_low = max(int(np.floor(np.min(x[finite]))) - margin, 0)
    x_high = min(int(np.ceil(np.max(x[finite]))) + margin, nx - 1)
    y_low = max(int(np.floor(np.min(y[finite]))) - margin, 0)
    y_high = min(int(np.ceil(np.max(y[finite]))) + margin, ny - 1)
    if x_low > x_high or y_low > y_high:
        return None
    return slice(y_low, y_high + 1), slice(x_low, x_high + 1)

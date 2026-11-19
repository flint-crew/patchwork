from __future__ import annotations

import importlib.util
import os
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from patchwork.resample import (
    Kernel,
    input_region,
    interpolate,
    kernel_taps,
    resolve_backend,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

KERNELS: list[Kernel] = [
    "nearest",
    "linear",
    "cubic",
    "lanczos2",
    "lanczos3",
    "lanczos4",
]


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_weights_sum_to_one(kernel: Kernel) -> None:
    positions = np.random.default_rng(0).uniform(0, 50, 100)
    _, weights = kernel_taps(positions, kernel)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)


@pytest.mark.parametrize("kernel", KERNELS)
def test_identity_reproduces_image(kernel: Kernel) -> None:
    window = np.random.default_rng(1).normal(size=(20, 30))
    y, x = np.mgrid[:20, :30].astype(float)
    values, valid = interpolate(window, x, y, kernel)
    assert np.all(valid)
    assert np.allclose(values, window)


@pytest.mark.parametrize("kernel", KERNELS)
def test_constant_stays_constant(kernel: Kernel) -> None:
    window = np.full((30, 30), 3.0)
    y, x = np.mgrid[10:20, 10:20] + 0.37
    values, valid = interpolate(window, x, y, kernel)
    assert np.all(valid)
    assert np.allclose(values, 3.0)


def test_nan_tap_is_invalid() -> None:
    window = np.ones((10, 10))
    window[5, 5] = np.nan
    x = np.array([4.5, 1.5, 5.0])
    y = np.array([4.5, 1.5, 5.0])
    values, valid = interpolate(window, x, y, "linear")
    assert valid.tolist() == [False, True, False]
    assert values[0] == 0
    assert values[1] == 1


def test_off_image_is_invalid() -> None:
    window = np.ones((10, 10))
    x = np.array([-0.6, -0.4, 9.0, 9.2, np.nan])
    y = np.full(5, 5.0)
    _, linear_valid = interpolate(window, x, y, "linear")
    assert linear_valid.tolist() == [False, False, True, False, False]
    _, nearest_valid = interpolate(window, x, y, "nearest")
    assert nearest_valid.tolist() == [False, True, True, True, False]


@pytest.mark.parametrize(
    ("kernel", "tolerance"), [("linear", 2e-2), ("cubic", 1e-3), ("lanczos3", 5e-3)]
)
def test_shifted_gaussian(kernel: Kernel, tolerance: float) -> None:
    def gaussian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(-((x - 32) ** 2 + (y - 30) ** 2) / (2 * 4**2)))

    y, x = np.mgrid[:64, :64].astype(float)
    window = gaussian(x, y)
    y_out, x_out = np.mgrid[10:50, 10:50] + np.array([0.3, 0.6])[:, None, None]
    values, valid = interpolate(window, x_out, y_out, kernel)
    assert np.all(valid)
    assert np.max(np.abs(values - gaussian(x_out, y_out))) < tolerance


def test_input_region() -> None:
    x = np.array([[5.2, 40.0], [np.nan, 12.0]])
    y = np.array([[-20.0, 3.0], [np.nan, 8.0]])
    region = input_region((30, 30), x, y, "linear")
    assert region == (slice(0, 11), slice(3, 30))
    assert input_region((30, 30), x + 100, y, "linear") is None
    assert (
        input_region((30, 30), np.full(2, np.nan), np.full(2, np.nan), "linear") is None
    )


numba_installed = pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="numba is not installed"
)


@numba_installed
@pytest.mark.parametrize("kernel", KERNELS)
def test_numba_matches_numpy(kernel: Kernel) -> None:
    rng = np.random.default_rng(3)
    window = rng.normal(size=(60, 70))
    window[20:23, 30:50] = np.nan
    y, x = np.mgrid[:50, :50].astype(float)
    x = x * 1.3 - 3.7 + 0.01 * y
    y = y * 1.1 + 0.4
    x[0, 0] = np.nan
    numpy_values, numpy_valid = interpolate(window, x, y, kernel, backend="numpy")
    numba_values, numba_valid = interpolate(window, x, y, kernel, backend="numba")
    assert np.array_equal(numpy_valid, numba_valid)
    assert np.allclose(numpy_values, numba_values, rtol=0, atol=1e-12)
    assert 0 < np.sum(numpy_valid) < numpy_valid.size

    # float32 images, as inputs are read, are interpolated in float64 by both
    single = window.astype(np.float32)
    exact, _ = interpolate(single.astype(np.float64), x, y, kernel, backend="numpy")
    for backend in ("numpy", "numba"):
        values, valid = interpolate(single, x, y, kernel, backend=backend)
        assert values.dtype == np.float64
        assert np.array_equal(valid, numpy_valid)
        assert np.allclose(values, exact, rtol=0, atol=1e-12)


def test_resolve_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_backend("numpy") == "numpy"
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert resolve_backend("auto") == "numpy"
    with pytest.raises(ImportError, match="numba"):
        resolve_backend("numba")


@pytest.fixture
def fresh_cache_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Callable[[], str]]:
    """``cache_directory``, worked out again with NUMBA_CACHE_DIR unset and the test's own temporary directory."""
    numba = pytest.importorskip("numba")
    from patchwork import numba_kernels  # noqa: PLC0415

    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    numba_kernels.cache_directory.cache_clear()
    yield numba_kernels.cache_directory
    numba_kernels.cache_directory.cache_clear()


def test_numba_cache_is_private(
    fresh_cache_directory: Callable[[], str], tmp_path: Path
) -> None:
    directory = fresh_cache_directory()
    assert directory == str(tmp_path / f"patchwork-numba-{os.getuid()}")
    assert stat.S_IMODE(Path(directory).stat().st_mode) == 0o700
    assert fresh_cache_directory() == directory


@pytest.mark.parametrize("found", ["open", "link"])
def test_numba_cache_skips_a_directory_it_cannot_trust(
    fresh_cache_directory: Callable[[], str], tmp_path: Path, found: str
) -> None:
    """A directory others can write to, or a link in its place, is not used: numba runs what it finds there."""
    usual = tmp_path / f"patchwork-numba-{os.getuid()}"
    if found == "open":
        usual.mkdir()
        usual.chmod(0o777)
    else:
        (tmp_path / "elsewhere").mkdir(mode=0o700)
        usual.symlink_to(tmp_path / "elsewhere")
    directory = fresh_cache_directory()
    assert directory != str(usual)
    assert directory.startswith(str(tmp_path / "patchwork-numba-"))
    assert stat.S_IMODE(Path(directory).stat().st_mode) == 0o700


def test_numba_cache_without_user_ids(
    fresh_cache_directory: Callable[[], str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows has no user ids, and a temporary directory of each user's own."""
    monkeypatch.delattr(os, "getuid")
    assert fresh_cache_directory() == str(tmp_path / "patchwork-numba")


def test_numba_cache_follows_numba_cache_dir(
    fresh_cache_directory: Callable[[], str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numba  # noqa: PLC0415

    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "chosen"))
    assert fresh_cache_directory() == str(tmp_path / "chosen")


@numba_installed
def test_kernels_are_cached_there() -> None:
    """The kernels are cached in ``cache_directory()``, and numba's own setting is left as it was."""
    import numba  # noqa: PLC0415

    from patchwork import numba_kernels  # noqa: PLC0415

    for kernel in (numba_kernels.interpolate_stack, numba_kernels._taps):
        assert kernel._cache.cache_path.startswith(numba_kernels.cache_directory())
    assert os.environ.get("NUMBA_CACHE_DIR", "") == numba.config.CACHE_DIR

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from astropy.io import fits

from patchwork.io import (
    create_output_fits,
    data_location,
    read_rows,
    write_rows,
    write_tile_rows,
)
from patchwork.logging import logger
from patchwork.tile import TileResult
from patchwork.write import _duration, bounded_results, logged_progress, write_tiles

if TYPE_CHECKING:
    from pathlib import Path

    from patchwork.plan import MosaicPlan


class Tracker:
    """Counts the calls started, so the calls submitted and not yet taken can be checked."""

    def __init__(self) -> None:
        self.started = 0
        self.lock = threading.Lock()

    def call(self, index: int, fail: bool = False) -> int:
        with self.lock:
            self.started += 1
        time.sleep(0.01)
        if fail:
            msg = f"Call {index} failed"
            raise RuntimeError(msg)
        return index


class CountingExecutor(ThreadPoolExecutor):
    """A thread pool that counts the calls submitted to it."""

    submitted = 0

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        self.submitted += 1
        return super().submit(fn, *args, **kwargs)


def test_bounded_results_holds_no_more_than_the_limit() -> None:
    tracker = Tracker()
    taken: list[int] = []
    with CountingExecutor(max_workers=8) as executor:
        for result in bounded_results(
            executor, ((tracker.call, (index,)) for index in range(40)), lambda: 3
        ):
            # No more than the limit are submitted and not yet taken, this one included
            assert executor.submitted - len(taken) <= 3
            taken.append(result)
            time.sleep(0.005)
    assert sorted(taken) == list(range(40))
    assert executor.submitted == 40


def test_bounded_results_cancels_the_rest_on_failure() -> None:
    tracker = Tracker()
    with ThreadPoolExecutor(max_workers=1) as executor:
        calls = ((tracker.call, (index, index == 2)) for index in range(100))
        with pytest.raises(RuntimeError, match="Call 2 failed"):
            list(bounded_results(executor, calls, lambda: 4))
    # Only those submitted before the failure were made
    assert tracker.started <= 2 + 4


def test_bounded_results_follows_the_limit() -> None:
    """A limit that grows, as a cluster does when it scales, is asked again and followed."""
    tracker = Tracker()
    asked: list[int] = []

    def limit() -> int:
        asked.append(len(asked))
        return 1 if len(asked) == 1 else 6

    in_flight: list[int] = []
    taken = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for _ in bounded_results(
            executor,
            ((tracker.call, (index,)) for index in range(30)),
            limit,
            refresh_s=0.0,
        ):
            in_flight.append(tracker.started - taken)
            taken += 1
    assert len(asked) > 1
    assert max(in_flight) > 1


def test_logged_progress(caplog: pytest.LogCaptureFixture) -> None:
    """Results pass on unchanged, with a message at most every interval and one when all are done."""
    assert list(logged_progress(iter(range(4)), 4, "tiles", interval_s=0.0)) == [
        0,
        1,
        2,
        3,
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0].startswith("Made 1 of 4 tiles (25%) in 0s, about 0s left")
    assert messages[-1] == "Made 4 of 4 tiles (100%) in 0s"
    caplog.clear()
    list(logged_progress(range(100), 100, "strips", interval_s=3600.0))
    assert [record.getMessage() for record in caplog.records] == [
        "Made 100 of 100 strips (100%) in 0s"
    ]


def test_duration() -> None:
    assert [_duration(seconds) for seconds in (0.4, 42, 185, 3725)] == [
        "0s",
        "42s",
        "3m 05s",
        "1h 02m",
    ]


def strip_test_header(axes: str) -> fits.Header:
    """A header of 7 rows of 5 pixels, with 3 channels on a FREQ axis in the order ``axes`` gives, or none."""
    header = fits.Header()
    cards = [("RA---SIN", 5), ("DEC--SIN", 7)]
    if axes == "freq_stokes":
        cards += [("FREQ", 3), ("STOKES", 1)]
    elif axes == "stokes_freq":
        cards += [("STOKES", 1), ("FREQ", 3)]
    header["NAXIS"] = len(cards)
    for axis, (ctype, size) in enumerate(cards, start=1):
        header[f"NAXIS{axis}"] = size
        header[f"CTYPE{axis}"] = ctype
        header[f"CRVAL{axis}"] = 1.0
        header[f"CDELT{axis}"] = 1.0
        header[f"CRPIX{axis}"] = 1.0
    return header


@pytest.mark.parametrize("axes", ["freq_stokes", "stokes_freq", "none"])
def test_rows_written_in_place(tmp_path: Path, axes: str) -> None:
    """Rows are written where astropy finds them, and nothing else in the file changes."""
    path = create_output_fits(tmp_path / "out.fits", strip_test_header(axes))
    before = path.read_bytes()
    location = data_location(path)
    channels = slice(1, 3) if axes != "none" else slice(0, 1)
    block = np.arange(
        (channels.stop - channels.start) * 3 * 5, dtype=np.float32
    ).reshape(-1, 3, 5)
    write_rows(location, slice(2, 5), channels, block)

    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data).reshape(-1, 7, 5)
    expected = np.zeros((location.n_channels, 7, 5), dtype=np.float32)
    expected[channels, 2:5] = block
    assert np.array_equal(data, expected)
    assert path.read_bytes()[: location.offset] == before[: location.offset]
    assert np.array_equal(read_rows(location, slice(2, 5), channels.start), block[0])
    assert np.array_equal(
        read_rows(location, slice(1, 7), channels.start, step=2),
        expected[channels.start, 1:7:2],
    )

    # Blocks that do not fill the rows they are given would write over others
    for rows, channels, shape in (
        (slice(0, 1), slice(0, 1), (1, 1, 4)),
        (slice(0, 1), slice(0, 1), (1, 2, 5)),
        (slice(6, 8), slice(0, 1), (1, 2, 5)),
        (
            slice(0, 1),
            slice(0, location.n_channels + 1),
            (location.n_channels + 1, 1, 5),
        ),
    ):
        with pytest.raises(ValueError, match="does not fill"):
            write_rows(location, rows, channels, np.zeros(shape))


def test_rows_only_of_float32(tmp_path: Path) -> None:
    path = tmp_path / "double.fits"
    fits.writeto(path, np.zeros((7, 5)), strip_test_header("none"))
    with pytest.raises(ValueError, match="not plain float32"):
        data_location(path)


def test_neighbouring_strips_written_at_once(tmp_path: Path) -> None:
    """Strips one row high, sharing every block of the file, are each written whole by many writers at once."""
    path = create_output_fits(tmp_path / "out.fits", strip_test_header("freq_stokes"))
    location = data_location(path)
    calls = [(row, channel) for row in range(7) for channel in range(3)] * 20

    def write(row: int, channel: int) -> None:
        block = np.full((1, 1, 5), 10 * channel + row, dtype=np.float32)
        write_rows(location, slice(row, row + 1), slice(channel, channel + 1), block)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda call: write(*call), calls))
    data = np.asarray(fits.getdata(path)).reshape(3, 7, 5)
    expected = 10 * np.arange(3)[:, None, None] + np.arange(7)[None, :, None]
    assert np.array_equal(data, np.broadcast_to(expected, (3, 7, 5)))


def test_tile_rows_written_in_place(tmp_path: Path) -> None:
    """Part of some rows is written where astropy finds it, and nothing else in the file changes."""
    path = create_output_fits(tmp_path / "out.fits", strip_test_header("freq_stokes"))
    location = data_location(path)
    block = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2) + 1
    write_tile_rows(location, (slice(2, 5), slice(1, 3)), slice(1, 3), block)
    expected = np.zeros((3, 7, 5), dtype=np.float32)
    expected[1:3, 2:5, 1:3] = block
    assert np.array_equal(np.asarray(fits.getdata(path)).reshape(3, 7, 5), expected)

    for region, shape in (
        ((slice(0, 1), slice(0, 2)), (1, 1, 3)),
        ((slice(0, 1), slice(4, 6)), (1, 1, 2)),
        ((slice(6, 8), slice(0, 2)), (1, 2, 2)),
    ):
        with pytest.raises(ValueError, match="does not fill"):
            write_tile_rows(location, region, slice(0, 1), np.zeros(shape))


def band_test_plan(tmp_path: Path, n_channels: int) -> tuple[MosaicPlan, list[Any]]:
    """A plan of a mosaic of one image and one weight, 13 x 11 pixels over some channels, in tiles of 4 pixels."""
    ny, nx, size = 11, 13, 4
    header = fits.Header()
    header["NAXIS"] = 3
    for axis, (ctype, length) in enumerate(
        (("RA---SIN", nx), ("DEC--SIN", ny), ("FREQ", n_channels)), start=1
    ):
        header[f"NAXIS{axis}"] = length
        header[f"CTYPE{axis}"] = ctype
    names = SimpleNamespace(
        images=(create_output_fits(tmp_path / "image.fits", header),),
        weights=(create_output_fits(tmp_path / "weight.fits", header),),
        noises=(),
    )
    tiles = [
        (slice(y, min(y + size, ny)), slice(x, min(x + size, nx)))
        for y in range(0, ny, size)
        for x in range(0, nx, size)
    ]
    plan = SimpleNamespace(
        names=names,
        frame=SimpleNamespace(shape=(ny, nx)),
        tiles=tiles,
        channels=slice(0, n_channels),
    )
    return cast("MosaicPlan", plan), tiles


def band_test_results(
    tiles: list[Any], chunks: list[slice], seed: int = 0
) -> tuple[list[TileResult], np.ndarray]:
    """Tiles of random images and weights, over chunks of channels, in a random order, and the whole image and weight they make."""
    rng = np.random.default_rng(seed)
    ny = max(tile[0].stop for tile in tiles)
    nx = max(tile[1].stop for tile in tiles)
    whole = rng.normal(size=(2, chunks[-1].stop, ny, nx)).astype(np.float32)
    results = [
        TileResult(
            slices=tile,
            channels=chunk,
            images=(whole[0][(chunk, *tile)],),
            weights=(whole[1][(chunk, *tile)],),
        )
        for chunk in chunks
        for tile in tiles
    ]
    order = rng.permutation(len(results))
    return [results[index] for index in order], whole


@pytest.mark.parametrize(
    "buffer_bands",
    # Every band held, none, or room for one band at a time
    [100, 0, 1],
)
def test_tiles_written_in_bands(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, buffer_bands: int
) -> None:
    """Tiles arriving in any order, over chunks of channels, are written where they belong, whether gathered into bands or written a row at a time."""
    plan, tiles = band_test_plan(tmp_path, n_channels=4)
    chunks = [slice(0, 2), slice(2, 4)]
    results, whole = band_test_results(tiles, chunks)
    # A band is 4 rows of 13 pixels over 2 channels of the image and the weight
    band_bytes = 4 * 2 * 2 * 4 * 13
    with caplog.at_level(logging.INFO, logger=logger.name):
        write_tiles(plan, results, buffer_bytes=buffer_bands * band_bytes)
    assert np.array_equal(fits.getdata(plan.names.images[0]), whole[0])
    assert np.array_equal(fits.getdata(plan.names.weights[0]), whole[1])
    written_by_rows = [
        record for record in caplog.records if "a row at a time" in record.message
    ]
    # Said once, however many bands do not fit
    assert len(written_by_rows) == (0 if buffer_bands == 100 else 1)


def test_unfinished_band_raises(tmp_path: Path) -> None:
    """Tiles that stop before a band is whole are not quietly left unwritten."""
    plan, tiles = band_test_plan(tmp_path, n_channels=1)
    results, _ = band_test_results(tiles, [slice(0, 1)])
    with pytest.raises(RuntimeError, match="Not every tile"):
        write_tiles(plan, results[:-1])

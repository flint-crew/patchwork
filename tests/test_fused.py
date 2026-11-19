from __future__ import annotations

import importlib.util
import shutil
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from astropy.io import fits
from distributed import Client, LocalCluster

from patchwork.io import read_window
from patchwork.mosaic import mosaic
from patchwork.options import (
    CombineOptions,
    FrameOptions,
    MosaicOptions,
    PolarisationOptions,
    PrimaryBeamOptions,
    ResampleOptions,
    RunOptions,
    WeightOptions,
)
from patchwork.pipeline.flow import run_pipeline
from patchwork.resample import available_cpus, resolve_backend, tile_threads

from .test_cube import (
    BEAM_CENTRES,
    FREQUENCIES,
    add_leakage,
    all_cubes,
    cube_options_for_test,
    write_beam_cubes,
    write_leaky_holography_cube,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from patchwork.plan import MosaicNames

    from .conftest import ImageFactory

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="numba is not installed"
)

THREADS = (1, 3)
"""Threads the fused backend is tried with: its serial kernels, and numba's threads"""


def read(path: Path) -> NDArray[np.float32]:
    """The data of a FITS file or a zarr store."""
    if path.suffix == ".zarr":
        import zarr  # noqa: PLC0415

        return np.asarray(zarr.open_array(str(path), mode="r")[...])
    return np.asarray(fits.getdata(path))


def assert_same_mosaic(got: MosaicNames, expected: MosaicNames) -> None:
    """Two mosaics have the same pixels blank, and agree to the round-off of their float32 values."""
    paths = zip(
        (*got.images, *got.weights, *got.noises),
        (*expected.images, *expected.weights, *expected.noises),
        strict=True,
    )
    for got_path, expected_path in paths:
        got_data, expected_data = read(got_path), read(expected_path)
        assert np.array_equal(np.isfinite(got_data), np.isfinite(expected_data))
        assert np.sum(np.isfinite(expected_data)) > 0
        scale = np.nanmax(np.abs(expected_data))
        assert np.allclose(
            got_data, expected_data, rtol=1e-5, atol=1e-6 * scale, equal_nan=True
        ), got_path.name


def assert_fused_matches(
    images: list[Path],
    tmp_path: Path,
    options: MosaicOptions,
    threads: tuple[int, ...] = THREADS,
    **kwargs: Any,
) -> MosaicNames:
    """Mosaic with the numpy backend and with the fused one, which must agree, and give the numpy mosaic."""
    tmp_path.mkdir(exist_ok=True)
    numpy = options.resample.with_options(backend="numpy")  # type: ignore[arg-type]
    expected = mosaic(
        images, tmp_path / "numpy", options._replace(resample=numpy), **kwargs
    )
    for count in threads:
        fused = options.resample.with_options(backend="fused", tile_threads=count)  # type: ignore[arg-type]
        got = mosaic(
            images,
            tmp_path / f"fused{count}",
            options._replace(resample=fused),
            **kwargs,
        )
        assert_same_mosaic(got, expected)
    return expected


@pytest.fixture
def noisy_images(make_image: ImageFactory) -> list[Path]:
    """Three overlapping images, one with a patch of blank pixels."""
    images = [
        make_image("a.fits", 150.0, -30.0, noise=1e-3, seed=1),
        make_image("b.fits", 150.5, -29.9, noise=2e-3, seed=2),
        make_image("c.fits", 150.2, -30.3, noise=1e-3, seed=3),
    ]
    with fits.open(images[1], mode="update") as hdul:
        hdul[0].data[0, 0, 60:75, 80:120] = np.nan
    return images


@pytest.mark.parametrize(
    "kernel", ["nearest", "linear", "cubic", "lanczos2", "lanczos3"]
)
@pytest.mark.parametrize("frame_type", ["reference", "fit"])
def test_kernels(
    noisy_images: list[Path], tmp_path: Path, kernel: str, frame_type: str
) -> None:
    """Every kernel, on a grid an input shares (where taps of no weight land on blank pixels) and on one none does."""
    options = MosaicOptions(
        frame=FrameOptions(frame_type=frame_type),  # type: ignore[arg-type]
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.1),
        resample=ResampleOptions(kernel=kernel, tile_size=120),  # type: ignore[arg-type]
    )
    expected = assert_fused_matches(noisy_images, tmp_path, options)
    data = read(expected.image)
    assert 0 < np.sum(np.isnan(data)) < data.size


@pytest.mark.parametrize("weight_type", ["primary_beam", "weight_image", "combined"])
@pytest.mark.parametrize("weight_state", ["corrected", "inherent", "weighted"])
@pytest.mark.parametrize("combine_type", ["weighted", "average", "sum"])
def test_weighting_and_combining(
    noisy_images: list[Path],
    tmp_path: Path,
    weight_type: str,
    weight_state: str,
    combine_type: str,
) -> None:
    options = MosaicOptions(
        frame=FrameOptions(frame_type="fit"),
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.1),
        weight=WeightOptions(
            weight_type=weight_type,  # type: ignore[arg-type]
            weight_state=weight_state,  # type: ignore[arg-type]
            noise_image=True,
        ),
        combine=CombineOptions(combine_type=combine_type),  # type: ignore[arg-type]
    )
    assert_fused_matches(noisy_images, tmp_path, options, threads=(3,))


@pytest.mark.parametrize("pb_model", ["gaussian", "none"])
def test_beam_models_and_no_cutoff(
    noisy_images: list[Path], tmp_path: Path, pb_model: str
) -> None:
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(pb_model=pb_model, pb_cutoff=0.0),  # type: ignore[arg-type]
        weight=WeightOptions(weight_state="corrected"),
    )
    assert_fused_matches(noisy_images, tmp_path, options)


@pytest.fixture
def beam_cubes(tmp_path: Path) -> list[dict[str, Path]]:
    return [
        write_beam_cubes(tmp_path, f"field.beam{index:02d}", centre, size=100)
        for index, centre in enumerate(BEAM_CENTRES)
    ]


@pytest.mark.parametrize("output_frame", ["north", "centre"])
@pytest.mark.parametrize("input_frame", ["crval", "north"])
def test_polarisation_frames(
    beam_cubes: list[dict[str, Path]],
    tmp_path: Path,
    input_frame: str,
    output_frame: str,
) -> None:
    """Q and U are rotated from each frame an input may have into each the mosaic may have, across a pole."""
    polarisation = PolarisationOptions(
        input_frame=input_frame,  # type: ignore[arg-type]
        output_frame=output_frame,  # type: ignore[arg-type]
    )
    options = cube_options_for_test(
        polarisation=polarisation, resample=ResampleOptions(tile_size=70)
    )
    assert_fused_matches(all_cubes(beam_cubes), tmp_path, options)


def test_blank_channels_and_some_channels(tmp_path: Path) -> None:
    beams = [
        write_beam_cubes(
            tmp_path, "beam00", BEAM_CENTRES[0], size=100, blank_channel=2
        ),
        write_beam_cubes(tmp_path, "beam01", BEAM_CENTRES[1], size=100),
    ]
    options = cube_options_for_test()
    assert_fused_matches(all_cubes(beams), tmp_path, options)
    some = RunOptions(channels=(1, 4), channel_chunk=2)
    assert_fused_matches(all_cubes(beams), tmp_path / "some", options, run_options=some)


def holography_options(
    holography: Path, scale: str = "centre", **kwargs: Any
) -> MosaicOptions:
    return cube_options_for_test(
        primary_beam=PrimaryBeamOptions(
            pb_model="holography",
            holography_cube=holography,
            holography_alpha_rad=0.4,
            holography_scale=scale,  # type: ignore[arg-type]
            pb_cutoff=0.2,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("scale", ["centre", "stored"])
@pytest.mark.parametrize("off_frequency", [False, True])
def test_holography_beams(
    beam_cubes: list[dict[str, Path]], tmp_path: Path, scale: str, off_frequency: bool
) -> None:
    """Holography beams at the frequencies of the channels, which share their taps, and between them, where each channel has its own."""
    holography = write_leaky_holography_cube(tmp_path / "holo.fits")
    with fits.open(holography, mode="update") as hdul:
        hdul[0].data[1] *= 1.3
        if off_frequency:
            hdul[0].header["CDELT3"] *= 0.83
    assert_fused_matches(
        all_cubes(beam_cubes), tmp_path, holography_options(holography, scale)
    )


@pytest.mark.parametrize("keep_centre_leakage", [False, True])
def test_leakage_of_q_u_and_v(tmp_path: Path, keep_centre_leakage: bool) -> None:
    """The leakage of Stokes I is removed from Q, U and V, with Stokes I mosaicked or only read."""
    holography = write_leaky_holography_cube(tmp_path / "holo.fits")
    with fits.open(holography, mode="update") as hdul:
        hdul[0].data[:, 3] = 0.5 * hdul[0].data[:, 1] - 0.2 * hdul[0].data[:, 2]
    beams = [
        write_beam_cubes(tmp_path, f"leaky.beam{index:02d}", centre, size=100)
        for index, centre in enumerate(BEAM_CENTRES)
    ]
    for index, beam in enumerate(beams):
        add_leakage(beam, index, 0.4, 0.03 - 0.02j)
        # Stokes V with the values of U, which is enough to see what is removed from it
        beam["v"] = beam["u"].with_name(beam["u"].name.replace(".u.", ".v."))
        shutil.copy(beam["u"], beam["v"])
        fits.setval(beam["v"], "CRVAL4", value=4)

    polarisation = PolarisationOptions(
        input_frame="crval",
        remove_leakage=True,
        keep_centre_leakage=keep_centre_leakage,
    )
    options = holography_options(holography, polarisation=polarisation)
    everything = assert_fused_matches(all_cubes(beams), tmp_path / "all", options)
    assert len(everything.images) == 4
    # Without Stokes I among the images it is read only to remove its leakage
    polarised = [path for beam in beams for key, path in beam.items() if key != "i"]
    assert_fused_matches(
        polarised,
        tmp_path / "polarised",
        options,
        stokes_i_images=[beam["i"] for beam in beams],
    )


def test_weights_logs_and_weight_images(
    beam_cubes: list[dict[str, Path]], tmp_path: Path
) -> None:
    """Weights from logs go through the kernels, and inputs with weight images are added as the numba backend adds them."""
    cubes = all_cubes(beam_cubes[:2])
    logs, weight_cubes = [], []
    for index, path in enumerate(cubes):
        weights = 1 + index // 3 + 0.1 * np.arange(len(FREQUENCIES))
        log = path.with_suffix(".txt")
        log.write_text(
            "#Channel Weight\n"
            + "".join(f"{channel} {weight}\n" for channel, weight in enumerate(weights))
        )
        logs.append(log)
        weight_cube = path.with_suffix(".weight.fits")
        shutil.copy(path, weight_cube)
        with fits.open(weight_cube, mode="update") as hdul:
            hdul[0].data[...] = weights[None, :, None, None]
        weight_cubes.append(weight_cube)

    options = MosaicOptions(
        frame=FrameOptions(frame_type="fit"),
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.2),
        polarisation=PolarisationOptions(input_frame="crval"),
    )
    assert_fused_matches(cubes, tmp_path / "logs", options, weights=logs)
    assert_fused_matches(cubes, tmp_path / "cubes", options, weights=weight_cubes)
    by_image = options._replace(weight=WeightOptions(weight_type="weight_image"))
    assert_fused_matches(cubes, tmp_path / "image", by_image, weights=logs)


@pytest.mark.parametrize(
    "run_options",
    [
        RunOptions(fits_writer="workers", channel_chunk=2),
        RunOptions(output_format="zarr", zarr_chunk=64, channel_chunk=3),
    ],
)
def test_tasks_that_write(
    beam_cubes: list[dict[str, Path]], tmp_path: Path, run_options: RunOptions
) -> None:
    """Strips of whole rows, which are split along their rows between the threads, and zarr chunks."""
    options = cube_options_for_test(resample=ResampleOptions(tile_size=64))
    assert_fused_matches(
        all_cubes(beam_cubes), tmp_path, options, run_options=run_options
    )


def test_debug_logs_where_the_time_went(
    beam_cubes: list[dict[str, Path]], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """At DEBUG level each tile logs how long it waited, added and wrote, for either number of threads."""
    cubes = [cube["i"] for cube in beam_cubes]
    options = cube_options_for_test(
        resample=ResampleOptions(backend="fused", tile_size=64, tile_threads=2)
    )
    with caplog.at_level("DEBUG", logger="patchwork"):
        mosaic(
            cubes,
            output_base=tmp_path / "debug",
            options=options,
            run_options=RunOptions(fits_writer="workers", channel_chunk=5),
        )
    tiles = [line for line in caplog.text.splitlines() if "waited" in line]
    writes = [line for line in caplog.text.splitlines() if "Wrote rows" in line]
    assert tiles
    assert len(writes) == len(tiles)
    assert "on 2 threads, 2 in the kernels and 2 reading" in tiles[0]
    assert "MB of them" in tiles[0]
    assert "2 threads" in writes[0]
    caplog.clear()
    with caplog.at_level("INFO", logger="patchwork"):
        mosaic(
            cubes,
            output_base=tmp_path / "info",
            options=options,
            run_options=RunOptions(fits_writer="workers", channel_chunk=5),
        )
    assert "waited" not in caplog.text


def test_kernels_can_use_fewer_threads_than_the_tile(
    beam_cubes: list[dict[str, Path]], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """kernel_threads keeps CPUs for the readers; more than tile_threads is ignored, and the mosaic is unchanged."""
    cubes = all_cubes(beam_cubes)
    options = cube_options_for_test(
        resample=ResampleOptions(backend="fused", tile_size=64, tile_threads=3)
    )
    with caplog.at_level("DEBUG", logger="patchwork"):
        fewer = mosaic(
            cubes,
            output_base=tmp_path / "fewer",
            options=options._replace(
                resample=options.resample.model_copy(update={"kernel_threads": 2})
            ),
        )
    assert "on 3 threads, 2 in the kernels and 1 reading" in caplog.text
    caplog.clear()
    with caplog.at_level("DEBUG", logger="patchwork"):
        more = mosaic(
            cubes,
            output_base=tmp_path / "more",
            options=options._replace(
                resample=options.resample.model_copy(update={"kernel_threads": 9})
            ),
        )
    assert "on 3 threads, 3 in the kernels and 3 reading" in caplog.text
    assert_same_mosaic(fewer, more)


@pytest.mark.parametrize(
    ("threads", "requested", "expected"),
    [
        (16, 0, (10, 6)),  # the default: 5 of 8 in the kernels
        (32, 0, (20, 12)),
        (4, 0, (2, 2)),  # 2.5 rounds to 2
        (3, 0, (3, 3)),  # too few to split: every thread does both
        (1, 0, (1, 1)),
        (16, 12, (12, 4)),  # as asked
        (16, 16, (16, 16)),  # all the threads asked for: every thread does both
        (16, 20, (16, 16)),
    ],
)
def test_split_threads(threads: int, requested: int, expected: tuple[int, int]) -> None:
    """kernel_threads 0 splits the threads 5:3 from four up; a count is used as given, and one at or above tile_threads has every thread do both."""
    from patchwork.fused import split_threads  # noqa: PLC0415

    assert split_threads(threads, requested) == expected


def test_tiles_made_at_once_share_the_threads(
    beam_cubes: list[dict[str, Path]], tmp_path: Path
) -> None:
    """Python threads that each make a tile take turns at numba's threads, and with 0 share the CPUs between them."""
    cubes = all_cubes(beam_cubes)
    options = cube_options_for_test(resample=ResampleOptions(tile_size=50))
    assert_fused_matches(cubes, tmp_path, options, threads=(1, 2, 0), max_workers=3)


def test_tile_threads(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from patchwork import fused, fused_kernels, resample  # noqa: PLC0415

    assert tile_threads(1) == 1
    assert tile_threads(500) == 500
    assert tile_threads(0) == available_cpus()
    monkeypatch.setattr(resample, "available_cpus", lambda: 2)
    assert tile_threads(0) == 2
    with pytest.raises(ValueError, match="0 or more"):
        tile_threads(-1)

    # The kernels have no more threads than numba, which is said once
    most = fused_kernels.numba_threads()
    assert fused.kernel_threads(most) == most
    assert fused.kernel_threads(0) == min(2, most)
    with caplog.at_level("WARNING", logger="patchwork"):
        assert fused.kernel_threads(most + 5) == most
        assert fused.kernel_threads(most + 5) == most
    assert caplog.text.count("as numba has no more") == 1


def test_worker_threads_share_the_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dask worker makes as many tiles at once as it has threads, so each is given its share of the CPUs."""
    from patchwork import resample  # noqa: PLC0415

    monkeypatch.setattr(resample, "available_cpus", lambda: 12)
    assert tile_threads(0) == 12
    for worker_threads, expected in ((1, 12), (3, 4), (16, 1)):
        with (
            LocalCluster(
                n_workers=1,
                threads_per_worker=worker_threads,
                processes=False,
                dashboard_address=None,
            ) as cluster,
            Client(cluster) as client,
        ):
            assert client.submit(tile_threads, 0, pure=False).result() == expected
            assert client.submit(tile_threads, 5, pure=False).result() == 5


def test_pipeline_on_a_worker_with_spare_cpus(
    beam_cubes: list[dict[str, Path]], tmp_path: Path
) -> None:
    """One dask thread makes every tile, each on the threads its worker leaves spare."""
    fused = ResampleOptions(backend="fused", tile_threads=0, tile_size=64)
    options = cube_options_for_test(resample=fused)
    expected = mosaic(
        all_cubes(beam_cubes),
        tmp_path / "numpy",
        options._replace(resample=fused.with_options(backend="numpy")),  # type: ignore[arg-type]
    )
    with (
        LocalCluster(
            n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None
        ) as cluster,
        Client(cluster) as client,
    ):
        got = run_pipeline(
            all_cubes(beam_cubes), tmp_path / "dask", client=client, options=options
        )
    assert_same_mosaic(got, expected)


def test_chunks_cover_a_region_once() -> None:
    from patchwork import fused_kernels  # noqa: PLC0415

    for n_rows, n_columns, threads in ((512, 512, 32), (15, 18164, 32), (3, 5, 8)):
        covered = np.zeros((n_rows, n_columns), dtype=int)
        chunks = fused_kernels.block_chunks(n_rows, n_columns, threads)
        for first_row, last_row, first_column, last_column in chunks:
            covered[first_row:last_row, first_column:last_column] += 1
        assert np.all(covered == 1)
        assert np.max(chunks[:, 3] - chunks[:, 2]) <= 1024
        # Enough blocks for every thread, where there are pixels enough
        assert len(chunks) >= min(threads, n_rows * n_columns // 1024)

    rows = fused_kernels.row_chunks(1000, 7)
    assert sorted(index for first, last in rows for index in range(first, last)) == (
        list(range(1000))
    )


def test_kernels_are_cached_with_the_others() -> None:
    from patchwork import fused_kernels, numba_kernels  # noqa: PLC0415

    for kernel in (
        fused_kernels.add_block,
        fused_kernels.add_chunks,
        fused_kernels.finalise_rows,
        fused_kernels.finalise_chunks,
    ):
        assert kernel._cache.cache_path.startswith(numba_kernels.cache_directory())


def test_channel_runs() -> None:
    from patchwork.fused import channel_runs  # noqa: PLC0415

    every = np.ones(16, dtype=bool)
    assert channel_runs(every) == [slice(0, 16)]
    assert channel_runs(every, 5) == [
        slice(start, start + 3) for start in range(0, 15, 3)
    ] + [slice(15, 16)]
    assert channel_runs(every, 100) == [slice(index, index + 1) for index in range(16)]
    some = np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert channel_runs(some) == [slice(0, 2), slice(3, 7), slice(9, 10)]
    assert channel_runs(some, 3) == [
        slice(0, 2),
        slice(3, 5),
        slice(5, 7),
        slice(9, 10),
    ]
    assert channel_runs(np.zeros(4, dtype=bool), 2) == []


def test_unusable_channels_are_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the channels of an input that hold an image are read, a stretch at a time."""
    from patchwork import fused  # noqa: PLC0415

    beams = [
        write_beam_cubes(tmp_path, "beam00", BEAM_CENTRES[0], size=100, blank_channel=2)
    ]
    read: list[tuple[str, int, int]] = []

    def recorded(path: Path, region: Any, channels: slice, out: Any) -> Any:
        read.append((path.name, channels.start, channels.stop))
        return read_window(path, region, channels, out)

    monkeypatch.setattr(fused, "read_window", recorded)
    options = cube_options_for_test(resample=ResampleOptions(backend="fused"))
    mosaic(all_cubes(beams), tmp_path / "field", options)
    assert sorted(set(read)) == sorted(
        (f"beam00.{stokes}.fits", start, stop)
        for stokes in "iqu"
        for start, stop in ((0, 2), (3, 5))
    )


def test_read_window(tmp_path: Path) -> None:
    """Windows are read straight from plain float32 files, and by astropy from any other, with the same result."""
    import gzip  # noqa: PLC0415

    from patchwork import io  # noqa: PLC0415

    cube = write_beam_cubes(tmp_path, "beam00", BEAM_CENTRES[0], size=40)["q"]
    casa = write_beam_cubes(
        tmp_path, "casa", BEAM_CENTRES[0], size=40, casa_order=True
    )["q"]
    double = tmp_path / "double.fits"
    with fits.open(cube) as hdul:
        fits.writeto(double, hdul[0].data.astype(np.float64), hdul[0].header)
    zipped = tmp_path / "zipped.fits.gz"
    zipped.write_bytes(gzip.compress(cube.read_bytes()))

    region, channels = (slice(3, 30), slice(10, 21)), slice(1, 4)
    expected = io.read_block(cube, region, channels, dtype=np.float32)
    for path, mapped in ((cube, True), (casa, True), (double, False), (zipped, False)):
        status = path.stat()
        location = io._mapped_location(str(path), status.st_mtime_ns, status.st_size)
        assert (location is not None) == mapped
        out = np.zeros((3, 27, 11), dtype=np.float32)
        assert io.read_window(path, region, channels, out) is out
        assert np.array_equal(out, expected)
    # Every channel, and a file that has been rewritten since it was last read
    with fits.open(cube, mode="update") as hdul:
        hdul[0].data[...] += 1
    out = np.zeros((5, 27, 11), dtype=np.float32)
    io.read_window(cube, region, None, out)
    assert np.allclose(out[1:4], expected + 1)


def test_backend_needs_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    with pytest.raises(ImportError, match="fused backend needs numba"):
        resolve_backend("fused")

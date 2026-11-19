from __future__ import annotations

import logging
import multiprocessing
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from astropy.io import fits

from patchwork.combine import accumulate, empty_sums, finalise
from patchwork.coordinates import offset_vectors, ra_dec
from patchwork.frame import (
    Frame,
    build_frame,
    input_bounding_box,
    new_wcs,
)
from patchwork.io import BeamShape, ImageInfo, celestial_wcs, read_image_info
from patchwork.logging import logger
from patchwork.mosaic import _own_pool, _PlanFile, cli, get_parser, mosaic
from patchwork.options import (
    CombineOptions,
    MosaicOptions,
    PrimaryBeamOptions,
    ResampleOptions,
    RunOptions,
    WeightOptions,
)
from patchwork.plan import InputImages, contributing_box, output_names, prepare_inputs
from patchwork.primary_beam import BeamModel, gaussian_beam
from patchwork.tile import PlaneWeight

if TYPE_CHECKING:
    from pathlib import Path

    from .conftest import ImageFactory, SkyModel


def read_mosaic(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        return np.array(hdul[0].data[0, 0]), hdul[0].header.copy()


def sky_on_frame(sky_model: SkyModel, header: fits.Header) -> np.ndarray:
    wcs = celestial_wcs(header)
    y, x = np.mgrid[: header["NAXIS2"], : header["NAXIS1"]]
    return sky_model(*wcs.pixel_to_world_values(x, y))


def test_combine_types() -> None:
    # Two images that share a weight, from two inputs, over three pixels where only two have weight
    sums = empty_sums((1, 3), n_images=2)
    for first, second, input_weights in (
        ([2.0, 0.0, 6.0], [1.0, 0.0, 3.0], [1.0, 0.0, 2.0]),
        ([4.0, 0.0, 2.0], [2.0, 0.0, 1.0], [1.0, 0.0, 2.0]),
    ):
        accumulate(
            sums,
            [np.array([first]), np.array([second])],
            1.0,
            np.array([input_weights]),
        )
    (image, half), weight = finalise(sums)
    assert np.allclose(image, [[3.0, np.nan, 2.0]], equal_nan=True)
    assert np.allclose(half, [[1.5, np.nan, 1.0]], equal_nan=True)
    assert np.allclose(weight, [[2.0, np.nan, 4.0]], equal_nan=True)
    assert image.dtype == weight.dtype == np.float32

    # Weighted images over their weights are the estimates, of 2 and 4
    for combine_type, image_value, weight_value in (
        ("average", 3.0, 4 / (1 + 1 / 3)),
        ("sum", 6.0, 1 / (1 + 1 / 3)),
    ):
        sums = empty_sums((1, 1), combine_type=combine_type)  # type: ignore[arg-type]
        for value, input_weight in ((2.0, 1.0), (12.0, 3.0)):
            accumulate(
                sums,
                [np.array([[value]])],
                1.0,
                np.array([[input_weight]]),
                combine_type,  # type: ignore[arg-type]
            )
        (image,), weight = finalise(sums, combine_type)  # type: ignore[arg-type]
        assert np.isclose(image[0, 0], image_value)
        assert np.isclose(weight[0, 0], weight_value)

    with pytest.raises(ValueError, match="weighted mean"):
        finalise(empty_sums((1, 1)), "average")


@pytest.mark.parametrize("weight_state", ["inherent", "corrected"])
def test_mosaic_recovers_sky(
    make_image: ImageFactory, sky_model: SkyModel, tmp_path: Path, weight_state: str
) -> None:
    corrected = weight_state == "corrected"
    images = [
        make_image("a.beam00.fits", 150.0, -30.0, corrected=corrected),
        make_image("b.beam01.fits", 150.5, -29.9, corrected=corrected),
        make_image("c.beam02.fits", 149.7, -29.6, corrected=corrected),
    ]
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.1),
        weight=WeightOptions(
            image_weight="none",
            weight_state=weight_state,  # type: ignore[arg-type]
        ),
        resample=ResampleOptions(tile_size=128),
    )
    names = mosaic(images, tmp_path / "field", options)

    image, header = read_mosaic(names.image)
    weight, _ = read_mosaic(names.weight)
    expected = sky_on_frame(sky_model, header)
    good = np.isfinite(image)
    assert np.mean(good) > 0.5
    assert np.max(np.abs(image[good] - expected[good])) < 1e-3
    assert np.array_equal(good, np.isfinite(weight))
    assert header["BUNIT"] == "Jy/beam"
    assert header["TELESCOP"] == "ASKAP"
    assert "BUNIT" not in fits.getheader(names.weight)


def test_cutoff_blanks_edges(make_image: ImageFactory, tmp_path: Path) -> None:
    images = [make_image("a.fits", 150.0, -30.0, size=400)]
    unweighted = WeightOptions(image_weight="none")
    loose = mosaic(
        images,
        tmp_path / "loose",
        MosaicOptions(
            primary_beam=PrimaryBeamOptions(pb_cutoff=0.0), weight=unweighted
        ),
    )
    tight = mosaic(
        images,
        tmp_path / "tight",
        MosaicOptions(
            primary_beam=PrimaryBeamOptions(pb_cutoff=0.5), weight=unweighted
        ),
    )

    loose_image, header = read_mosaic(loose.image)
    tight_image, _ = read_mosaic(tight.image)
    assert np.all(np.isfinite(loose_image[2:-2, 2:-2]))

    # A cutoff of 0.5 on the beam keeps pixels within the half power radius
    wcs = celestial_wcs(header)
    y, x = np.mgrid[: header["NAXIS2"], : header["NAXIS1"]]
    ra, dec = wcs.pixel_to_world_values(x, y)
    radius = np.hypot((ra - 150.0) * np.cos(np.radians(dec)), dec + 30.0)
    half_width = np.degrees(1.09 * 2.998e8 / 9.4e8 / 12) / 2
    assert np.all(np.isfinite(tight_image[radius < half_width * 0.98]))
    assert np.all(np.isnan(tight_image[radius > half_width * 1.02]))


def test_blanking_ignores_noise(make_image: ImageFactory, tmp_path: Path) -> None:
    """An input far noisier than the others keeps all of its area within the cutoff."""
    images = [
        make_image("quiet.fits", 150.0, -30.0, noise=1e-3, seed=1),
        make_image("noisy.fits", 150.6, -30.0, noise=5e-2, seed=2),
    ]
    beam = PrimaryBeamOptions(pb_cutoff=0.1)
    weighted = mosaic(
        images,
        tmp_path / "weighted",
        MosaicOptions(primary_beam=beam, weight=WeightOptions(image_weight="mad")),
    )
    unweighted = mosaic(
        images,
        tmp_path / "unweighted",
        MosaicOptions(primary_beam=beam, weight=WeightOptions(image_weight="none")),
    )
    weighted_image, header = read_mosaic(weighted.image)
    unweighted_image, _ = read_mosaic(unweighted.image)
    weight, _ = read_mosaic(weighted.weight)
    assert np.array_equal(np.isfinite(weighted_image), np.isfinite(unweighted_image))
    # Where only the noisy input covers, its weight is 2500 times below the peak
    x, y = celestial_wcs(header).world_to_pixel_values(150.9, -30.0)
    pixel = (round(float(y)), round(float(x)))
    assert np.isfinite(weighted_image[pixel])
    assert weight[pixel] < np.nanmax(weight) / 1000


def test_mosaic_outside_coverage_is_blank(
    make_image: ImageFactory, tmp_path: Path
) -> None:
    images = [
        make_image("a.fits", 150.0, -30.0, size=60),
        make_image("b.fits", 150.0, -29.0, size=60),
    ]
    names = mosaic(
        images,
        tmp_path / "gap",
        MosaicOptions(
            primary_beam=PrimaryBeamOptions(pb_cutoff=0.0),
            weight=WeightOptions(image_weight="none"),
        ),
    )
    image, header = read_mosaic(names.image)
    assert header["NAXIS2"] > 250
    assert np.all(np.isnan(image[100:150]))


def test_average_and_sum(
    make_image: ImageFactory, sky_model: SkyModel, tmp_path: Path
) -> None:
    images = [
        make_image("a.fits", 150.0, -30.0, corrected=True),
        make_image("b.fits", 150.0, -30.0, corrected=True),
    ]
    flat = PrimaryBeamOptions(pb_model="none")
    weights = WeightOptions(image_weight="none")
    average = mosaic(
        images,
        tmp_path / "average",
        MosaicOptions(
            primary_beam=flat,
            weight=weights,
            combine=CombineOptions(combine_type="average"),
        ),
    )
    total = mosaic(
        images,
        tmp_path / "sum",
        MosaicOptions(
            primary_beam=flat,
            weight=weights,
            combine=CombineOptions(combine_type="sum"),
        ),
    )

    average_image, header = read_mosaic(average.image)
    sum_image, _ = read_mosaic(total.image)
    sum_weight, _ = read_mosaic(total.weight)
    expected = sky_on_frame(sky_model, header)
    assert np.allclose(average_image, expected, atol=1e-5)
    assert np.allclose(sum_image, 2 * expected, atol=1e-5)
    assert np.allclose(sum_weight, 0.5)


def test_weight_images(
    make_image: ImageFactory, sky_model: SkyModel, tmp_path: Path
) -> None:
    images = [
        make_image("a.fits", 150.0, -30.0, corrected=True),
        make_image("b.fits", 150.3, -30.0, corrected=True),
    ]
    weight_paths = []
    for image in images:
        with fits.open(image) as hdul:
            weight_path = image.with_suffix(".rms.fits")
            fits.writeto(weight_path, np.full_like(hdul[0].data, 0.5), hdul[0].header)
            weight_paths.append(weight_path)

    options = MosaicOptions(
        weight=WeightOptions(
            weight_type="weight_image",
            weight_state="corrected",
            weight_map_type="rms",
        )
    )
    names = mosaic(images, tmp_path / "weighted", options, weights=weight_paths)
    mosaic_image, header = read_mosaic(names.image)
    mosaic_weight, _ = read_mosaic(names.weight)
    good = np.isfinite(mosaic_image)
    expected = sky_on_frame(sky_model, header)
    assert np.allclose(mosaic_image[good], expected[good], atol=1e-3)
    assert np.isclose(np.nanmax(mosaic_weight), 8.0)

    with pytest.raises(ValueError, match="weights for"):
        mosaic(images, tmp_path / "bad", options, weights=weight_paths[:1])


def test_mosaic_beam_and_overwrite(make_image: ImageFactory, tmp_path: Path) -> None:
    images = [make_image("a.fits", 150.0, -30.0, size=50)]
    names = mosaic(images, tmp_path / "beam", beam=BeamShape(40.0, 30.0, 10.0))
    assert np.isclose(fits.getheader(names.image)["BMAJ"], 40 / 3600)
    with pytest.raises(FileExistsError):
        mosaic(images, tmp_path / "beam")
    mosaic(images, tmp_path / "beam", overwrite=True)


def test_mosaic_cli(make_image: ImageFactory, tmp_path: Path) -> None:
    images = [
        make_image("a.fits", 150.0, -30.0, size=50),
        make_image("b.fits", 150.2, -30.0, size=50),
    ]
    args = get_parser().parse_args(
        [
            *map(str, images),
            "--output-base",
            str(tmp_path / "cli"),
            "--kernel",
            "linear",
            "--pb-cutoff",
            "0.2",
            "--centre",
            "150.1",
            "-30",
            "--image-weight",
            "none",
            "--max-workers",
            "2",
            "--processes",
        ]
    )
    assert args.processes
    cli(args)
    names = output_names(tmp_path / "cli", (1,))
    assert names.image.exists()
    assert names.weight.exists()
    assert np.isclose(fits.getheader(names.image)["CRVAL1"], 150.1)


@pytest.mark.parametrize(
    "resample",
    [
        ResampleOptions(coordinate_step=1, backend="numpy"),
        ResampleOptions(coordinate_step=64, backend="numpy"),
        ResampleOptions(backend="auto", tile_size=90),
    ],
)
def test_speed_settings_agree(
    make_image: ImageFactory, tmp_path: Path, resample: ResampleOptions
) -> None:
    """Coordinate steps, backends and tile sizes change the speed, not the mosaic."""
    images = [
        make_image("a.fits", 150.0, -30.0, noise=1e-3, seed=1),
        make_image("b.fits", 150.5, -29.9, noise=1e-3, seed=2),
    ]
    beam = PrimaryBeamOptions(pb_cutoff=0.1)
    reference = mosaic(images, tmp_path / "reference", MosaicOptions(primary_beam=beam))
    other = mosaic(
        images,
        tmp_path / "other",
        MosaicOptions(primary_beam=beam, resample=resample),
    )
    expected, _ = read_mosaic(reference.image)
    got, _ = read_mosaic(other.image)
    assert np.array_equal(np.isfinite(expected), np.isfinite(got))
    # Positions may move by the 0.001 pixel tolerance of the coordinate grid
    assert np.nanmax(np.abs(expected - got)) < 1e-4


def test_mosaic_logs_its_progress(
    make_image: ImageFactory, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    images = [
        make_image("a.fits", 150.0, -30.0, noise=1e-3),
        make_image("b.fits", 150.5, -29.9, noise=1e-3, seed=4),
    ]
    mosaic(
        images,
        tmp_path / "field",
        MosaicOptions(resample=ResampleOptions(tile_size=64)),
    )
    assert re.search(r"Made (\d+) of \1 tiles \(100%\)", caplog.text)


def test_threads_match_serial(make_image: ImageFactory, tmp_path: Path) -> None:
    images = [
        make_image("a.fits", 150.0, -30.0, noise=1e-3),
        make_image("b.fits", 150.5, -29.9, noise=1e-3, seed=4),
    ]
    options = MosaicOptions(resample=ResampleOptions(tile_size=64))
    serial = mosaic(images, tmp_path / "serial", options)
    threaded = mosaic(images, tmp_path / "threaded", options, max_workers=3)
    for serial_path, threaded_path in (
        (serial.image, threaded.image),
        (serial.weight, threaded.weight),
    ):
        assert np.array_equal(
            fits.getdata(serial_path), fits.getdata(threaded_path), equal_nan=True
        )


@pytest.mark.parametrize("fits_writer", ["driver", "workers"])
def test_processes_match_serial(
    make_image: ImageFactory, tmp_path: Path, fits_writer: Literal["driver", "workers"]
) -> None:
    """Processes make the mosaic one thread makes, whether they or the main process write it."""
    images = [
        make_image("a.fits", 150.0, -30.0, noise=1e-3),
        make_image("b.fits", 150.5, -29.9, noise=1e-3, seed=4),
    ]
    options = MosaicOptions(resample=ResampleOptions(tile_size=64))
    run_options = RunOptions(fits_writer=fits_writer)
    serial = mosaic(images, tmp_path / "serial", options, run_options)
    spread = mosaic(
        images,
        tmp_path / "processes",
        options,
        run_options,
        max_workers=2,
        processes=True,
    )
    for serial_path, spread_path in (
        (serial.image, spread.image),
        (serial.weight, spread.weight),
    ):
        assert np.array_equal(
            fits.getdata(serial_path), fits.getdata(spread_path), equal_nan=True
        )


def test_processes_are_sent_the_plan_once(tmp_path: Path) -> None:
    """What a task is sent is a short reference to the plan, which each process reads once."""
    plan = {"tiles": list(range(10000))}
    shared = _PlanFile.write(tmp_path, plan)
    sent = pickle.dumps(shared)
    assert len(sent) < 500
    received = pickle.loads(sent)
    assert received == plan
    assert pickle.loads(sent) is received


def test_processes_log_through_the_main_process(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """What a process logs reaches the main process's handlers, as what a thread logs does."""
    with (
        caplog.at_level(logging.INFO, logger="patchwork"),
        _own_pool(1, processes=True) as (pool, _),
    ):
        pool.submit(logger.info, "made in a process").result()
    made = [
        record
        for record in caplog.records
        if record.getMessage() == "made in a process"
    ]
    assert len(made) == 1
    assert made[0].process != os.getpid()


def test_processes_cannot_be_started_everywhere(
    make_image: ImageFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = [make_image("a.fits", 150.0, -30.0, size=50)]
    with pytest.raises(ValueError, match="with an executor"):
        mosaic(
            images, tmp_path / "given", executor=ThreadPoolExecutor(), processes=True
        )
    # A dask worker is a daemonic process, and those cannot start processes of their own
    monkeypatch.setattr(
        multiprocessing, "current_process", lambda: SimpleNamespace(daemon=True)
    )
    with pytest.raises(ValueError, match="daemonic"):
        mosaic(images, tmp_path / "daemon", processes=True)
    assert not list(tmp_path.glob("daemon*"))


def test_beam_cutoff_limits_contributions(make_image: ImageFactory) -> None:
    """Each input is only worked on where its beam is above the cutoff."""
    info = read_image_info(make_image("a.fits", 150.0, -30.0, size=400))
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.9),
        weight=WeightOptions(image_weight="none"),
    )
    frame = build_frame([info], options.frame)
    images = InputImages(
        info=info, planes=(info.path,), stokes=(0,), weights=(PlaneWeight(),)
    )
    (prepared,) = prepare_inputs([images], frame, options)
    footprint = input_bounding_box(frame, info)
    assert prepared.bounding_box is not None
    assert footprint is not None
    assert (
        prepared.bounding_box[1] - prepared.bounding_box[0]
        < footprint[1] - footprint[0]
    )


def test_beam_box_holds_the_whole_cutoff_circle(tmp_path: Path) -> None:
    """An input's region holds all of its beam above the cutoff, wherever the frame is turned against the sky."""
    frame_wcs = new_wcs((0.0, -90.0), 2 / 3600, "SIN")
    frame_wcs.wcs.crpix = [10000, 10000]
    frame_wcs.wcs.set()
    frame = Frame(wcs=frame_wcs, shape=(20000, 20000))
    # Near the pole the frame turns by the RA, here halfway between the points sampled on the circle
    centre = (47.8, -87.0)
    input_wcs = new_wcs(centre, 2 / 3600, "SIN")
    input_wcs.wcs.crpix = [5000, 5000]
    input_wcs.wcs.set()
    info = ImageInfo(
        path=tmp_path / "beam00.fits",
        shape=(10000, 10000),
        wcs=input_wcs,
        header=fits.Header(),
        frequency_hz=888e6,
        beam=None,
        beam_number=0,
        beam_centre=centre,
    )
    model = BeamModel(options=PrimaryBeamOptions(), frequencies_hz=(888e6,))
    radius = gaussian_beam(888e6).support_radius(0.01)
    assert radius is not None

    box = contributing_box(frame, info, model, cutoff=0.01, margin=0)
    assert box is not None
    angles = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    circle = offset_vectors(
        centre, np.sin(radius) * np.sin(angles), np.sin(radius) * np.cos(angles)
    )
    x, y = frame.wcs.world_to_pixel_values(*ra_dec(circle))
    assert box[2] <= np.min(x)
    assert np.max(x) <= box[3]
    assert box[0] <= np.min(y)
    assert np.max(y) <= box[1]


def test_noise_image_is_the_noise(
    make_image: ImageFactory, sky_model: SkyModel, tmp_path: Path
) -> None:
    """With inputs weighted by the inverse of their variance, the noise image is the standard deviation of the mosaic about the sky."""
    images = [
        make_image(
            f"beam{index:02d}.fits", 150.0 + 0.3 * index, -30.0, noise=sigma, seed=index
        )
        for index, sigma in enumerate((0.5, 1.0, 0.7))
    ]
    # Nearest pixels keep the noise of each pixel independent, as an image's is over a beam
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.2),
        weight=WeightOptions(image_weight="mad", noise_image=True),
        resample=ResampleOptions(kernel="nearest"),
    )
    names = mosaic(images, tmp_path / "noisy", options)
    assert names.noises == (tmp_path / "noisy.i.noise.fits",)
    image, header = read_mosaic(names.image)
    weight, _ = read_mosaic(names.weight)
    noise, noise_header = read_mosaic(names.noises[0])
    assert noise_header["BUNIT"] == "Jy/beam"
    assert np.allclose(noise, 1 / np.sqrt(weight), equal_nan=True)

    deviation = (image - sky_on_frame(sky_model, header)) / noise
    good = np.isfinite(deviation)
    assert np.sum(good) > 20000
    assert np.isclose(np.std(deviation[good]), 1, atol=0.05)


def test_relative_noise_warns(
    make_image: ImageFactory, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image = make_image("a.fits", 150.0, -30.0, size=60)
    options = MosaicOptions(weight=WeightOptions(image_weight="none", noise_image=True))
    mosaic([image], tmp_path / "relative", options)
    assert "only relative" in caplog.text


def test_weights_log_must_cover_the_image(
    make_image: ImageFactory, tmp_path: Path
) -> None:
    image = make_image("a.fits", 150.0, -30.0, size=40)
    log = tmp_path / "a.weights.txt"
    log.write_text("#Channel Weight\n0 1.0\n1 2.0\n")
    with pytest.raises(ValueError, match="2 channels"):
        mosaic([image], tmp_path / "logged", weights=[log])

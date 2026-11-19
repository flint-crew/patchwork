from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml
from astropy.io import fits
from distributed import Client, LocalCluster

from patchwork.io import BeamShape
from patchwork.mosaic import mosaic
from patchwork.options import (
    MosaicOptions,
    PolarisationOptions,
    PrimaryBeamOptions,
    ResampleOptions,
    RunOptions,
    WeightOptions,
)
from patchwork.pipeline.cluster import get_cluster, get_cluster_spec
from patchwork.pipeline.flow import get_parser, run_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .conftest import ImageFactory
from .test_polarisation import write_polarised_beam


@pytest.fixture
def client() -> Iterator[Client]:
    with (
        LocalCluster(
            n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None
        ) as cluster,
        Client(cluster) as dask_client,
    ):
        yield dask_client


def test_pipeline_matches_serial(
    make_image: ImageFactory, tmp_path: Path, client: Client
) -> None:
    images = [
        make_image("a.beam00.fits", 150.0, -30.0, noise=1e-3, seed=1),
        make_image("b.beam01.fits", 150.5, -29.9, noise=2e-3, seed=2),
    ]
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.1),
        resample=ResampleOptions(tile_size=100),
    )
    serial = mosaic(images, tmp_path / "serial", options)
    parallel = run_pipeline(
        images, tmp_path / "parallel", client=client, options=options
    )

    for serial_path, parallel_path in zip(
        (*serial.images, serial.weight),
        (*parallel.images, parallel.weight),
        strict=True,
    ):
        assert np.array_equal(
            fits.getdata(serial_path), fits.getdata(parallel_path), equal_nan=True
        )


def test_pipeline_convolves(
    make_image: ImageFactory, tmp_path: Path, client: Client
) -> None:
    images = [
        make_image("a.fits", 150.0, -30.0, size=80),
        make_image("b.fits", 150.2, -30.0, size=80),
    ]
    with fits.open(images[1], mode="update") as hdul:
        hdul[0].header["BMAJ"] = 0.012

    names = run_pipeline(
        images,
        tmp_path / "convolved",
        client=client,
        options=MosaicOptions(weight=WeightOptions(image_weight="none")),
        run_options=RunOptions(convolve="natural", keep_convolved=True),
    )
    beam = BeamShape.from_header(fits.getheader(names.image))
    assert beam is not None
    assert beam.bmaj_arcsec >= 0.012 * 3600
    assert (tmp_path / "a.conv.fits").exists()


def test_cluster_from_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "cluster.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "cluster_class": "distributed.LocalCluster",
                "cluster_kwargs": {
                    "n_workers": 1,
                    "processes": False,
                    "dashboard_address": None,
                },
            }
        )
    )
    assert get_cluster_spec(spec_path)["cluster_class"] == "distributed.LocalCluster"
    with get_cluster(spec_path) as cluster:
        assert len(cluster.workers) == 1

    spec_path.write_text(yaml.safe_dump({"cluster_kwargs": {}}))
    with pytest.raises(ValueError, match="cluster_class"):
        get_cluster_spec(spec_path)


def test_pipeline_parser(tmp_path: Path) -> None:
    args = get_parser().parse_args(
        [
            "a.fits",
            "b.fits",
            "--convolve",
            "natural",
            "--n-workers",
            "3",
            "--target-beam",
            "30",
            "30",
            "0",
            "--output-base",
            str(tmp_path / "x"),
        ]
    )
    assert args.convolve == "natural"
    assert args.n_workers == 3
    assert args.target_beam == ["30", "30", "0"]


def test_pipeline_polarisation(tmp_path: Path, client: Client) -> None:
    beams = [
        write_polarised_beam(tmp_path, f"beam{index:02d}", centre, None)
        for index, centre in enumerate([(0.0, -87.5), (120.0, -87.5)])
    ]
    q_images, u_images = [beam[0] for beam in beams], [beam[1] for beam in beams]
    # The images have no noise to weight them by
    options = MosaicOptions(
        primary_beam=PrimaryBeamOptions(pb_cutoff=0.2),
        weight=WeightOptions(image_weight="none"),
        resample=ResampleOptions(tile_size=128),
        polarisation=PolarisationOptions(input_frame="crval"),
    )
    serial = mosaic([*q_images, *u_images], tmp_path / "serial", options)
    parallel = run_pipeline(
        [*q_images, *u_images], tmp_path / "parallel", client=client, options=options
    )
    for serial_path, parallel_path in zip(
        (*serial.images, serial.weight),
        (*parallel.images, parallel.weight),
        strict=True,
    ):
        assert np.array_equal(
            fits.getdata(serial_path), fits.getdata(parallel_path), equal_nan=True
        )


def test_pipeline_retries_sending_the_plan(
    make_image: ImageFactory,
    tmp_path: Path,
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scatter that fails as a worker leaves is retried, and the mosaic is made."""
    images = [
        make_image("a.beam00.fits", 150.0, -30.0, noise=1e-3, seed=1),
        make_image("b.beam01.fits", 150.5, -29.9, noise=2e-3, seed=2),
    ]
    monkeypatch.setattr("patchwork.pipeline.flow.time.sleep", lambda _: None)
    real_scatter = client.scatter
    failures = iter([KeyError("tcp://10.0.0.1:1"), OSError("stream closed")])

    def flaky_scatter(*args: object, **kwargs: object) -> object:
        failure = next(failures, None)
        if failure is not None:
            raise failure
        return real_scatter(*args, **kwargs)

    monkeypatch.setattr(client, "scatter", flaky_scatter)
    options = MosaicOptions(resample=ResampleOptions(tile_size=100))
    serial = mosaic(images, tmp_path / "serial", options)
    with caplog.at_level("WARNING", logger="patchwork"):
        parallel = run_pipeline(
            images, tmp_path / "parallel", client=client, options=options
        )
    assert caplog.text.count("retrying") == 2
    assert np.array_equal(
        fits.getdata(serial.images[0]), fits.getdata(parallel.images[0]), equal_nan=True
    )

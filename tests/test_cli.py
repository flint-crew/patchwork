from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from astropy.io import fits

from patchwork.cli import cli, get_parser

if TYPE_CHECKING:
    from pathlib import Path

    from .conftest import ImageFactory


def test_parser_modes() -> None:
    parser = get_parser()
    assert parser.parse_args(["mosaic", "a.fits"]).mode == "mosaic"
    assert parser.parse_args(["convolve", "a.fits"]).mode == "convolve"
    assert parser.parse_args(["frame", "a.fits"]).mode == "frame"
    assert parser.parse_args(["rotate", "a.fits"]).mode == "rotate"


@pytest.mark.parametrize("frame_type", ["reference", "fit"])
def test_frame_command(
    make_image: ImageFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_type: str,
) -> None:
    images = [
        make_image("a.fits", 150.0, -30.0, size=50),
        make_image("b.fits", 150.3, -30.0, size=50),
    ]
    output = tmp_path / "frame.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "patchwork",
            "frame",
            *map(str, images),
            "--output",
            str(output),
            "--frame-type",
            frame_type,
        ],
    )
    cli()
    header = fits.Header.fromtextfile(output)
    assert header["NAXIS1"] > 50


def test_unknown_mode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sys, "argv", ["patchwork"])
    cli()
    assert "not known" in caplog.text

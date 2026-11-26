from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from radio_beam import Beams
from radio_beam.utils import BeamError

from patchwork.convolve import (
    beam_covariance,
    beam_from_covariance,
    cli,
    common_beam,
    convolve_fits,
    convolve_images,
    convolve_to_beam,
    convolved_name,
    get_parser,
)
from patchwork.io import BeamShape, celestial_wcs
from patchwork.options import ConvolveOptions

from .conftest import image_header

if TYPE_CHECKING:
    from pathlib import Path

    from patchwork.convolve import ConvolveBackend

BACKENDS: tuple[ConvolveBackend, ...] = ("scipy", "rust")
RETRIED_BEAMS: list[BeamShape | None] = [BeamShape(12, 8, 30), BeamShape(10, 10, 0)]
has_convolve_rs = importlib.util.find_spec("convolve_rs") is not None
needs_convolve_rs = pytest.mark.skipif(not has_convolve_rs, reason="needs convolve-rs")


@pytest.fixture(params=["scipy", pytest.param("rust", marks=needs_convolve_rs)])
def backend(request: pytest.FixtureRequest) -> ConvolveBackend:
    """Each convolution backend."""
    return cast("ConvolveBackend", request.param)


def sky_wcs(cdelt1_sign: int = -1, rotation_deg: float = 0.0) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.crval = [150.0, -30.0]
    wcs.wcs.cdelt = [cdelt1_sign * 2 / 3600, 2 / 3600]
    wcs.wcs.crpix = [129, 129]
    angle = np.radians(rotation_deg)
    wcs.wcs.pc = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    wcs.wcs.set()
    return wcs


def point_sources(wcs: WCS, shape: tuple[int, int], beam: BeamShape) -> np.ndarray:
    """Point sources in Jy/beam drawn with a beam, from its orientation on the sky."""
    y, x = np.mgrid[: shape[0], : shape[1]]
    ra, dec = wcs.pixel_to_world_values(x, y)
    inverse = np.linalg.inv(beam_covariance(beam))
    image = np.zeros(shape)
    for source_ra, source_dec, flux in ((150.0, -30.0, 1.0), (150.02, -29.98, 0.5)):
        east = (ra - source_ra) * np.cos(np.radians(source_dec))
        north = dec - source_dec
        quadratic = (
            inverse[0, 0] * east**2
            + 2 * inverse[0, 1] * east * north
            + inverse[1, 1] * north**2
        )
        image += flux * np.exp(-0.5 * quadratic)
    return image


@pytest.mark.parametrize(
    ("cdelt1_sign", "rotation_deg"), [(-1, 0.0), (1, 0.0), (-1, 25.0)]
)
def test_convolve_matches_drawn_beam(
    cdelt1_sign: int, rotation_deg: float, backend: ConvolveBackend
) -> None:
    wcs = sky_wcs(cdelt1_sign, rotation_deg)
    old, new = BeamShape(12, 8, 30), BeamShape(25, 15, -50)
    image = point_sources(wcs, (256, 256), old).astype(np.float32)
    convolved = convolve_to_beam(
        image, old, new, wcs.pixel_scale_matrix, backend=backend
    )
    assert convolved.dtype == np.float32
    # Corners, with most of their kernel beyond the image, are blanked
    finite = np.isfinite(convolved)
    assert np.isnan(convolved[0, 0])
    assert np.all(finite[64:192, 64:192])
    expected = point_sources(wcs, (256, 256), new)
    assert np.max(np.abs(convolved[finite] - expected[finite])) < 1e-4


def test_convolve_identity_and_cutoff(backend: ConvolveBackend) -> None:
    wcs = sky_wcs()
    beam = BeamShape(12, 8, 30)
    image = point_sources(wcs, (64, 64), beam)
    assert np.array_equal(
        convolve_to_beam(image, beam, beam, wcs.pixel_scale_matrix, backend=backend),
        image,
    )
    assert np.all(
        np.isnan(
            convolve_to_beam(
                image,
                beam,
                BeamShape(20, 20, 0),
                wcs.pixel_scale_matrix,
                cutoff_arcsec=10,
                backend=backend,
            )
        )
    )
    with pytest.raises(ValueError, match="does not enclose"):
        convolve_to_beam(
            image,
            BeamShape(20, 20, 0),
            beam,
            wcs.pixel_scale_matrix,
            backend=backend,
        )


def test_convolve_blanks_around_nans(backend: ConvolveBackend) -> None:
    wcs = sky_wcs()
    image = point_sources(wcs, (128, 128), BeamShape(12, 8, 30))
    image[:, :20] = np.nan
    convolved = convolve_to_beam(
        image,
        BeamShape(12, 8, 30),
        BeamShape(30, 30, 0),
        wcs.pixel_scale_matrix,
        backend=backend,
    )
    assert np.all(np.isnan(convolved[:, :20]))
    assert np.all(np.isfinite(convolved[:, 40:100]))


def test_common_beam(backend: ConvolveBackend) -> None:
    beams = [
        BeamShape(12, 8, 30),
        BeamShape(10, 10, 0),
        BeamShape(15, 6, -60),
        None,
        BeamShape(0, 0, 0),
        BeamShape(90, 90, 0),
    ]
    found = common_beam(beams, cutoff_arcsec=60, backend=backend)
    assert found.bmaj_arcsec < 20
    for beam in beams[:3]:
        assert beam is not None
        difference = beam_covariance(found) - beam_covariance(beam)
        assert np.min(np.linalg.eigvalsh(difference)) > -1e-12
    assert np.isclose(found.bmaj_arcsec * 10, round(found.bmaj_arcsec * 10))

    circular = common_beam(beams, cutoff_arcsec=60, circularise=True, backend=backend)
    assert circular.bmin_arcsec == circular.bmaj_arcsec
    with pytest.raises(ValueError, match="usable"):
        common_beam([None, BeamShape(0, 0, 0)], backend=backend)


def test_common_beam_retries_with_a_finer_tolerance(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A search that fails is retried once, at a tenth of the tolerance."""
    tolerances: list[float] = []
    found = Beams.common_beam

    def first_fails(self: Beams, tolerance: float, **kwargs: Any) -> Any:
        tolerances.append(tolerance)
        if len(tolerances) == 1:
            raise BeamError
        return found(self, tolerance=tolerance, **kwargs)

    monkeypatch.setattr(Beams, "common_beam", first_fails)
    assert common_beam(RETRIED_BEAMS, tolerance=1e-4).bmaj_arcsec >= 12
    assert tolerances == pytest.approx([1e-4, 1e-5])
    assert "Retrying the common beam search" in caplog.text


@needs_convolve_rs
def test_rust_common_beam_retries_with_a_finer_tolerance(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """convolve-rs is retried the same way."""
    cr = pytest.importorskip("convolve_rs")
    tolerances: list[float] = []
    found = cr.common_beam

    def first_fails(beams: list[Any], tolerance: float, *args: Any) -> Any:
        tolerances.append(tolerance)
        if len(tolerances) == 1:
            raise ValueError
        return found(beams, tolerance, *args)

    monkeypatch.setattr(cr, "common_beam", first_fails)
    beam = common_beam(RETRIED_BEAMS, tolerance=1e-4, backend="rust")
    assert beam.bmaj_arcsec >= 12
    assert tolerances == pytest.approx([1e-4, 1e-5])
    assert "Retrying the common beam search" in caplog.text


def test_common_beam_says_what_it_left_out(caplog: pytest.LogCaptureFixture) -> None:
    assert str(BeamShape(11.5, 7.8, -19.24)) == '11.5" x 7.8", PA -19.24 deg'
    beams = [BeamShape(12, 8, 30), None, BeamShape(0, 0, 0), BeamShape(90, 90, 0)]
    found = common_beam(beams, cutoff_arcsec=60, name="channel 3")
    assert (
        f'Common beam of channel 3 is {found}, from 1 of 4 beams (2 with no beam, 1 larger than 60")'
        in caplog.text
    )


def write_beam_image(path: Path, beam: BeamShape) -> Path:
    header = image_header(150.0, -30.0, size=128, cdelt=2 / 3600)
    beam.to_header(header)
    image = point_sources(celestial_wcs(header), (128, 128), beam)
    fits.writeto(path, image[None, None].astype(np.float32), header)
    return path


def test_convolve_fits(tmp_path: Path, backend: ConvolveBackend) -> None:
    path = write_beam_image(tmp_path / "a.beam00.fits", BeamShape(12, 8, 30))
    new = BeamShape(25, 15, -50)
    output = convolve_fits(path, new, backend=backend)
    assert output == convolved_name(path) == tmp_path / "a.beam00.conv.fits"
    header = fits.getheader(output)
    assert BeamShape.from_header(header) == pytest.approx(new)
    assert fits.getdata(output).shape == (1, 1, 128, 128)

    blanked = convolve_fits(
        path,
        new,
        output_path=tmp_path / "blank.fits",
        cutoff_arcsec=10,
        backend=backend,
    )
    assert BeamShape.from_header(fits.getheader(blanked)) == (0, 0, 0)
    assert np.all(np.isnan(fits.getdata(blanked)))


def test_convolve_images_and_cli(tmp_path: Path, backend: ConvolveBackend) -> None:
    images = [
        write_beam_image(tmp_path / "a.fits", BeamShape(12, 8, 30)),
        write_beam_image(tmp_path / "b.fits", BeamShape(10, 10, 0)),
    ]
    beam, outputs = convolve_images(images, ConvolveOptions(convolve_backend=backend))
    assert all(
        BeamShape.from_header(fits.getheader(output)) == pytest.approx(beam)
        for output in outputs
    )

    _, targeted = convolve_images(
        images,
        ConvolveOptions(
            target_beam=(30, 30, 0), suffix="big", convolve_backend=backend
        ),
    )
    assert targeted[0].name == "a.big.fits"

    cli(get_parser().parse_args([*map(str, images), "--overwrite", "--circularise"]))
    assert BeamShape.from_header(fits.getheader(outputs[0])).bpa_deg == 0  # type: ignore[union-attr]


def test_convolution_does_not_wrap_around(backend: ConvolveBackend) -> None:
    """Flux spread off one edge does not come back on the opposite edge, even for an image whose size needs no padding to transform quickly."""
    pixel_scale = np.diag([-2 / 3600, 2 / 3600])
    old, new = BeamShape(10, 10, 0), BeamShape(60, 60, 0)
    image = np.zeros((256, 256), dtype=np.float32)
    image[128, 0] = 1.0
    convolved = convolve_to_beam(image, old, new, pixel_scale, backend=backend)
    # The same source well inside a wider image, where nothing can wrap
    wide = np.zeros((256, 768), dtype=np.float32)
    wide[128, 256] = 1.0
    expected = convolve_to_beam(wide, old, new, pixel_scale, backend=backend)[
        :, 256:512
    ]
    # Away from the edges, where the kernel lies wholly on the image
    assert np.allclose(convolved[80:176, 80:176], expected[80:176, 80:176], atol=1e-5)
    assert abs(convolved[128, -1]) < 1e-5


@pytest.mark.parametrize("new", [BeamShape(40, 30, 20), BeamShape(10.5, 10.3, 0)])
def test_flat_image_stays_flat(new: BeamShape, backend: ConvolveBackend) -> None:
    """Pixels near the edges and NaNs are divided by the share of their kernel on
    data, so a flat image stays flat, even for a kernel under a pixel wide."""
    pixel_scale = np.diag([-2 / 3600, 2 / 3600])
    old = BeamShape(10, 10, 0)
    image = np.ones((100, 120), dtype=np.float32)
    image[30:50, 40:70] = np.nan
    convolved = convolve_to_beam(image, old, new, pixel_scale, backend=backend)
    area_ratio = (new.bmaj_arcsec * new.bmin_arcsec) / (
        old.bmaj_arcsec * old.bmin_arcsec
    )
    finite = np.isfinite(convolved)
    assert np.all(np.isnan(convolved[35:45, 50:60]))
    assert np.mean(finite) > 0.7
    assert np.allclose(convolved[finite], area_ratio, rtol=1e-5)


def test_convolved_name_never_names_the_image() -> None:
    assert convolved_name("a.beam00.fits").name == "a.beam00.conv.fits"
    assert convolved_name("a.beam00.FITS").name == "a.beam00.conv.FITS"
    assert convolved_name("a.beam00.fit").name == "a.beam00.fit.conv.fits"
    assert (
        convolved_name("cutouts.fits_v2.beam00.fits", "smooth").name
        == "cutouts.fits_v2.beam00.smooth.fits"
    )


def test_convolve_refuses_to_write_over_an_input(tmp_path: Path) -> None:
    """Images matched by a pattern can include ones convolved before, which are never written over."""
    convolve_images([write_beam_image(tmp_path / "a.fits", BeamShape(12, 8, 30))])
    again = sorted(tmp_path.glob("*.fits"))
    assert [path.name for path in again] == ["a.conv.fits", "a.fits"]
    with pytest.raises(FileExistsError, match="also an input"):
        convolve_images(again, overwrite=True)


@needs_convolve_rs
@pytest.mark.parametrize(
    ("cdelt1_sign", "rotation_deg"), [(-1, 0.0), (1, 0.0), (-1, 25.0)]
)
def test_rust_matches_scipy(cdelt1_sign: int, rotation_deg: float) -> None:
    """The two backends give the same image and NaN mask, edges included, on
    any grid, and the same common beam to within its rounding."""
    wcs = sky_wcs(cdelt1_sign, rotation_deg)
    old, new = BeamShape(12, 8, 30), BeamShape(25, 15, -50)
    image = point_sources(wcs, (200, 230), old).astype(np.float32)
    image += np.random.default_rng(0).normal(0, 1e-3, image.shape).astype(np.float32)
    image[40:90, 20:70] = np.nan
    image[150, 180] = np.nan
    scipy, rust = (
        convolve_to_beam(image, old, new, wcs.pixel_scale_matrix, backend=b)
        for b in BACKENDS
    )
    assert np.array_equal(np.isnan(scipy), np.isnan(rust))
    finite = np.isfinite(scipy)
    assert np.max(np.abs(scipy[finite] - rust[finite])) < 1e-5

    beams: list[BeamShape | None] = [
        BeamShape(12, 8, 30),
        BeamShape(10, 10, 0),
        BeamShape(15, 6, -60),
    ]
    scipy_beam, rust_beam = (common_beam(beams, backend=b) for b in BACKENDS)
    assert rust_beam.bmaj_arcsec == pytest.approx(scipy_beam.bmaj_arcsec, abs=0.11)
    assert rust_beam.bmin_arcsec == pytest.approx(scipy_beam.bmin_arcsec, abs=0.11)
    pa_difference = (rust_beam.bpa_deg - scipy_beam.bpa_deg + 90) % 180 - 90
    assert abs(pa_difference) < 0.5


@needs_convolve_rs
def test_rust_uses_the_clipped_kernel() -> None:
    """A target a hair inside the old beam, as a rounded common beam can be,
    is convolved with the same clipped kernel by both backends."""
    wcs = sky_wcs()
    old = BeamShape(12, 8, 30)
    new = BeamShape(12, 8 - 1e-4, 30)
    image = point_sources(wcs, (64, 64), old)
    scipy, rust = (
        convolve_to_beam(image, old, new, wcs.pixel_scale_matrix, backend=b)
        for b in BACKENDS
    )
    assert np.allclose(scipy, rust, atol=1e-6)


def test_beam_from_covariance_round_trips() -> None:
    for beam in (BeamShape(12, 8, 30), BeamShape(20, 5, -75), BeamShape(9, 9, 0)):
        again = beam_from_covariance(beam_covariance(beam))
        assert again.bmaj_arcsec == pytest.approx(beam.bmaj_arcsec)
        assert again.bmin_arcsec == pytest.approx(beam.bmin_arcsec)
        if beam.bmaj_arcsec != beam.bmin_arcsec:
            assert (again.bpa_deg - beam.bpa_deg + 90) % 180 - 90 == pytest.approx(0)


def test_rust_backend_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "convolve_rs", None)
    wcs = sky_wcs()
    image = np.zeros((32, 32))
    with pytest.raises(ImportError, match=r"askap-patchwork\[rust\]"):
        convolve_to_beam(
            image,
            BeamShape(10, 10, 0),
            BeamShape(20, 20, 0),
            wcs.pixel_scale_matrix,
            backend="rust",
        )

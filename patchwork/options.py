"""Options that control each stage of the mosaicking"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

from capn_crunch import BaseOptions, add_options_to_parser, create_options_from_parser

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


class FrameOptions(BaseOptions):
    """How the output frame of the mosaic is constructed. The ``reference`` frame
    reuses the pixel grid of one input, while the ``fit`` frame builds a new grid
    from all inputs. Either way the frame is sized to cover every input."""

    frame_type: Literal["reference", "fit"] = "reference"
    """Reuse the grid of a reference input (``reference``), or fit a new grid to all inputs (``fit``)"""
    reference_index: int | None = None
    """Index of the input whose grid a ``reference`` frame reuses. If None the central input is used"""
    centre: tuple[float, float] | None = None
    """The (RA, Dec) in degrees of the frame centre. If None it is derived from the inputs"""
    centre_type: Literal["all", "most"] = "all"
    """How a ``fit`` frame finds its centre: the middle of all inputs (``all``), or of the region most inputs cover (``most``)"""
    pixel_scale_type: Literal["median", "min", "max"] = "median"
    """Which of the input pixel scales a ``fit`` frame takes"""
    pixel_scale_arcsec: float | None = None
    """The pixel scale of a ``fit`` frame in arcseconds, overriding ``pixel_scale_type``"""
    projection: str = "SIN"
    """The projection code of a ``fit`` frame, or of a ``reference`` frame given a new ``centre``"""
    image_size: tuple[int, int] | None = None
    """The (nx, ny) size of a ``fit`` frame in pixels. If None it is sized to cover all inputs"""
    full_frame: bool = False
    """Cover the whole images of the inputs. Otherwise the frame is trimmed to where they contribute, within ``pb_cutoff`` of their primary beams, as no pixel beyond that has a value. A frame of a given ``image_size`` is never trimmed"""


class PrimaryBeamOptions(BaseOptions):
    """The primary beam model that weights each input"""

    pb_model: Literal["gaussian", "holography", "none"] = "gaussian"
    """The primary beam model. With ``none`` every input has a flat response"""
    aperture_m: float = 12.0
    """Dish diameter in metres of the Gaussian beam"""
    fwhm_scaling: float = 1.09
    """The Gaussian beam FWHM, in units of wavelength over the dish diameter"""
    holography_cube: Path | None = None
    """Holography cube of beam responses, with axes (beam, stokes, freq, dec, ra). Needed by the ``holography`` model"""
    holography_alpha_rad: float = 0.0
    """Rotation in radians of the holography beams from the frame the cube was measured in"""
    holography_scale: Literal["centre", "stored"] = "centre"
    """Scale each holography beam to 1 at its centre, where calibration sets the gain (``centre``), or use it as the cube stores it, with ``pb_cutoff`` measured from the beam's peak (``stored``)"""
    pb_cutoff: float = 0.01
    """Each input only contributes where its primary beam is at least this response, whatever its noise. The mosaic is not blanked by its weight. With 0 or less an input contributes wherever its beam responds at all"""


class WeightOptions(BaseOptions):
    """How each input is weighted. ``weight_type`` sets where the
    weights come from and ``weight_state`` whether the inputs are already corrected
    for the primary beam."""

    weight_type: Literal["primary_beam", "weight_image", "combined"] = "combined"
    """Weight by the primary beam squared, by a weight (an image or a single value), or by both"""
    weight_state: Literal["corrected", "inherent", "weighted"] = "inherent"
    """Whether inputs are corrected for the primary beam (``corrected``), still carry it (``inherent``), or are already weighted (``weighted``)"""
    image_weight: Literal["mad", "std", "none"] = "mad"
    """Without weight images, weight each input by its inverse variance from this noise statistic. With ``none`` every input weighs 1"""
    weight_stride: int = 8
    """Use every n'th row when measuring the noise of an input, which reads only those rows"""
    weight_map_type: Literal["weight", "variance", "rms"] = "weight"
    """What the values of weight images are"""
    noise_image: bool = False
    """Also write the noise of the mosaic, the inverse square root of its weight. This is the standard deviation of each pixel when every input is weighted by the inverse of its variance, as ``image_weight`` and ``variance`` or ``rms`` weight maps weight them, and otherwise only relative"""


class ResampleOptions(BaseOptions):
    """How inputs are resampled onto the output frame"""

    kernel: Literal[
        "nearest", "linear", "cubic", "lanczos2", "lanczos3", "lanczos4"
    ] = "cubic"
    """The interpolation kernel"""
    tile_size: int = 512
    """Size in pixels of the square tiles the output frame is worked in. Each is a task, whose memory grows with its area"""
    coordinate_step: int = 16
    """Spacing in pixels of the grid on which coordinates are transformed exactly, and between which they are interpolated. Interpolated positions are checked to be within 0.001 pixels, and every pixel is transformed where they are not. 1 transforms every pixel"""
    backend: Literal["auto", "numpy", "numba", "fused"] = "auto"
    """Interpolate with numpy or numba, or make each tile in one compiled pass with numba (``fused``), which can use several threads (``tile_threads``). ``auto`` uses numba when it is installed"""
    kernel_threads: int = 0
    """Threads of ``tile_threads`` that the ``fused`` kernels run on; the rest read the inputs. 0 gives the kernels 5 of every 8 threads when there are at least 4, which balanced the two on a 16-core node, and has every thread do both with fewer. Setting it to ``tile_threads`` or more has every thread do both, so the readers compete with the kernels for CPUs, and a parallel kernel waits for its slowest thread"""
    tile_threads: int = 1
    """Threads each task uses: the ``fused`` backend makes each tile with them, and the noise of each image is measured with them by every backend. 0 uses every CPU the process may run on, shared between the tasks it runs at once: between ``max_workers``, or the threads of its dask worker. Leave it at 1 where each core has a worker of its own, and give a number where several worker processes share the CPUs of one job"""


class PolarisationOptions(BaseOptions):
    """How linear polarisation is mosaicked. Stokes Q and U are rotated from the
    frame of each input into the frame of the mosaic, which matters most towards
    the celestial poles. With a holography primary beam the leakage of Stokes I
    into Q, U and V can be removed."""

    input_frame: Literal["reference", "crval", "north"] = "reference"
    """What the Q and U of each input are referenced to. ``reference``: north at ``frame_centre``, or at the position the input's POLFRAME keywords give; an input of Q and U with neither is refused. ``crval``: north at each input's own reference position, as an imager references a single pointing. ``north``: north at every pixel. An input whose POLFRAME keywords say otherwise is refused"""
    frame_centre: tuple[float, float] | None = None
    """The (RA, Dec) in degrees that the Q and U of every input are referenced to north at, such as the boresight of an ASKAP observation, with ``input_frame="reference"``. It also orients holography beams"""
    output_frame: Literal["north", "centre"] = "north"
    """Reference the Q and U of the mosaic to north at every pixel (``north``, the IAU convention), or to north at the centre of the frame (``centre``), which stays smooth across a pole"""
    remove_leakage: bool = False
    """Subtract the leakage of Stokes I into Q, U and V given by the holography cube, which needs a Stokes I image of each input"""
    keep_centre_leakage: bool = False
    """Keep the leakage at the beam centre. Otherwise it is removed before the leakage is subtracted, as calibration has already removed it there"""


class CombineOptions(BaseOptions):
    """How resampled inputs are combined"""

    combine_type: Literal["weighted", "average", "sum"] = "weighted"
    """Combine as a weighted mean, a plain mean or a sum"""


class RunOptions(BaseOptions):
    """How a mosaic is made: which channels of cubes, how many each task makes,
    whether the inputs are first convolved to a common beam, and what the
    mosaic is written as"""

    channels: tuple[int, int] | None = None
    """The (start, stop) channels to mosaic, counted from 0 and stopping before ``stop``. If None every channel is"""
    channel_chunk: int = 16
    """Number of channels each task makes. More channels share more of the work (the positions and primary beams of a tile), and take more memory"""
    tasks_in_flight: int | None = None
    """Most tiles made at once. The process writing a FITS mosaic holds each finished tile until it is written, so this bounds its memory: if None, twice the number of tasks the executor runs at once. Where the workers write the mosaic (``fits_writer`` ``workers``, or zarr) nothing is held, and None queues every tile, so an adaptive dask cluster sees all the work and grows to its maximum"""
    convolve: Literal["none", "natural", "total"] = "none"
    """Convolve the inputs to a common beam first: each channel to the smallest beam that encloses that channel of every input (``natural``), or every channel to one beam (``total``). For images, which have one channel, the two are the same"""
    output_format: Literal["fits", "zarr"] = "fits"
    """Write the mosaic as FITS, or as zarr stores (one (channel, y, x) array per image, as rm-lite reads them)"""
    fits_writer: Literal["driver", "workers"] = "driver"
    """Which processes write FITS outputs: the one driving the mosaic, as tiles arrive (``driver``), or the tasks themselves, each into its own strip of whole rows (``workers``). Workers need a file system that keeps one file consistent while several machines write to it, such as Lustre or GPFS, and not NFS"""
    zarr_chunk: int = 256
    """Size in pixels of the square chunks of zarr output. Each is made over every channel by one task"""
    keep_convolved: bool = False
    """Keep the convolved inputs, which are otherwise removed once the mosaic is made"""


class ConvolveOptions(BaseOptions):
    """How images are convolved to a common restoring beam"""

    target_beam: tuple[float, float, float] | None = None
    """The (major, minor, position angle) of the beam to convolve to, in arcseconds and degrees. If None the smallest beam that encloses all inputs is used"""
    beam_cutoff_arcsec: float | None = None
    """Images whose beam major axis is larger than this are blanked and left out of the common beam"""
    circularise: bool = False
    """Make the common beam circular, with its minor axis set to its major axis"""
    tolerance: float = 1e-4
    """Tolerance of the common beam search"""
    nsamps: int = 200
    """Number of samples of the common beam search"""
    epsilon: float = 5e-4
    """Epsilon of the common beam search"""
    suffix: str = "conv"
    """Suffix inserted before ``.fits`` in the names of convolved images"""
    convolve_backend: Literal["scipy", "rust"] = "scipy"
    """Convolve with scipy, or with convolve-rs, which is installed with ``pip install askap-patchwork[rust]``. Both give the same images and common beams"""


class RotateOptions(BaseOptions):
    """How the Q and U of images or cubes are rotated from the frame they were
    imaged in to north at each pixel, the IAU convention, as ``patchwork mosaic``
    rotates them in flight"""

    input_frame: Literal["reference", "crval"] = "reference"
    """What the Q and U of each input are referenced to, as for the mosaic. ``reference``: north at ``frame_centre``, or at the position the input's POLFRAME keywords give; an input with neither is refused. ``crval``: north at each input's own reference position, as an imager references a single pointing. An input whose keywords say otherwise is refused, as rotating it from another frame would turn it twice"""
    frame_centre: tuple[float, float] | None = None
    """The (RA, Dec) in degrees that the Q and U of every input are referenced to north at, such as the boresight of an ASKAP observation, with ``input_frame="reference"``"""
    suffix: str = "rot"
    """Suffix inserted before ``.fits`` in the names of rotated images"""
    reverse_rotation: bool = False
    """Turn Q and U the opposite way, by minus the angle between the frame of each input and north at each pixel. This is for testing the direction of the rotation, such as where the angles of overlapping fields disagree. The outputs still record that they are referenced to north at each pixel, and their HISTORY that the rotation was reversed"""


class MosaicOptions(NamedTuple):
    """The options of every stage of the mosaicking"""

    frame: FrameOptions = FrameOptions()
    """How the output frame is constructed"""
    primary_beam: PrimaryBeamOptions = PrimaryBeamOptions()
    """The primary beam model"""
    weight: WeightOptions = WeightOptions()
    """How each input is weighted"""
    resample: ResampleOptions = ResampleOptions()
    """How inputs are resampled"""
    combine: CombineOptions = CombineOptions()
    """How resampled inputs are combined"""
    polarisation: PolarisationOptions = PolarisationOptions()
    """How linear polarisation is mosaicked"""


def add_mosaic_options_to_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add the options of every mosaicking stage to a parser."""
    for options_class in MosaicOptions._field_defaults.values():
        add_options_to_parser(parser=parser, options_class=type(options_class))
    return parser


def create_mosaic_options_from_parser(args: Namespace) -> MosaicOptions:
    """Create the options of every mosaicking stage from parsed arguments."""
    return MosaicOptions(
        **{
            name: create_options_from_parser(
                parser_namespace=args, options_class=type(default)
            )
            for name, default in MosaicOptions._field_defaults.items()
        }
    )

# Mosaicking

`patchwork` combines overlapping images into one mosaic, as a weighted sum of
the images. A mosaic is made in four steps:

1. An output pixel grid, the frame, is built to cover the inputs
   (`patchwork.frame`).
2. Each input is interpolated onto the frame (`patchwork.resample`).
3. Each input is weighted by its primary beam and its noise
   (`patchwork.primary_beam`, `patchwork.weights`).
4. The weighted inputs are summed and normalised (`patchwork.combine`).

The frame is divided into square tiles of `tile_size` pixels, and
`patchwork.tile.mosaic_tile` makes each tile separately. A tile reads only the
parts of the inputs it needs, and uses each input only where its primary beam is
at least `pb_cutoff`. Tiles can be made in any order, and each one is final once
it is written. `patchwork.mosaic.mosaic` makes them in turn, or in threads or
processes with `max_workers`, and the [pipeline](pipeline.md) makes them on a
dask cluster. The number of tiles finished is logged every 30 seconds, with an
estimate of the time remaining. [Performance](performance.md) describes the
options that affect speed.

Stokes Q and U are mosaicked together and rotated into the frame of the mosaic,
as described in [polarisation](polarisation.md). This needs to know what the Q
and U of each input are referenced to, so patchwork refuses to mosaic them
unless that is given.

`patchwork mosaic` takes images and cubes in one list. The Stokes parameter of
each is read from its STOKES axis, and the images on one pixel grid are taken to
be the Stokes parameters of one beam. An image is treated as a cube of one
channel, so [cubes](cubes.md) are mosaicked in the same way. Images of the same
Stokes parameter on the same grid, such as two epochs of one pointing, are
separate inputs. Degenerate frequency and Stokes axes, as in ASKAP and wsclean
images, are kept in the mosaic. The outputs are named by the Stokes parameters
they hold: `<base>.i.mosaic.fits` and `<base>.i.weight.fits`, with Q and U
sharing `<base>.qu.weight.fits`.

## The frame

With `frame_type="reference"`, the default, the mosaic uses the pixel grid of
one input. That input is `reference_index` or, if that is not given, the input
whose centre is nearest the mean of all the centres. If `centre` is given, the
grid is instead a north-up grid centred there, with the reference input's pixel
scale.

With `frame_type="fit"`, a new north-up grid is fitted to the inputs. Its pixel
scale is the median, minimum or maximum of the input pixel scales
(`pixel_scale_type`), or `pixel_scale_arcsec`. Its centre is the middle of all
the inputs (`centre_type="all"`), the middle of the region most of them cover
(`centre_type="most"`), or `centre`. Its projection is `projection`.

In both cases the frame is first sized to cover every input, unless `image_size`
sets the size of a fitted frame. It is then trimmed to the region where the
inputs contribute, since pixels outside it would be blank. `full_frame` keeps
the untrimmed frame.

An input contributes only within its own image, and only where its primary beam
is at least `pb_cutoff` in one or more of its channels:

- For a Gaussian beam this region is a circle, set by the lowest-frequency
  channel, where the beam is widest.
- For a holography beam it is the set of cube pixels at or above the cutoff in
  any channel, widened by one pixel for the interpolation, and mapped to the sky
  in the same way as the beam is sampled. At low cutoffs the beam stays above
  the cutoff almost to the edge of the cube, and the region is close to the
  cube's square grid.

The frame is sized from every channel of the inputs, even when `channels`
selects only some of them, so mosaics of different channel ranges of the same
cubes have the same frame. A cube has one frame for all of its channels, and the
beam narrows as the frequency rises, so the higher channels have wider blank
borders. zarr output does not store chunks that are blank in every channel.

## Resampling

The position of each output pixel in each input is found through the coordinates
(output pixel to sky to input pixel), and the input is interpolated there with a
separable kernel:

| `kernel`   | Taps | Notes                                    |
| ---------- | ---- | ---------------------------------------- |
| `nearest`  | 1    |                                          |
| `linear`   | 2    | bilinear                                 |
| `cubic`    | 4    | Keys cubic convolution, `a = -0.5`       |
| `lanczos2` | 4    | Lanczos with `a = 2`, scaled to sum to 1 |
| `lanczos3` | 6    | Lanczos with `a = 3`, scaled to sum to 1 |
| `lanczos4` | 8    | Lanczos with `a = 4`, scaled to sum to 1 |

An output pixel gets no value from an input if any kernel tap with non-zero
weight falls on a blank (NaN) pixel or outside the input. Blank regions
therefore grow by the half-width of the kernel, rather than lowering the values
next to them.

The inputs are in Jy/beam, so the resampled values are not rescaled for the
change in pixel area.

## Weighting

Each input `i` contributes a weighted image `v_i` and a weight `w_i` at every
pixel it covers, and the mosaic is `sum(v_i) / sum(w_i)`. `weight_type` sets
where the weights come from, and `weight_state` whether the inputs are already
corrected for their primary beam `P`. With `W` the weight of an input (one
value, or a weight image):

| `weight_type`  | `corrected` | `inherent`  | `weighted` | weight `w` |
| -------------- | ----------- | ----------- | ---------- | ---------- |
| `primary_beam` | `I P^2`     | `I P`       | `I`        | `P^2`      |
| `combined`     | `I W P^2`   | `I W P`     | `I W`      | `W P^2`    |
| `weight_image` | `I W`       | `I sqrt(W)` | `I`        | `W`        |

The default, `combined` with `inherent` inputs, is what flint uses. Images from
the imager still carry their primary beam, so the mosaic is

```
sum(W_i P_i I_i) / sum(W_i P_i^2)
```

This weights each primary-beam-corrected input by its inverse variance, which
gives the mosaic the lowest variance.

Without weight images, `W` is the inverse variance of each input, measured from
every `weight_stride`th row with the median absolute deviation
(`image_weight="mad"`) or the standard deviation (`"std"`). With
`image_weight="none"`, every input has a weight of 1. Weight images can hold
weights, variances or rms values (`weight_map_type`). A linmos weights log can
be given in place of a weight image: a `.txt` file with a `#Channel Weight`
header and a weight for each channel, as flint writes them.

### Primary beams

- `pb_model="gaussian"`: a circular Gaussian with a FWHM of `fwhm_scaling`
  wavelengths divided by `aperture_m` (1.09 λ / 12 m by default).
- `pb_model="holography"`: one beam of a holography cube (`holography_cube`)
  with axes (beam, stokes, freq, dec, ra). The beam is chosen by the `beamNN` in
  each input's file name, and the usable plane of the cube nearest the input's
  frequency is used. `holography_alpha_rad` rotates the beams;
  `patchwork.primary_beam.pb_rotation_from_pol_axis` calculates it from a
  measurement set's polarisation axis. See
  [holography beams and leakage](holography.md).
- `pb_model="none"`: a flat response.

The beam is centred on each input's reference position (`CRVAL1`, `CRVAL2`).

### The primary beam cutoff

An input contributes only where its primary beam is at least `pb_cutoff`
(`--pb-cutoff`, 0.01 by default) of its response at the beam centre. The cutoff
depends on the beam alone, not on the noise of the input or on the other inputs.
An input noisier than the rest therefore keeps all of its area, and in a cube
the blank region of each channel depends only on its frequency. Apart from
pixels that no input covers, nothing else blanks the mosaic.

The peak of a holography beam can be slightly higher than its value at the
centre, and `holography_scale="stored"` measures the cutoff from the peak
instead (see [holography beams and leakage](holography.md)). With
`weight_type="weight_image"` the primary beam is not used, and inputs contribute
wherever they have weight.

The weight image, or the noise image written with `noise_image`, can be used to
blank a mosaic by its sensitivity afterwards.

## Combination

`combine_type` sets how the inputs are combined. Each input gives an estimate
`e_i = v_i / w_i` with weight `w_i`:

| `combine_type` | Image                     | Weight             |
| -------------- | ------------------------- | ------------------ |
| `weighted`     | `sum(w_i e_i) / sum(w_i)` | `sum(w_i)`         |
| `average`      | `mean(e_i)`               | `n^2 / sum(1/w_i)` |
| `sum`          | `sum(e_i)`                | `1 / sum(1/w_i)`   |

Pixels that no input covers are NaN in both the image and the weight.

### Noise

With `noise_image` (`--noise-image`), a noise image, `1 / sqrt(weight)`, is
written next to each weight: `<base>.<stokes>.noise.fits`, or
`<base>.<stokes>.noise.cube.fits` for cubes. When every input is weighted by the
inverse of its variance, this is the standard deviation of each pixel of the
mosaic, in the units of the images, for each combination above. For the weighted
mean with combined, inherent weights `w_i = 1 / sigma_i^2`, the mosaic is
`sum(w_i pb_i I_i) / W` with `W = sum(w_i pb_i^2)`, and its variance is
`sum(w_i^2 pb_i^2 sigma_i^2) / W^2 = 1 / W`.

This holds for weights measured from the noise of each input (`image_weight`),
and for `variance` or `rms` weight maps. With primary beam weights, or with
`image_weight="none"`, the noise image is only relative, and patchwork warns
about it. Weights logs can be in any units, so their noise images are relative
unless the logs hold inverse variances, as flint's do.

For cubes this gives the noise of every channel, which the error cubes of RM
synthesis need, without measuring the noise of the mosaic. It assumes that the
noise of different inputs is independent, as it is for different beams.

## Equivalent linmos and SWarp options

| linmos                                          | SWarp                                         | patchwork                                     |
| ----------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| `linmos.names`                                  | input images                                  | `images`                                      |
| `linmos.weights`                                | `WEIGHT_IMAGE`                                | `weights` (`--weights`)                       |
| `linmos.weighttype`                             |                                               | `weight_type`                                 |
| `linmos.weightstate`                            |                                               | `weight_state`                                |
|                                                 | `WEIGHT_TYPE`                                 | `weight_map_type`                             |
| `linmos.cutoff`                                 |                                               | `pb_cutoff`                                   |
| `linmos.finalcutoff`                            |                                               | none: always 0                                |
| `linmos.regrid.method`                          | `RESAMPLING_TYPE`                             | `kernel`                                      |
| `linmos.outputref`                              |                                               | `reference_index`                             |
| `linmos.psfref`                                 |                                               | `beam` of `mosaic`                            |
| `linmos.beams`                                  |                                               | `beam_numbers`                                |
| `linmos.outputcentre`                           | `CENTER`                                      | `centre`                                      |
|                                                 | `CENTER_TYPE`                                 | `centre_type`                                 |
|                                                 | `PIXELSCALE_TYPE`                             | `pixel_scale_type`                            |
|                                                 | `PIXEL_SCALE`                                 | `pixel_scale_arcsec`                          |
|                                                 | `PROJECTION_TYPE`                             | `projection`                                  |
|                                                 | `IMAGE_SIZE`                                  | `image_size`                                  |
|                                                 | `HEADER_ONLY`                                 | `patchwork frame`                             |
| the weighted mean only                          | `COMBINE_TYPE` (`WEIGHTED`, `AVERAGE`, `SUM`) | `combine_type` (`weighted`, `average`, `sum`) |
| `linmos.primarybeam` (`GaussianPB`, `ASKAP_PB`) |                                               | `pb_model` (`gaussian`, `holography`)         |
| `linmos.primarybeam.ASKAP_PB.image`             |                                               | `holography_cube`                             |
| `linmos.primarybeam.ASKAP_PB.alpha`             |                                               | `holography_alpha_rad`                        |
| `linmos.primarybeam.GaussianPB.*`               |                                               | `aperture_m`, `fwhm_scaling`                  |
| `linmos.useweightslog`                          |                                               | `.txt` files in `weights`                     |
| `linmos.regrid.decimate`                        | `PROJECTION_ERR`                              | `coordinate_step`                             |
| `linmos.removeleakage`                          |                                               | `remove_leakage`                              |
| `linmos.stokesinames`                           |                                               | `stokes_i_images`                             |
| `linmos.primarybeam.ASKAP_PB.normaliseleakage`  |                                               | `keep_centre_leakage` (inverted)              |

## Differences from linmos and SWarp

- The primary beam is evaluated on the output grid, from the offset of each
  pixel from the beam centre. linmos evaluates it on the input grid from pixel
  offsets, and then regrids it.
- A holography plane is scaled to an input's frequency by `f_image / f_plane`,
  as the beam widens with wavelength. linmos scales by the inverse ratio. The
  two differ noticeably only when the input frequency is far from the
  frequencies of the cube.
- A holography beam is scaled to 1 at its centre, where the ASKAP cubes hold
  values from 0.997 to 1.03. linmos uses the beam as stored, so its mosaics of
  `inherent` inputs are fainter by that factor. `holography_scale="stored"` uses
  the beam as linmos does, cutoff included. Flagged planes of the cube are
  skipped, whereas linmos takes the nearest plane even if it is flagged (see
  [holography beams and leakage](holography.md)).
- The frame is trimmed to the region where the inputs contribute. linmos covers
  the whole of every input image, and flint trims its mosaics afterwards.
- An input is left out where its primary beam is below `pb_cutoff` of its
  response at the beam centre. For a weight that is one value per input, and a
  beam that peaks at its centre, as a Gaussian beam does, this is the same as
  the linmos per-input cutoff, `w >= cutoff^2 max(w)`. linmos measures the
  cutoff of a holography beam from its peak, as `holography_scale="stored"`
  does.
- linmos also blanks its mosaic wherever the total weight is below
  `finalcutoff^2` times its peak, with `finalcutoff` defaulting to `cutoff`.
  That blanking depends on the noise of the inputs, and patchwork does not apply
  it. flint runs linmos with `finalcutoff = 0`, which turns it off.
- Pixels that no input covers are NaN, rather than 0.
- Interpolation does not extrapolate past the edge of an input or into blank
  pixels, as in SWarp. linmos extrapolates by up to a pixel.
- `cubic` is the Keys cubic convolution kernel, not casacore's bicubic
  interpolation, so results differ slightly from linmos `cubic`.
- A fitted frame is sized from the projected footprints of the inputs, rather
  than by SWarp's range heuristic, and is centred in a plane tangent to the sky,
  which also works near the poles.
- Stokes Q and U are rotated from the frame of each input, which has to be
  given, into the frame of the mosaic. linmos combines them as scalars, which is
  accurate where the frames agree and least accurate near the south celestial
  pole (see [polarisation](polarisation.md)).
- Not implemented:
  - SWarp's background subtraction, flux scaling (`FSCALE_*`,
    `FSCALASTRO_TYPE`), `RESCALE_WEIGHTS`, `WEIGHT_THRESH` and non-linear
    combinations (`MEDIAN`, `CLIPPED`, `MIN`, `MAX`, `CHI*`, flag combinations).
  - linmos Taylor terms, `removebeam`, sensitivity images and `findmosaics`.

# Convolution to a common beam

Images made with different restoring beams can be convolved to one beam before
they are mosaicked, so that every source in the mosaic has the same shape
(`patchwork.convolve`).

## The common beam

`patchwork.convolve.common_beam` finds the smallest beam that encloses every
input beam with `radio_beam.Beams.common_beam`, retrying with a smaller
tolerance if the search fails, as racs-tools does. Missing, zero and NaN beams
are left out, as are beams whose major axis is larger than `beam_cutoff_arcsec`.
The axes are rounded up to 0.1 arcsec and the position angle up to 0.01 degrees.
`circularise` makes the beam circular, and `target_beam` gives a beam to
convolve to instead.

## The convolution

`patchwork.convolve.convolve_to_beam` is a port of the racs-tools `robust`
convolution. The image is transformed, multiplied by the ratio of the Fourier
transforms of the new and old beams, and transformed back, so no kernel image is
made. As both beams are Gaussian their ratio is

```
exp(-2 π² kᵀ (C_new - C_old) k)
```

where `C` is the covariance of a beam and `k` the spatial frequency. The
covariances are converted to pixels with the pixel scale matrix of the image, so
any pixel orientation, including a rotated grid, is handled. The convolved image
is multiplied by the ratio of the beam areas, keeping it in Jy/beam.

The new beam must enclose the old one. A new beam that is smaller than the old
one by no more than rounding error, as a common beam can be, is accepted; a
larger shortfall is an error.

The transforms are circular, so the image is padded with zeros by five standard
deviations of the kernel, rounded up to a size the FFT handles efficiently. Flux
spread past one edge therefore does not wrap around onto the opposite edge.

NaNs are filled with zeros for the convolution. They are blanked again
afterwards, along with every pixel where more than half of the convolved signal
came from them.

An image whose beam is larger than `beam_cutoff_arcsec` is blanked and given a
zero beam, as racs-tools does.

Each convolved image is written next to its input, with `.conv` (or `--suffix`)
inserted before a final `.fits`, or added with `.fits` to any other name. An
output is never written over an input, even with `--overwrite`: images matched
by a pattern such as `beam*.fits` after an earlier run include its convolved
images, and convolving those again is refused before any work is done.

```bash
patchwork convolve beam*.fits
patchwork convolve beam*.fits --target-beam 30 30 0 --suffix 30arcsec
```

To convolve and mosaic in one step, give `--convolve natural` (or `total`) to
`patchwork mosaic`, or to `patchwork-pipeline` on a dask cluster. The convolved
inputs are removed once the mosaic is made, unless `--keep-convolved`.

```bash
patchwork mosaic beam*.fits --convolve natural --output-base field
```

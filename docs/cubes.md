# Cubes

Cubes of Stokes I, Q, U and V are mosaicked in one command, in the same way as
images: with `patchwork mosaic` on one machine, or `patchwork-pipeline` on a
dask cluster (`patchwork-cube` is another name for `patchwork-pipeline`).

```bash
patchwork-pipeline SB*/beam*.{i,q,u}.cube.fits --weights SB*/beam*.{i,q,u}.weights.txt \
    --output-base field --convolve natural --beam-cutoff-arcsec 30 \
    --pb-model holography --holography-cube akpb.iquv.cube.fits --remove-leakage \
    --frame-centre 12.5 -81.0 --cluster-config slurm.yaml
```

This writes `field.i.cube.fits`, `field.q.cube.fits` and `field.u.cube.fits`,
with `field.i.weight.cube.fits` and `field.qu.weight.cube.fits` (Q and U share a
weight). With `--output-format zarr` it writes zarr stores instead.

## Inputs

The cubes can be given in any order. The Stokes parameter of each cube is read
from its STOKES axis. Cubes on the same pixel grid are taken to be the Stokes
parameters of one beam, and every beam must have the same Stokes parameters.
Cubes of a single Stokes parameter on one grid, such as two epochs of one
pointing, are separate inputs, but mixed with cubes of other Stokes parameters
they cannot be paired with them, and are an error. Q or U alone is mosaicked
without rotation, with a warning, because it cannot be rotated into the frame of
the mosaic.

Weights are given one for each cube, in the same order as the cubes. A `.txt`
file is a linmos weights log: a `#Channel Weight` header, then a weight for each
channel, as arrakis writes them. Any other file is a weight cube on the grid of
its cube. Without `--weights`, each channel is weighted by the inverse variance
of its noise (`image_weight`), measured on the workers from every
`weight_stride`th row. Q and U are weighted by the inverse of their mean
variance, because the rotation mixes them.

Each cube holds one Stokes parameter and any number of channels along a FREQ
axis. Both axis orders are read: (RA, Dec, FREQ, STOKES), as wsclean writes, and
(RA, Dec, STOKES, FREQ), as CASA and arrakis write. The mosaic has the axis
order of its reference cube.

Every cube must have the same channels, to within 1% of a channel width.
`--channels START STOP` mosaics only some of them.

The restoring beam of each channel is read from a BEAMS table (as CASA and
fitscube write), or from the header. A channel whose beam is zero, NaN or
fitscube's blank placeholder holds no image, and that cube gives the channel no
weight.

## Steps

1. Convolution, with `--convolve`. Each cube is convolved to a common beam in
   one task, one channel at a time:
   - `natural`: each channel to the smallest beam that encloses that channel of
     every cube, of all Stokes parameters, so that I, Q and U stay matched;
   - `total`: every channel to the same beam;
   - `--target-beam`: to a given beam.

   A channel whose beam is larger than `--beam-cutoff-arcsec`, or that has no
   beam, is blanked. With `--channels`, only those channels are convolved, into
   cubes named `<cube>.conv.ch<start>-<stop>.fits`, so that runs over different
   channels can share a directory. The convolved cubes are deleted afterwards
   unless `--keep-convolved` is given.

2. Weights are read, or measured from the noise of each channel.
3. Planning. The frame is fitted or taken from a reference cube, and trimmed to
   where the beams contribute.
4. Mosaicking. Each task makes one tile over a few channels (`--channel-chunk`)
   for all Stokes parameters at once:
   - the positions of the tile in each beam, and the interpolation weights of
     each pixel, are calculated once for all channels and Stokes parameters;
   - the primary beam is evaluated for each channel: a Gaussian whose width
     scales as 1/f, or the usable holography plane nearest the channel, read on
     the worker;
   - the leakage of Stokes I is removed channel by channel;
   - Q and U are rotated into the frame of the mosaic; the rotation is the same
     for every channel (see [polarisation](polarisation.md)).
5. If `noise_image` is given, the noise is written from the weights. Each tile
   is otherwise final once written, because inputs are only left out where their
   primary beam is below `pb_cutoff`.

## Outputs

FITS is the default. There is one cube for each Stokes parameter, and one weight
cube for each group of Stokes parameters that share a weight, named after the
group: the weight that Q and U share is `field.qu.weight.cube.fits`. The
restoring beams are written to the header if every channel has the same beam,
and otherwise to a BEAMS table, as fitscube writes them. These are the cubes
that the RACS-all RM synthesis flow reads. The main process, the one that runs
the mosaic, writes each tile as it arrives, so only one process writes each
file. With `--fits-writer workers`, each task writes its own strip of rows
instead (see [pipeline](pipeline.md)).

With `--output-format zarr` (which needs `pip install askap-patchwork[zarr]`)
there is one store for each cube. Each store is a single (channel, y, x) float32
array, chunked over all channels, with the FITS header in `attrs["fits_header"]`
and the beam of each channel in `attrs["beams"]`. This is the layout that
rm-lite's `fits_cube_to_zarr` writes and `read_zarr_cube_dask` reads, so rm-lite
can read the stores as they are. Each task makes one chunk over all channels and
writes it, so no image data passes through the main process. `--zarr-chunk` sets
the chunk size, which also sets the memory a task needs: about
`channels x chunk^2 x 20` bytes for I, Q and U.

The FITS header of a zarr store describes its beams in the same way as a FITS
cube: one beam if every channel has the same beam, and otherwise none, with
`CASAMBM` set.

Before any work starts, the outputs, and any convolved cubes, are checked
against the inputs and their weights. An output is never written over an input,
even with `--overwrite`, because the outputs are created empty before the inputs
are read.

## Memory and speed

A FITS task needs about `tile_size^2 x channel_chunk x stokes x 50` bytes: 0.6
GB for the default 512² tiles over 16 channels of I, Q and U. Halving the tile
size quarters this, and halving the chunk halves it. Tiles share their positions
and primary beams between channels, so a task over more channels of a smaller
tile is faster in the same memory: 512² tiles over 16 channels took two thirds
of the time of 1024² tiles over 4 channels.

For FITS output, every tile passes through the main process on its way to disk.
It gathers the tiles of each band of rows and writes the band whole, holding up
to `--fits-buffer-mb` of bands (4096 MB by default; see
[pipeline](pipeline.md)). A band of a cube takes
`tile_size x frame width x channel_chunk x 4` bytes for each image and weight:
3.3 GB for 512² tiles over 16 channels of I, Q and U across a frame 20000 pixels
wide. The default then holds one band at a time, and the tiles of a band that
starts before it is written are written a row at a time, which is far slower on
a shared file system, so raise `--fits-buffer-mb` to two or three bands where
the main process has the memory. On a cluster, use `--fits-writer workers` on
Lustre or GPFS, or zarr output on any file system; both leave the writing to the
workers (see [pipeline](pipeline.md)). The main process holds no more than
`--tasks-in-flight` tiles, by default twice the number of threads making them.

On four threads of a laptop, four beams of I, Q and U, each of 16 channels of
1200² pixels, are mosaicked at about 55 million input pixels per second (cubic
kernel, numba).

## Compared with linmos

| linmos                            | patchwork                                                                 |
| --------------------------------- | ------------------------------------------------------------------------- |
| One run for each Stokes parameter | One run for all of them                                                   |
| `linmos.names`, `linmos.weights`  | `cubes`, `--weights`, in the same order                                   |
| `linmos.useweightslog`            | A `.txt` weight is a log                                                  |
| `linmos.stokesinames`             | The Stokes I cube of each beam, found by its grid, or `--stokes-i-images` |
| `linmos.psfref`                   | The beams convolved to, otherwise those of the first cube                 |
| Whole cubes held in memory        | Tiles over a few channels                                                 |
| Channels assumed to match         | Channels checked                                                          |

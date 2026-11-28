# Performance

## How a tile is made

For each input that covers a tile, patchwork:

1. finds where the tile's pixels fall in the input (sky coordinates, two WCS
   transforms),
2. reads the region of the input those pixels need,
3. interpolates the input there,
4. evaluates the primary beam, weights the input and adds it to the sums.

## How the work is reduced

- Each input is used only where its primary beam is above the cutoff. flint's
  beam images extend well past the primary beam: about half of a typical ASKAP
  beam image lies within the cutoff.
- An input that covers only part of a tile is used only over that part.
- Sky coordinates are transformed exactly on a grid of points and interpolated
  between them (see the coordinate grid, below).
- Interpolation takes a simpler path where a region has no NaN pixels and no
  kernel taps beyond its edge, which is true of most regions.
- numba can be used for the interpolation (see numba, below), or for the whole
  of a tile, on several threads (see the fused backend, below).
- Tiles can be made in threads or processes (`max_workers`, `processes`) without
  dask. The numba kernels release the GIL, so threads run them in parallel.
- Convolution pads each image to a size the FFT handles efficiently. flint's
  default images are 10128 pixels across, and 10128 has a prime factor of 211;
  padding makes their convolution 1.6 times faster.
- Noise is measured from every `weight_stride`th row. Each plane of a plain
  float32 image is still read whole, in reads of up to 64 MB of rows, and the
  rows taken from it: on Lustre a read of a few MB takes little longer than a
  read of one row, so reading the rows alone, through a memory map, is slower.
- Only the holography planes that are used are read, each with one read, and
  each is kept: the Stokes parameters of a beam, and its beams built with other
  rotations as the mosaic is planned, read it once. The planes nearest each
  frequency are looked at first, so the centre of every plane is not read. The
  beams of different images are read in `tile_threads` threads.

## Timings

Four beam images of 6144² pixels of 2.5″ at 888 MHz
(`tests/conftest.py:image_header`), with a primary beam cutoff of 0.1, on an
Apple M-series laptop:

| Kernel   | numpy | numba | numba, 4 threads | dask, 4 workers |
| -------- | ----- | ----- | ---------------- | --------------- |
| linear   | 9.8 s | 6.1 s |                  |                 |
| cubic    | 18 s  | 7–9 s | 3.3 s            | 3.2 s           |
| lanczos3 | 43 s  | 12 s  |                  |                 |

The times vary by 10 to 20% between runs. Other machines will give different
times, but similar ratios.

### The coordinate grid

Sky coordinates are calculated exactly every `coordinate_step` pixels (16 by
default) and interpolated in between. Between projections a few degrees apart
the mapping is smooth, and the interpolated positions are accurate to about 10⁻⁴
pixels. Both the sky positions of a tile and their positions in each input are
checked against exact values at the middle of every cell of the grid (and
between the points of a tile one pixel thick). Where either is more than 0.001
pixels out, or where part of a tile cannot be mapped into an input, every pixel
is transformed exactly. `coordinate_step=1` transforms every pixel exactly.

The sky positions are kept as unit vectors. The primary beam and the
polarisation angles are calculated from them, and remain well defined at the
poles.

## numba

numba is optional:

```bash
pip install askap-patchwork[numba]
```

`ResampleOptions.backend` is `auto` by default, which uses numba if it is
installed and numpy otherwise. `numpy` or `numba` can be chosen explicitly.

The numba kernel follows the numpy one. Their results are identical for the
nearest, linear and cubic kernels, and agree to 10⁻¹⁵ for Lanczos, whose taps
numba calculates with fewer trigonometric calls. `fastmath` is off.

The first call compiles the kernel, which takes about a second. The compiled
kernel is cached on disk, so later processes on the same machine, such as dask
workers, load it in much less time.

The cache is kept in a directory of your own in the temporary directory,
`patchwork-numba-<uid>` under `TMPDIR`, which is usually `/tmp`. Set
`NUMBA_CACHE_DIR` to put it somewhere else, such as `/dev/shm` to keep it in
memory. numba's own default is a `__pycache__` directory beside patchwork's
code. On a cluster whose workers share one installation, they all compile into
that one directory at the same time, and on NFS a worker can fail with
`OSError: [Errno 116] Stale file handle` while another rewrites the cache's
index, which stops the mosaic. Each machine has its own temporary directory, so
this does not happen there.

numba runs whatever it finds in its cache, so patchwork only uses the directory
if it belongs to you and no one else can write to it. If not, for example
because another user made it first, each process makes a new private directory
and compiles the kernel again. The kernels of other packages are still cached
where numba would put them.

The kernels run on one thread and release the GIL. Parallelism comes from making
several tiles at once, in threads or processes (`max_workers`) or on dask
workers, so dask's usual one single-threaded worker per core is not
oversubscribed. For this reason these kernels do not use `prange`. Those of the
fused backend, below, do when they are given more than one thread.

numba depends on llvmlite, whose wheels can lag behind new Python versions and
are not built for every platform; llvmlite 0.50, for example, has no wheel for
Intel macOS. Because numba is optional, patchwork can be installed without it
and use the numpy backend. flint already depends on numba, through racs-tools.

## The fused backend

With the `numba` backend only the interpolation is compiled. The primary beam,
the leakage, the rotation of Q and U, the weighting and the summing are numpy
passes over each region, on one core, holding the GIL. `backend="fused"`
(`--backend fused`) does all of them in one compiled pass over each input
(`patchwork.fused_kernels`), a row of one channel at a time, and can share a
tile between threads. It needs numba, and is not what `auto` chooses.

It is meant for a dask worker that runs one task at a time but has many cores,
as a cluster set up for a multithreaded imager has: each task then uses all of
the worker's cores (see [pipeline](pipeline.md)). On one thread it is also
faster than the `numba` backend.

`tile_threads` (`--tile-threads`) sets how many threads each tile is made with:

- `1`, the default, runs in the calling thread, as the `numba` backend does.
  Keep it where every core already has a worker of its own.
- `0` shares the CPUs the process may run on between the tiles it makes at once.
  A dask worker makes as many tiles at once as it has threads, so a worker with
  one thread in a 32-CPU SLURM job uses 32 threads for each tile, and a worker
  with a thread for each CPU uses one. `patchwork mosaic` shares them between
  its `--max-workers`.
- Any other number is used as it is. Give one where several worker processes
  share the CPUs of one job (`processes: 4` in a `dask_jobqueue` job of 16
  CPUs): they cannot see each other, so with `0` each would use all 16.

The CPUs a process may run on are those of its affinity, which SLURM sets to the
job's allocation. numba starts no more threads than that, or than
`NUMBA_NUM_THREADS`. Each process logs once how many threads it makes tiles
with, and which of numba's threading layers runs them.

Every backend measures the noise with `tile_threads` threads too, each reading
and measuring one channel of one image at a time, and holding up to 64 MB of its
rows. patchwork's own pool shares the images between its `max_workers` tasks, so
the many single-channel images of flint's polarisation flow are measured
`tile_threads` at once. A dask cluster, whose threads are counted as it scales,
is given a task for each image, so it sees them all waiting. Each cube is
convolved in one task, its channels in `tile_threads` threads (each channel's
FFTs and copies release the GIL). Without these threads, both stages on a single
worker take as long as every channel done one after another: measuring 36 cubes
of 32 channels took 12 minutes on one core of a Lustre-backed node, and a
convolution is two padded FFTs for each channel, and small ones near its edges
and NaNs. A convolving thread holds about three padded planes, some 1.2 GB for
10000² planes, so give the worker memory for `tile_threads` of them.

With more than one thread, three things happen at once in a tile:

- The inputs are read by `tile_threads` threads, a few inputs ahead of the one
  being added (up to 256 MB for each thread), each image in several runs of
  channels. Reading is mostly waiting for the file system, and a parallel file
  system serves many reads at once. Channels without an image are not read.
- numba's threads add each input to the tile in turn, each thread to its own
  pixels.
- They then turn the sums into the tile, and for `--fits-writer workers` its
  files are written side by side.

Finding where each input falls on a tile is still Python, so larger tiles keep
the threads busier: use a `tile_size` of 1024 or 2048 with many threads.

The reading and finding threads run on the same CPUs as numba's, and a parallel
kernel finishes when its slowest thread does, so a numba thread that waits for a
CPU holds up the rest. On a 16-core node the kernels alone ran 13 times faster
on 16 threads than on one, but only 3 times faster inside a mosaic where 16
reader threads ran beside them. `kernel_threads` (`--kernel-threads`) runs the
kernels on that many of `tile_threads` and reads with the rest, so no more
threads than CPUs are busy. On that node, with the inputs on a striped Lustre
file system, 12 kernel threads added 18 inputs to a strip of 2048 rows in 2.8 s
(3.8 s when 16 shared their CPUs with the readers), but the 4 readers took 6 s
to find and read 2.3 GB for it, so the kernels waited; the two balance at about
10 kernel threads of 16. The default, 0, gives the kernels 5 of every 8 threads
when a tile has at least 4, and has every thread do both with fewer; setting
`kernel_threads` to `tile_threads` has every thread do both whatever the count.

Timings for nine beams of I, Q and U cubes of 1500² pixels and 16 channels, with
holography beams and the leakage removed, the strips written by their tasks, on
an Apple M-series laptop with 10 performance cores. The files were in the page
cache, so these measure the computing and not the file system:

| Backend | How it was run                        | Time  |
| ------- | ------------------------------------- | ----- |
| numba   | 1 thread                              | 31 s  |
| numba   | 10 processes                          | 8.2 s |
| fused   | 1 thread                              | 22 s  |
| fused   | 2 threads for each tile               | 12 s  |
| fused   | 4 threads for each tile               | 7.6 s |
| fused   | 10 threads for each tile              | 5.6 s |
| fused   | 10 threads, `tile_size` 2048          | 4.3 s |
| fused   | 10 processes of 1 thread              | 5.7 s |
| fused   | 1 dask worker of 1 thread, 14 threads | 6.3 s |

The kernels alone are 6 to 7 times faster on 10 threads than on one on this
machine. They have not been timed on a cluster node, or reading from a parallel
file system, where reading the inputs may take longer than computing with them.

The fused kernels follow the numpy code, and their mosaics agree with those of
the other backends to round-off (the taps of a pixel are summed row by row).
Three things differ:

- An input weighted by weight images is added as the `numba` backend adds it,
  because the kernels do not interpolate weights. Weights logs and measured
  weights go through the kernels.
- numba's default threading layer stops the process if two Python threads start
  parallel kernels at once, so Python threads that each make a tile
  (`--max-workers`, or a dask worker with several threads) take turns at numba's
  threads, and read and write while they wait.
- Each plain float32 FITS input has its header parsed once in each process, and
  is then read from a memory map without holding the GIL. Any other file is read
  as the other backends read it.

## Choosing how to run

- On one machine, use `patchwork mosaic ... --max-workers N`, which makes tiles
  in N threads. The numba kernels release the GIL, but the Python code around
  them holds it, so the threads take turns at that part. A few threads keep up
  with as many processes, but many do not: on a 32-core node, 32 threads kept 10
  to 20 cores busy, and 32 processes kept all of them busy. With more than a few
  cores, add `--processes` (`processes=True`). Each process reads the mosaic's
  plan once, and what it logs is passed to the main process, which prints it or
  hands it to your own logging as it would its own. The processes are spawned,
  so a script that calls `mosaic(..., processes=True)` must do so under
  `if __name__ == "__main__":`. A dask worker cannot start processes of its own,
  so from inside one, use threads or run patchwork in a subprocess.
  `patchwork-pipeline --n-workers N` also works on one machine, with a local
  dask cluster of N single-threaded processes.
- On a cluster, use `patchwork-pipeline ...` with single-threaded workers, one
  for each core, and have the workers write the mosaic: `--fits-writer workers`
  on Lustre or GPFS, or `--output-format zarr` on any file system. Otherwise
  every tile passes through the process running the pipeline, which the workers
  can end up waiting for. Each tile is a task, and no more than
  `--tasks-in-flight` tiles are made at once. On a cluster that adapts its size,
  set that to twice the most threads it can have, so it has enough work in view
  to grow (see [pipeline](pipeline.md)).
- On a cluster whose workers have one thread and many cores, use
  `--backend fused --tile-threads 0` (see the fused backend, above), with a
  larger `--tile-size`.
- The default tile size of 512 pixels keeps the memory of a tile to a few tens
  of megabytes per input, and is as fast as larger tiles. Smaller tiles spread
  better over many workers. For cubes, the number of channels each task makes
  (`channel_chunk`, 16 by default) matters more than the tile size; see
  [cubes](cubes.md).
- `cubic` is the default kernel. The Lanczos kernels take about twice as long,
  and are better only for images sampled close to their resolution.

## Not implemented

- Interpolating weight images in the fused kernels, and making the fused backend
  the one `auto` chooses.
- Convolving in flight, as each tile reads its inputs, instead of writing
  convolved cubes and reading them back. The kernel is the difference of the
  beams, which reaches only a few pixels (`reach` in `convolve_to_beam`), so a
  tile needs its input window plus that margin, convolved with window-sized FFTs
  in the reader threads; neighbouring tiles repeat only the margin. It would
  remove the intermediate cubes, about 4 TB for each Stokes of a full RACS cube
  mosaic, for about the same computing. What has to be settled first: the noise
  weights are measured on the convolved cubes, so they would instead be measured
  on convolved samples of the strided rows (with the margin); the NaN mask
  doubles the FFTs of inputs with NaNs; the tapers become one per input, channel
  and tile; and the convolution joins each tile's read phase, so the readers
  must stay ahead of the kernels. The convolved cubes are also a product
  (`keep_convolved`, `patchwork convolve`), so the separate pass stays as an
  option.
- Interpolating the primary beam from the coordinate grid. The beam is smooth,
  but it is evaluated at every pixel, because the cutoff is applied to it.
- Summing in float32. This would halve the memory traffic of the sums, but
  reduce their precision.
- Compiled extensions (Cython, C++) and GPUs. These would need compiled wheels
  or extra hardware; the flint crew packages are all pure Python.

# Pipeline

The `patchwork` library uses numpy and astropy, and does not import dask.
`patchwork mosaic` does all of the steps below, convolution included, on one
machine with `--max-workers` threads or processes. The pipeline in
`patchwork.pipeline` does the same steps on a dask cluster. Below, the main
process is the one that runs `patchwork-pipeline`, or calls `run_pipeline`.

1. The restoring beams of the images are read, and the beam that each channel is
   convolved to is found (with `--convolve natural` or `total`).
2. Each image or cube is convolved on a worker (with `--convolve`).
3. The noise of each channel of each image is measured on a worker (with
   `image_weight` other than `none`).
4. The images are grouped into beams, and the frame and inputs of the mosaic are
   prepared (`patchwork.plan.plan_mosaic`).
5. Each tile is made on a worker (`patchwork.tile.mosaic_tile`). For FITS
   output, the main process gathers the tiles as they arrive and writes each
   band of them, a row of tiles across the frame, once it is whole, so only one
   process writes the files, unless `--fits-writer workers` is given (see
   below). For zarr output, the workers write the tiles. When the main process
   writes, no more than `--tasks-in-flight` tiles are made at once, by default
   twice the number of threads in the cluster, which is checked again as the
   cluster scales, so it holds no more than that many tiles however far ahead of
   it the workers are, besides the bands it gathers (up to `--fits-buffer-mb`).
   When the workers write, every tile is queued at once unless
   `--tasks-in-flight` is given. The number of tiles finished is logged every 30
   seconds, with an estimate of the time remaining.
6. If a noise image is requested, it is written from the weights. Otherwise each
   tile is final once written.

```bash
# On this machine, with eight workers
patchwork-pipeline beam*.fits --output-base field --convolve natural --n-workers 8

# On a cluster
patchwork-pipeline beam*.fits --output-base field --convolve natural --cluster-config slurm.yaml
```

Without `--cluster-config`, a local cluster of single-threaded workers is
started. A cluster configuration file gives the class of the cluster, the
arguments it is created with and, optionally, how it adapts its number of
workers. For example, with `dask-jobqueue` installed:

```yaml
cluster_class: "dask_jobqueue.SLURMCluster"
cluster_kwargs:
  cores: 1
  processes: 1
  memory: "16GB"
  walltime: "0-1:00:00"
  log_directory: "patchwork_logs"
adapt_kwargs:
  minimum: 1
  maximum: 32
```

From Python, pass a `distributed.Client` to
`patchwork.pipeline.flow.run_pipeline`.

## Workers

Each task uses one core unless `tile_threads` says otherwise: the interpolation
runs on one thread, and so does the FFT of a convolution. Give the cluster
single-threaded workers, one for each core, as in the example above.

- A worker with one thread and many cores, as a cluster set up for wsclean might
  have, runs one task at a time and leaves its other cores idle. There, use the
  fused backend with `--tile-threads 0`, which makes each tile with every CPU of
  the worker's job (see below).
- A worker with many threads runs that many tasks at once, but they share
  Python's lock (the GIL), so they keep fewer cores busy than as many
  single-threaded workers. `patchwork mosaic --max-workers 32` shows the same
  thing on one machine: on a 32-core node its threads kept 10 to 20 cores busy,
  and with `--processes` all 32 were busy.

An adaptive cluster grows according to the work dask can see waiting. When the
workers write the mosaic (`--fits-writer workers`, or zarr) patchwork queues
every tile at once, so the cluster sees the whole mosaic and grows to its
maximum; the convolution and noise stages queue every image likewise. When the
main process writes, patchwork submits no more than `--tasks-in-flight` tasks at
a time, by default twice the number of threads the cluster has when it checks,
so a cluster that starts small may grow slowly, or not at all if each task takes
less than a few seconds. Either give the cluster a fixed size, with `minimum`
the same as `maximum`, or set `--tasks-in-flight` to twice the most threads it
can grow to: 64 for the example above, at the cost of the main process holding
that many tiles at once.

## Workers with more cores than threads

A cluster whose jobs each hold one dask worker with one thread and many CPUs
suits a multithreaded imager, and a flow that images and mosaics on one cluster
has such workers:

```yaml
cluster_class: "dask_jobqueue.SLURMCluster"
cluster_kwargs:
  cores: 1
  processes: 1
  job_cpu: 32
  memory: "256GB"
```

With `backend="fused"` and `tile_threads=0` each task makes its tile with every
CPU of its job, reading, computing and writing on 32 threads (see
[performance](performance.md)). Each worker logs the number of threads it uses.
The noise of each image is measured, and each cube convolved, with the same
threads, with any backend; without them, those stages on a lone worker do every
channel of every cube one after another on one core.

```python
from patchwork.options import MosaicOptions, ResampleOptions, RunOptions

options = MosaicOptions(
    resample=ResampleOptions(backend="fused", tile_threads=0, tile_size=2048)
)
run_options = RunOptions(fits_writer="workers")
```

- A task of a larger tile keeps its threads busier, and holds more: about
  `tile_size^2 x channel_chunk x 12` bytes for each image and weight of the
  mosaic, and up to 256 MB of inputs for each thread.
- The cluster still runs one task for each worker, so when the main process
  writes `--tasks-in-flight` matters as it does above: an adaptive cluster that
  starts with one worker has two tasks in view, and may not grow.
- Have the workers write the mosaic (below), or the tiles queue for the main
  process as before.
- Where a job holds several worker processes, give `tile_threads` the CPUs of
  the job over its processes, as they cannot see each other.

## Writing FITS from the workers

By default every tile of a FITS mosaic is sent to the main process, which writes
it. Each row of a tile is a short run of bytes in a file, and on a shared file
system such as Lustre a short write takes about as long as a long one. So the
main process gathers the tiles of each band, a row of tiles across the frame,
and writes the band once all its tiles have arrived, as one run of bytes for
each channel of each file. On Lustre, 4 GB written this way took 6 s; written a
tile row at a time it took 3 minutes, and through a memory map, as patchwork
0.1.0 wrote it, 8 to 10 minutes. The bands are held in memory, up to
`--fits-buffer-mb` (4096 MB by default). A band takes
`tile_size x frame width x channel_chunk x 4` bytes for each image and weight,
and the tiles of a band that does not fit are written a row at a time. Every
tile still passes through the main process, so on a large cluster the workers
can wait for it: in a test on Lustre with 32 workers, with patchwork 0.1.0, a
mosaic the main process wrote took about ten times as long as one the workers
wrote themselves. With `--fits-writer workers` (`fits_writer="workers"`), each
task writes its own part of the FITS files and returns nothing to the main
process. On a file system where several machines cannot write one file, such as
NFS, write zarr instead (`--output-format zarr`, see [cubes](cubes.md)), which
is about as fast. Leave the writing to the main process only on one machine.

The main process creates the files, without data, and finds where the data of
each begins. The frame is divided into strips of whole rows, each about the size
of a `tile_size` tile, so the rows of one channel that a task writes are one
contiguous block of each file. Each task writes them in place, without changing
the header or any other part of the file. The workers also write the noise
images strip by strip, so no image data passes through the main process.

Several machines then write to the same file at once. This needs a file system
that keeps the file consistent, such as Lustre or GPFS. NFS does not, which is
why the main process writes the files by default.

On Lustre, each file is spread over as many storage targets as its directory
specifies, and many sites default to one. With a single target, the workers are
limited much as one process is, so set the striping of the output directory
first, for example `lfs setstripe -c 8 <output directory>`.

The coordinates of a strip are sampled on a different grid from those of a
square tile, so the values differ from those of a mosaic written by the main
process by up to a few parts in 10⁵ of the peak, with the same pixels blank, as
they do between tile sizes.

```bash
patchwork-pipeline beam*.{i,q,u}.cube.fits --weights beam*.{i,q,u}.weights.txt \
    --frame-centre 12.5 -81.0 --output-base field --fits-writer workers \
    --cluster-config slurm.yaml
```

## From a Prefect flow

A Prefect task running on a `DaskTaskRunner`, as flint's tasks do, can run the
mosaic on the same cluster:

```python
from prefect import task
from prefect_dask import get_dask_client

from patchwork.pipeline.flow import run_pipeline


@task
def task_mosaic(images, output_base):
    with get_dask_client(set_as_default=False) as client:
        return run_pipeline(images, output_base, client)
```

- The task is the main process of the mosaic. For FITS output it receives and
  writes every tile, so its worker holds up to `tasks_in_flight` tiles, and the
  bands of tiles it gathers, up to `fits_buffer_mb`. Each tile is about
  `channel_chunk x tile_size^2 x 4` bytes for each output image and weight: 84
  MB for 512² tiles over 16 channels of I, Q and U. Set `tasks_in_flight` and
  `fits_buffer_mb` to fit the memory of that worker, or on Lustre have the
  workers write the files (`fits_writer="workers"`), in which case the task
  receives no tiles.
- The task keeps its worker thread while it waits for the tiles, so the cluster
  needs at least one other thread to make them.
- Add `patchwork` to `PREFECT_LOGGING_EXTRA_LOGGERS` to see its messages in
  Prefect.
- Make the whole mosaic in one Prefect task. If each of thousands of tiles were
  a Prefect task, each would be recorded as a task run on the Prefect server.

## Logging

patchwork logs to the `patchwork` logger, at INFO level. Importing it does not
add a handler, so in an application such as flint its messages go wherever that
application's root logger sends them, and appear once. The command-line tools
call `patchwork.logging.setup_logging()` to print the messages, and
`patchwork-pipeline` does the same on every dask worker, including workers that
join later. Call it from your own scripts to print them in the same way.

With `--log-level DEBUG` (`setup_logging("DEBUG")`) each tile also logs where
its time went: how long it waited for its inputs to be read, spent adding them
and finalising, how long its threads spent finding and reading the inputs and
how much they read, and how long its strip took to write. The waiting and adding
times are those of the tile; the finding and reading times are summed over the
threads, so can exceed them. A tile that waits more than it adds is held up by
the file system.

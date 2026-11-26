# Holography beams and leakage

An ASKAP holography cube is a measurement of each beam of the phased-array feed:
how it attenuates Stokes I, and how much unpolarised Stokes I it turns into
apparent Q, U and V. patchwork uses the Stokes I response to weight every Stokes
parameter of an input, and the Q, U and V responses to remove that leakage.

```bash
# ALPHA is the field's polarisation axis plus 45 degrees, in radians, and
# the boresight (here RA 12.5, Dec -81) orients the beams and Q and U
patchwork mosaic beam*.{i,q,u,v}.fits --pb-model holography \
    --holography-cube akpb.iquv.closepack36.54.943MHz.SB55219.cube.fits \
    --holography-alpha-rad "$ALPHA" --remove-leakage \
    --frame-centre 12.5 -81.0 --output-base field
```

## The cube

The cube has axes (beam, stokes, freq, dec, ra) in numpy order, which are the
FITS axes RA, DEC, FREQ, STOKES, BEAM. The ASKAP cubes in use, such as
`akpb.iquv.closepack36.54.943MHz.SB55219.cube.fits`, have:

- 36 beams;
- Stokes I, Q, U and V for each beam;
- 288 planes, 1 MHz apart;
- planes of 101 × 101 pixels of 2.07′, which extend 1.7° from the beam centre.

patchwork reads the cube as follows.

The spatial axes are offsets from the beam centre. Only CRPIX and CDELT are
used, and CRVAL is ignored. A cube whose CRPIX is within two pixels of its edge
is refused.

The Stokes I plane is the response of the beam. It is close to 1 at the centre,
but not exactly 1: in the three ASKAP cubes it is between 0.997 and 1.03 there.

The Q, U and V planes are fractional leakages, Q/I, U/I and V/I. In the 943 MHz
cube, where Stokes I is below 5% of its peak, the total leakage √(Q² + U² + V²)
is larger than I at almost every pixel, which would be impossible for a
response.

A cube without a Stokes axis (beam, freq, dec, ra) is read as Stokes I only. It
can be used to weight inputs, but not to remove leakage.

## The beam of each channel

The beam number is taken from the `beamNN` in the input's file name, as flint
names its images. Otherwise it is given as the `beam_numbers` of
`patchwork.mosaic.mosaic`.

Each channel of each input uses the plane nearest its frequency, and its leakage
comes from the same plane. Of two planes as near, the first is used.

- Planes with no response at the beam centre are flagged channels, which are NaN
  in the cubes, and are skipped. In the 1295 MHz cube, for example, every beam
  has the 144 planes from 1152 to 1295 MHz flagged, so a channel at 1200 MHz
  uses the plane at 1296 MHz.
- Scaling a plane to another frequency (see below) is only an approximation to
  the beam close to that plane's frequency. patchwork therefore warns, once per
  mosaic and before any work starts, when the plane used for a channel is more
  than 2 MHz from it. This happens among flagged planes, and beyond either end
  of the cube.
- A beam with no usable plane is an error, as for beam 12 of the 1655 MHz cube.
- The planes are looked at from the nearest out, so only those nearer than the
  usable plane are read. `patchwork.primary_beam.usable_planes` reads the centre
  of every plane, to list the usable ones.

The beam widens with wavelength, so a channel at frequency `f` reads the plane
at offset `s r`, with `s = f / f_plane`. On a plane, `s` is 1.

The Stokes I plane is divided by its value at the beam centre, so the beam is 1
there. Calibration on a source at the beam centre sets the gain there to 1, so
this is the attenuation that the image carries. The leakage planes are fractions
and are not rescaled. With `holography_scale="stored"`
(`--holography-scale stored`), the planes are used as stored, and each input's
`pb_cutoff` is measured from the beam's peak rather than its centre. This is for
comparison with linmos mosaics (see below).

## Where the pattern is sampled

Each pixel has an offset from the beam centre, which is the input's reference
position (CRVAL), given as direction cosines: `l` to the east and `m` to the
north. From these, `r = asin √(l² + m²)` and `PA = atan2(l, m)`, and the pixel
of the cube that is read is

```
x = sin(s r) sin(PA − α − η)
y = sin(s r) cos(PA − α − η)
cube pixel = (x / CDELT1, y / CDELT2) + CRPIX − 1
```

The cube is interpolated linearly at that pixel. Positions outside the cube give
zero, and NaN pixels count as zero. The offsets are those of the pixels of the
mosaic, so the beam is evaluated at the position of each output pixel.

`α` is the rotation of the feeds from their position when the holography was
measured, `holography_alpha_rad`. The holography is measured with the feeds at
−45°, so for a field `α` is its polarisation axis plus 45°.
`patchwork.primary_beam.pb_rotation_from_pol_axis` calculates it, as flint's
`compute_pb_rotation_alpha` does.

`η` is zero unless the input's frame is north at some position other than its
own centre: `frame_centre`, or `POLFRAME = 'FIXED'` keywords in its header (see
[polarisation](polarisation.md)). It is the angle, at the beam centre, between
local north and the north of the boresight. The feeds are fixed relative to the
north of the boresight, not of each beam centre (see
[polarisation](polarisation.md)), so the pattern is also rotated by `η`.

## Removing the leakage

For an unpolarised source, the beam produces apparent Q, U and V equal to the
fractional leakage times Stokes I, so each input holds

```
Q_obs = Q + L_Q I,   U_obs = U + L_U I,   V_obs = V + L_V I
```

This holds whether the images still carry the primary beam or are corrected for
it, because Q and I carry the same beam in either case. The leakage is removed
by `Q = Q_obs − L_Q I`, and likewise for U and V. Here `I` is the Stokes I image
of the same beam, in the same weight state and at the same resolution as Q.

For each input, channel and pixel of the mosaic (`patchwork.tile._add_input`):

1. Q, U and V, and the Stokes I image of the beam, are interpolated at the
   pixel. The pixel is used only where the Stokes I image is also valid.
2. `L_Q`, `L_U` and `L_V` are sampled from the channel's plane at the position
   given above. `L_V` is sampled only if there is a V image.
3. The leakage at the beam centre is subtracted from the whole pattern, because
   calibration on a source at the beam centre has already removed it there. This
   centre leakage is a few per cent: for example, 3.5% in U for beam 0 of the
   943 MHz cube at 800 MHz. `keep_centre_leakage` (`--keep-centre-leakage`)
   keeps it.
4. The linear leakage is rotated by `2α`, from the frame of the feeds into the
   frame of the input:

   ```
   L_Q' = L_Q cos 2α − L_U sin 2α
   L_U' = L_Q sin 2α + L_U cos 2α
   ```

   `L_V` is not rotated. `η` affects only the position, not this rotation: the Q
   and U of the input are referenced to the north of the boresight, carried
   across the field, and so are the feeds, so the two frames differ by `α`
   everywhere.

5. The leakage is subtracted: `Q −= L_Q' I`, `U −= L_U' I`, `V −= L_V I`.
6. Q and U are rotated into the frame of the mosaic (see
   [polarisation](polarisation.md)).
7. Q, U and V are weighted by the Stokes I beam, and cut off where it is below
   `pb_cutoff`, in the same way as Stokes I, and then combined.

### The Stokes I image

The Stokes I image of each beam is found by its pixel grid. It is either the
Stokes I image among the inputs, or one given with `--stokes-i-images`
(`stokes_i_images`), which is used to remove the leakage but is not mosaicked. A
beam without a Stokes I image is an error, reported before any work starts. The
Stokes parameter of every image is read from its STOKES axis.

With `--convolve`, the Stokes I images are convolved to the same beam as the
inputs, so that `L I` is subtracted at the resolution of Q.

### Cubes

The leakage is removed channel by channel. Each channel uses the beam and
leakage of its own plane, so changes in the leakage with frequency are followed
across the band. Channels at the frequencies of the cube's planes share their
positions in the cube, so these are calculated once per tile for all of them.
`patchwork-pipeline` reads the planes on the workers, so the cube must be
readable from every node.

## Compared with linmos

yandasoft's linmos also weights by a holography beam and removes its leakage.
These steps behave the same in both:

- the spatial reference of the cube;
- where the pattern is sampled, and its rotation by `α`, for `η = 0`;
- the removal of the centre leakage;
- the `2α` rotation of the linear leakage;
- the weighting of Q, U and V by the Stokes I beam.

These steps differ:

| Step                   | linmos                                 | patchwork                                                                        |
| ---------------------- | -------------------------------------- | -------------------------------------------------------------------------------- |
| Plane of a channel     | Nearest, flagged or not                | Nearest unflagged, with a warning beyond 2 MHz                                   |
| Scaling to the channel | `s = f_plane / f`                      | `s = f / f_plane`                                                                |
| Stokes I beam          | As stored                              | 1 at the beam centre, or as stored with `holography_scale="stored"`              |
| Per-input cutoff       | From the peak of each input's weight   | From the beam centre, or from its peak with `holography_scale="stored"`          |
| Leakage subtracted     | On the input grid, before regridding   | At each output pixel, after interpolating                                        |
| Q and U frames         | Combined as they are                   | Rotated into the mosaic, unless `input_frame="north"` and `output_frame="north"` |
| Stokes parameters      | From `.q.`, `.u.` or `.v.` in the name | From the STOKES axis                                                             |
| Stokes I images        | `stokesinames`, or the name with `.i.` | Matched by pixel grid, and convolved with the inputs by `--convolve`             |

The effect of each difference:

- Flagged planes: linmos gives NaN beams for channels at flagged planes, such as
  those from 1152 to 1295 MHz in the 1295 MHz cube.
- Scaling: on a plane the two scalings are the same. For beam 17 of the 943 MHz
  cube, at a channel 0.4 MHz from a plane, they differ by less than 0.001 within
  the 10% level of the beam.
- Stokes I beam: linmos mosaics of `inherent` inputs are fainter by the beam's
  value at the centre, by 0.6% for beam 17 at 944 MHz and by up to 3% with the
  1655 MHz cube. `corrected` inputs are only slightly reweighted.
- Per-input cutoff: the peak of a holography beam can be higher than its centre
  value, by up to 2.8% at 944 MHz, with a median of 0.5%. linmos therefore cuts
  slightly further in. For beam 17 at a 10% cutoff, linmos keeps 16,687 pixels
  and patchwork keeps 16,828.
- Order of subtraction: both steps are linear, so subtracting before or after
  interpolating differs only by how much the leakage changes within one
  interpolation kernel. The leakage changes on the scale of the cube's 2′
  pixels.

`holography_scale="stored"` removes the differences in the Stokes I beam and the
per-input cutoff, for comparison with mosaics that linmos makes with
`finalcutoff = 0`, as flint runs it. patchwork does not blank a mosaic by its
final weight (see [mosaicking](mosaicking.md)).

## Tests

The ASKAP cubes are too large for the test suite, which uses synthetic cubes:

- `tests/test_primary_beam.py::test_holography_sampling` checks that the beam
  and its leakage are read at the positions, rotation and scale given above, and
  as stored with `holography_scale="stored"`.
- `tests/test_polarisation.py::test_leakage_removal` and
  `tests/test_cube.py::test_leakage_removed_channel_by_channel` check that the
  leakage is removed from mosaics of images and of cubes.
- `tests/test_primary_beam.py::test_holography_scale_stored` checks that both
  scales keep the same pixels, and that their mosaics differ by the beam's value
  at the centre.

A separate check on beam 17 of the 943 MHz cube mosaicked an input holding a
known sky plus leakage taken from the cube's own Q, U and V planes. The
recovered Q, U and V agreed with the known sky to within 3e-8, the float32
precision of the values. Without the correction, the leakage would have added up
to 0.17, 0.36 and 0.11 Jy/beam to Q, U and V.

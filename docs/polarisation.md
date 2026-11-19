# Polarisation

Stokes I and V are scalars, and are mosaicked like any other image. Stokes Q and
U are not: together they give a polarisation angle

```
χ = ½ atan2(U, Q)
```

measured east of a reference direction on the sky. Q and U from different images
can only be combined once they are referenced to the same direction. patchwork
therefore mosaics Q and U together, rotating each input into the frame of the
mosaic, and can remove the leakage of Stokes I into Q, U and V.

```bash
patchwork mosaic beam*.{q,u}.fits --frame-centre 12.5 -81.0 --output-base field
```

This writes `field.q.mosaic.fits`, `field.u.mosaic.fits` and
`field.qu.weight.fits`. The Stokes parameter of each image is read from its
STOKES axis, and the Q and U images of each beam are paired by their pixel grid.
Q or U can be mosaicked alone, with a warning, but is then not rotated.

To rotate an input, patchwork has to know what its Q and U are referenced to:
here, north at the boresight of the observation, at RA 12.5°, Dec −81°. Without
that it refuses to mosaic Q and U (see the frame of each input, below).

## Frames

### The frame of the mosaic

By the IAU convention, the polarisation angle of a mosaic is measured from north
at each pixel, that is, from the direction towards the north celestial pole
along the local meridian. This is `output_frame="north"`, the default.

### The frame of an input

An interferometric image does not follow that convention. Its Q and U come from
the correlations of feeds with fixed directions, corrected by one rotation for
the whole image (the parallactic angle, or for ASKAP the angle of the third
axis). Every pixel is therefore referenced to one direction: north at a
reference position, carried across the field. On the sky this is the projection
of that north direction at each pixel, which is also the y axis of the image's
SIN grid to within 0.04° anywhere within 1.5° of its centre, at any declination
(tested in `tests/test_polarisation.py`).

### The difference between them

Away from the reference position, local north turns away from the reference
direction by about the east-west offset times `tan(Dec)`. For an ASKAP-like
footprint of 6×6 beams 1.05° apart, the largest error in polarisation angle from
treating every input as referenced to local north is:

| Dec  | Between beams (boresight frame) | Within 1.5° of a beam centre |
| ---- | ------------------------------- | ---------------------------- |
| −30° | 1.6°                            | 1.0°                         |
| −50° | 3.3°                            | 2.0°                         |
| −70° | 8.2°                            | 4.8°                         |
| −80° | 19°                             | 12°                          |
| −85° | 48°                             | 38°                          |

Q and U are rotated by twice these angles. Near the equator the frames agree to
about a degree, so mosaicking Q and U as scalars works there. Towards the south
celestial pole it does not.

## Method

For each input and each pixel of the mosaic:

1. Q and U, and Stokes I for the leakage, are interpolated on the input's own
   grid, in its own frame. That frame is smooth everywhere, including across a
   pole, so interpolating in it is valid.
2. The leakage is subtracted in that frame (see below).
3. (Q, U) is rotated by `2 (η_input − η_output)`, where `η` is the position
   angle, east of local north at that pixel, of a frame's reference direction
   (`patchwork.polarisation.frame_angle`). `η_output` is 0 for
   `output_frame="north"`.
4. Q and U are weighted by the Stokes I primary beam, in the same way as Stokes
   I, and combined.

`input_frame="north"` states that the inputs are already referenced to local
north. They are then not rotated.

## The frame of each input

An image's header does not say what its Q and U are referenced to, and the
answer differs between instruments, so patchwork has to be told. There are three
options, which apply to every input:

- `--frame-centre RA DEC` (`frame_centre`): north at that position, in degrees.
  For ASKAP this is the boresight of the observation (see below).
- `--input-frame north`: north at each pixel already.
- `--input-frame crval`: north at each input's own reference position (CRVAL),
  as an imager makes the image of a single pointing.

An input can also say what its frame is itself, with the keywords described
under frame keywords, below. If neither the options nor the keywords give the
frame of an input that has Q and U, patchwork stops with an error that lists
these choices, before it reads or writes anything. If both give the frame of an
input of Q and U, they must agree. A mosaic patchwork made records its frame in
these keywords, so rotating it again from some other frame would turn it twice.

`frame_centre` only goes with the default `input_frame="reference"`. Images that
are not rotated, of Stokes I or V, or of Q or U alone, are never refused: their
frame only orients holography beams. Options that disagree with their keywords
are used instead, and keywords that cannot be read are ignored, both with a
warning. When nothing gives their frame, it is north at the image's own
reference position.

### ASKAP beams

For ASKAP, the frame is north at the boresight of the observation, not at the
centre of each beam. The third axis rotates the phased-array feed to keep a
fixed angle on the sky at the boresight, the direction the dish points.
Off-axis, the beams see the feed directions carried across from there. FixMS
corrects the correlations of each beam by the angle of the third axis, which is
the same for all beams, so every beam image of a field is referenced to north at
the boresight. Treating each beam as referenced to its own centre, which is what
`--input-frame crval` does, gives the errors in the "between beams" column
above.

```bash
patchwork mosaic beam*.{q,u}.fits --frame-centre 12.5 -81.0
```

FixMS moves the phase centre of each beam's measurement set to the beam, and
keeps the original direction, the boresight, as the `PHASE_DIR` of the
`FIELD_OLD` table. For a single field the boresight is also the centre of the
footprint, which for the usual symmetric footprints is close to the mean
position of the 36 beam centres.

Preliminary comparisons of overlapping SPICE-RACS and POSSUM fields support
this: the polarisation angles of sources seen in two fields disagree in the way
that referencing each field to its own boresight predicts, and agree once that
is corrected. The sign of the correction has not yet been checked against an
off-axis source of known angle.

`frame_centre` also sets the orientation of holography beams, as do the frame
keywords. Their patterns and leakage are fixed to the feeds, so at a beam centre
they are rotated by the angle between north there and the north of the
boresight, in addition to `holography_alpha_rad`.

## Rotating images before mosaicking

`patchwork rotate` applies the same rotation to Q and U images or cubes on their
own grid, so other tools see them referenced to north at each pixel before any
mosaic is made. Each pair of Q and U on one pixel grid is turned from the frame
it was imaged in, given as `patchwork mosaic` takes it (the frame keywords
below, `--frame-centre`, or `--input-frame crval`), and written beside its input
with `.rot` inserted before `.fits` (`--suffix`), never over an input. The
outputs carry `POLFRAME NORTH`, so mosaicking them later needs no frame options,
and rotating them again is refused.

```bash
patchwork rotate beam*.{q,u}.cube.fits --frame-centre 12.5 -81.0 --threads 8
patchwork mosaic beam*.{q,u}.cube.rot.fits --output-base field
```

The angle of a pixel is the same for every channel, so a cube is read and
written once, in blocks of rows across all its channels, on `--threads` threads.
A mosaic of rotated images differs from one of the originals rotated in flight
only by the rounding of float32 and the 0.001 pixel tolerance of the
interpolated coordinates on each grid: about 1e-5 radians of angle near a pole.

## Frame keywords

patchwork records the frame of a mosaic's Q and U in the headers of its Q and U
images, and reads the same keywords from its inputs:

| Keyword    | Value                                                                   |
| ---------- | ----------------------------------------------------------------------- |
| `POLFRAME` | `NORTH`: angles from north at each pixel                                |
|            | `FIXED`: angles from north at `POLRA`, `POLDEC`, carried across the map |
| `POLRA`    | The RA of that position in degrees, with `FIXED`                        |
| `POLDEC`   | The Dec of that position in degrees, with `FIXED`                       |

A mosaic made with `output_frame="north"` gets `POLFRAME = 'NORTH'`, and one
made with `output_frame="centre"` gets `FIXED` at its reference position. The
weights do not get them, and neither does a mosaic of Q or U alone, because it
is not rotated.

The frame of an input of Q and U is read from its Q and U images: both must have
the keywords, or neither, and they must agree. The keywords of its I and V
images are not used. An input's keywords take the place of the frame options, so
a mosaic can be mosaicked again with no frame options at all. The keywords can
also be written into the beam images before mosaicking, for example
`POLFRAME = 'FIXED'` with the boresight as `POLRA` and `POLDEC`, and then no
frame options are needed.

## Several observations

`frame_centre` is one position for every input, which suits the beams of one
observation. To combine several observations, each with its own boresight,
either:

- write each observation's boresight into the keywords of its beams, and mosaic
  all of them at once, or
- mosaic each observation with its own `--frame-centre`, then mosaic those
  mosaics. Their keywords say they are referenced to north at each pixel, so the
  second step needs no frame options.

## Leakage

With `remove_leakage=True`, a holography primary beam and a Stokes I image for
each beam, the leakage of Stokes I into Q, U and V is removed:

```
Q -= L_Q I,   U -= L_U I,   V -= L_V I
```

The Stokes I images can be among the images, or given with `--stokes-i-images`
to remove the leakage without being mosaicked. `L` is the fractional leakage in
the Q, U and V planes of the holography cube, at the pixel's offset from the
beam centre, rotated with the feeds. The leakage is subtracted in the frame of
the input, before the input is rotated into the mosaic.
[Holography beams and leakage](holography.md) describes each step, and how they
compare with linmos.

The leakage planes of the ASKAP cubes are fractional (Q/I). Towards the edge of
the cube they exceed 1, where Stokes I is small, so the choice of `pb_cutoff`
matters for polarisation.

## Near the south celestial pole

North is undefined at a pole, and turns through a full circle around it. A
uniformly polarised region across the pole therefore shows Q and U rotating
around it in a north-referenced mosaic. That is the convention, not an error.
With `output_frame="centre"`, the mosaic is instead referenced to north at the
frame centre, carried across the field as an image's frame is, and remains
smooth across the pole.

Smoothing, regridding or source finding in Q and U assumes one frame over the
area being processed. Within a few degrees of the pole, a north-referenced
mosaic does not have one. patchwork convolves the inputs to a common beam before
mosaicking, in their own smooth frames, so that order is safe. Convolving or
regridding a north-referenced mosaic near the pole is not. Make that mosaic with
`output_frame="centre"`, or rotate it first. RM synthesis works pixel by pixel,
so it is not affected, although the angles it derotates to are in the frame of
the mosaic.

The reference frame (an input's own SIN grid) works at a pole. A fitted frame is
centred on the inputs in a plane tangent to the sky at their mean centre
(`patchwork.frame.fitted_centre`), rather than from their range of RA and Dec,
which fails when the inputs surround the pole. Use a zenithal projection (SIN,
TAN, ZEA, ARC) near a pole, not CAR or another cylindrical projection.

Positions, primary beam offsets and frame angles are all calculated from unit
vectors, and none of them is singular at a pole.

## Not handled

- Faraday rotation, and changes in the leakage within one image or one channel.
  Across the channels of a cube, the leakage follows the planes of the
  holography cube.
- Direction-dependent effects other than the primary beam and its leakage, and
  any wide-field polarimetric distortion beyond the second order in offset that
  the projected-north frame leaves.
- Circular feeds, and frames other than equatorial. A galactic mosaic of Q and U
  would need a further rotation into galactic north.

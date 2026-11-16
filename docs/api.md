# API and coordinate conventions

## Stable root API

| Name | Responsibility |
|---|---|
| `Panorama` | A declared stitched equirectangular array; produces perspective views |
| `Perspective` | Output dimensions, horizontal field of view, yaw, pitch, and roll |
| `Equirectangular` | Panorama dimensions and orientation |
| `View` | Extracted pixels, sampling mode, provenance, and preprocessing operations |
| `ViewTransform` | Forward/inverse point mapping and clipped box mapping |
| `Box2D` | Positive-area pixel-edge rectangle with prediction metadata |
| `SphericalRegion` | One box-defined spherical object, independent of rendering seam pieces |
| `Polygon2D` | A rendered boundary with metadata and a hole flag |
| `ProjectionMap` | Reusable NumPy coordinates from a panorama to a perspective |
| `MapCache` | Thread-safe LRU of maps, bounded by stored array bytes |
| `coverage_views` | A conservative full-sphere perspective layout |
| `cube_projections` | Six named 90-degree perspective projections |
| `to_cubemap` | A `Panorama` to six named NumPy arrays |
| `to_cubemap_cross` | A `Panorama` to one image with all six faces in a cross |
| `from_cubemap` | Six matching arrays to an equirectangular array |

Methods documented here and in the README form the first-release contract.
`ray360.geometry` contains coordinate utilities. Other helpers in implementation
modules are internal; the root API, `ray360.image`, and `ray360.experimental`
are the intended import locations. Experimental operations may change independently.

## Coordinates

Rays use x right, y up, z forward. Longitude zero is forward, positive longitude
turns right, and longitude wraps to [-180, 180). Latitude is positive upward.
Public angles and fields of view are degrees.

Positive yaw points right, positive pitch points up, and positive roll rotates
camera-right toward camera-down. Camera-to-world rotation is
`Ry(yaw) Rx(-pitch) Rz(-roll)`.

Continuous image coordinates describe edges. An image spans [0,W] by [0,H], and
its first pixel center is (0.5,0.5). Boxes use `(x1,y1,x2,y2)` edges. Dimensions
are positive integers. Perspective horizontal field of view lies strictly
between 0 and 180 degrees. Pixels are square; vertical field of view follows
from aspect ratio. Equirectangular images represent the full sphere even when
their pixel aspect ratio is not 2:1.

`project(rays)` and `unproject(xy)` return values and visibility. Coordinates outside
a perspective image may remain mathematically defined; their validity is false.
Finite panorama x coordinates wrap; y coordinates outside the sphere are invalid.
Nonfinite coordinates and zero/nonfinite rays raise `ValueError`.

## Preprocessing and validity

`Panorama.view(Perspective(...), interpolation="bilinear", cache=...)` extracts pixels.
`view.crop(x,y,width,height)`, `.resize(width,height)`, and `.letterbox(width,height)`
return new views. A crop must lie within the current image. Fractional crop origins
and fractional symmetric letterbox padding are retained in the transform.

Each operation resamples the current pixels. Repeated resizing can lose image
detail even though inverse coordinates remain accurate. Sampling defaults to the
view's existing interpolation setting; a method's `interpolation=` overrides it.
Nearest sampling preserves label IDs. Integer bilinear results round back to the
input dtype. Boolean and 64-bit integer bilinear inputs are rejected.

`view.transform.to_panorama(points)` maps current output xy into source panorama xy.
`from_panorama(points)` performs the reverse. Both accept shape (...,2) and return
`(xy, valid)`. Validity includes the original perspective content, crop history,
current image bounds, and padding. It does not encode application-specific missing
pixels or annotation visibility. Applications retain any keypoint visibility flags
and combine them with this geometric mask.

`view.map_boxes(boxes)` returns retained spherical regions in input order, omitting
boxes entirely outside content. For one-to-one correspondence, call
`view.transform.map_box(box)` for each prediction and retain its `None` results.

## Regions and rendering

`SphericalRegion(box, projection)` retains its source definition. This avoids the
ambiguous interior of a standalone spherical ring. Source perspective boxes must
be clipped to their image. Source panorama boxes may use unwrapped x at the seam,
but cannot span more than one revolution. Latitude bounds must lie in the image.

`contains(rays)` tests source-defined membership. `rasterize(target)` samples that
membership at target pixel centers in row chunks.

`polygons(target, samples_per_edge=32)` returns visible boundary pieces:

- Perspective-source boxes sample every edge. Panorama output unwraps longitude,
  splits the seam, and closes pole caps. Increase sampling for tighter boundary
  approximation. Perspective targets clip in 3D before perspective division.
- Panorama-source boxes use raster contours at the target resolution. This preserves
  wide regions, both poles, and rotated interiors. Subpixel features can disappear;
  increase output resolution when needed.

Combine polygon rings with even-odd filling. `is_hole` identifies raster holes.
`boxes(target)` encloses outer polygon components and excludes holes; these are
approximate image rectangles, not exact transformed detection boxes.

`region.box` holds the original or clipped prediction metadata. Arrays in polygons
and generated maps are marked read-only. Metadata dictionaries are shallow copies;
keep nested application metadata immutable when sharing objects.

## Persistence

Transforms and regions expose `to_dict()` and `from_dict()` with `schema_version=1`.
Unknown schema versions are rejected. The transform schema stores the source/view
projections, positive scale, offset, output size, and retained content bounds.
It intentionally does not represent arbitrary affine rotations or shears.

`View.provenance()` contains source ID, caller-supplied timestamp, interpolation,
and the transform. Timestamps have no implicit time base or decoder semantics.
Application code writes sidecars and retains any additional identifiers.

## Cubemap conversion

`to_cubemap(panorama, size=512, interpolation="bilinear", cache=...)` returns six
square arrays. The panorama's declared orientation determines where its pixels fall
on the world-aligned cube. To retain transforms and source metadata, call
`panorama.view(face)` with each projection from `cube_projections(size)` instead.

| Face | Center direction | Image right | Image up |
|---|---|---|---|
| front | +Z | +X | +Y |
| right | +X | -Z | +Y |
| back | -Z | -X | +Y |
| left | -X | +Z | +Y |
| up | +Y | +X | -Z |
| down | -Y | +X | +Z |

`to_cubemap_cross(panorama, size=512, interpolation="bilinear", cache=...)` renders
the same square faces in one array with `3 * size` rows and `4 * size` columns.
Each face keeps its orientation and touches its neighbor at the folded edge.

```text
             up
    left    front    right    back
            down
```

The output preserves the source dtype and channel dimensions. Empty tiles are
zero-filled. It reuses the same projection maps as `to_cubemap()` and renders
directly into the face tiles, with no full-size face images to assemble.

`from_cubemap(faces, target, interpolation="bilinear", chunk_rows=128)` returns an
array with `target.height` rows and `target.width` columns. The target must be
`Equirectangular` and may have its own rotation. All six names are required, and
faces must have identical square shapes and dtypes. Input dictionary order does
not affect conversion. No packed strip, cross layout, or external engine convention
is inferred.

Reverse conversion assigns each ray to its dominant cube axis, with ties resolved
in X, Y, Z order. Bilinear filter taps outside a face sample the neighboring face
at its nearest pixel. This avoids clamping every tap to the same face edge. It is
not a mipmap or antialiasing filter; downsampling can lose detail or alias. Coordinate
working arrays are processed in row chunks. The output image is allocated in full.

Both directions retain HW/HWC shape semantics and dtype. Nearest interpolation
preserves label IDs, including boolean and 64-bit integer values. Bilinear has the
same dtype restrictions as view extraction. Conversion does not preserve arbitrary
application metadata; construct a new `Panorama` with the target projection and
explicit source ID/timestamp when needed.

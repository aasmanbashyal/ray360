# Ray360

Perspective views and reversible prediction mapping for **already-stitched 360 images**.

Ray360 extracts model inputs from equirectangular NumPy arrays and retains the
geometry through crop, resize, and letterbox operations. Map points and predicted
boxes back to the panorama, including objects crossing its seam or covering a pole.

Python 3.11+. NumPy is the only required dependency. Version 0.1.0 is the focused
first-release API; the earlier development API has been removed.

## Install from this checkout

```bash
uv sync --locked                         # NumPy core
uv sync --locked --extra image           # Optional Pillow image helpers
uv sync --locked --group dev --extra image
```

To install into another project, build the wheel with `uv build --no-sources`,
then use `uv pip install /path/to/ray360-0.1.0-py3-none-any.whl` in its environment.
An index release is not required to use this checkout.

## Extract a view and map a prediction

```python
import numpy as np
from ray360 import Box2D, Panorama, Perspective

# Supplying Panorama explicitly declares that these pixels are already stitched.
panorama = Panorama(np.zeros((1024, 2048, 3), dtype=np.uint8), source_id="capture")
view = panorama.view(Perspective(960, 540, yaw=180, pitch=10, hfov=100))
model_view = view.crop(40, 20, 880, 500).letterbox(640, 640)

# Run any model in the application. Its predictions must use model_view pixels.
boxes = [Box2D(250, 240, 390, 400, label="person", score=0.93, track_id=7)]
regions = model_view.map_boxes(boxes)
for region in regions:
    polygons = region.polygons(panorama.projection)
    rectangles = region.boxes(panorama.projection)
    membership = region.rasterize(panorama.projection)

points, valid = model_view.transform.to_panorama([[320, 320], [10, 0]])
```

One `SphericalRegion` remains one object even when rendering produces two seam
pieces. `region.box` retains labels, scores, tracking IDs, and attributes. Exported
rectangles enclose sampled boundaries and are approximate. Use `rasterize()` for
membership at pixel centers.

Predictions entirely in preprocessing padding are discarded; partial boxes are
clipped. Point mapping returns a validity mask. Ignore coordinates marked false.
`model_view.valid` identifies retained pixels, including after a crop followed by
new padding. These masks describe geometry, not detector confidence or occlusion.

## Arrays, sampling, and repeated frames

HW and HWC arrays retain their dtype. Bilinear interpolation is the default for
images. Use `interpolation="nearest"` for class labels; the choice carries through
all subsequent view preprocessing unless explicitly overridden. Boolean and
64-bit integer arrays require nearest sampling.

```python
from ray360 import MapCache, coverage_views

cache = MapCache(max_bytes=64 * 1024 * 1024)
targets = coverage_views(width=320, height=320)
for view in panorama.views(targets, cache=cache):
    pass  # Process and release each view instead of retaining all images.
```

Reuse the cache and targets for subsequent panoramas with the same dimensions and
orientation. The default shared cache holds at most 128 MiB of coordinate maps.
Pass `cache=None` to disable caching. The budget covers cached maps, not caller
images, returned views, or temporary allocations. Arrays remain caller-owned.

`coverage_views()` defaults to 24 views covering the sphere. This does not
guarantee detection recall. Choose view size, field of view, and overlap using the
application's accuracy and latency measurements.

## Persistence and optional image helpers

`ViewTransform.to_dict()` and `.from_dict()` preserve the mapping without storing
pixels. `view.provenance()` adds source ID, optional timestamp, and interpolation.
`SphericalRegion.to_dict()` and `.from_dict()` preserve both a region's boundary
definition and its intended interior. Both schemas carry a version. Keep custom
annotation attributes JSON-compatible when writing them as JSON.

```python
from ray360.image import overlay, read_panorama, write_image

panorama = read_panorama("stitched.jpg")  # Explicit declaration of stitched geometry
view = panorama.view(Perspective(640, 360, yaw=180))
regions = view.map_boxes([Box2D(240, 100, 400, 260)])
write_image("mapped.png", overlay(panorama.image, regions, projection=panorama.projection))
```

Image loading returns uint8 RGB and preserves stored pixel orientation. Saving
refuses existing files unless `overwrite=True` is supplied.

## Cubemap conversion

```python
from ray360 import Equirectangular, cube_projections, from_cubemap, to_cubemap

faces = to_cubemap(panorama, size=512)  # dict of six HW or HWC arrays
restored = from_cubemap(faces, Equirectangular(2048, 1024))

# For predictions on cube faces, keep the ordinary View and its transform.
front = panorama.view(cube_projections(512)["front"])
regions = front.map_boxes([Box2D(200, 180, 300, 320)])
```

Faces are named `front`, `right`, `back`, `left`, `up`, and `down`. Each is square
with a 90-degree field of view. [The orientation table](docs/api.md#cubemap-conversion)
defines the pole faces. Bilinear conversion back to a panorama samples adjacent
faces at cube boundaries. Use `interpolation="nearest"` in both directions for
label arrays. Conversion preserves dtype and can lose detail through resampling.
Atlas layouts and raw-camera stitching remain application concerns.

## Scope and experimental operations

The package owns projection geometry, cubemap conversion, preprocessing transforms,
box/point mapping, and map caching. Applications own inference, video decoding, timestamps, dataset
formats, and encoding. Raw-camera stitching, fisheye calibration, tensor backends,
network streaming, and segmentation/pose annotation codecs are outside this release.

`ray360.experimental` contains sampled spherical IoU and class-aware NMS. They
are outside the stable root API. Small objects can miss every sample and receive
zero overlap, even against themselves. NMS suppresses predictions; it does not
fuse partial detections or establish detector accuracy. See [scope](docs/scope.md).

## Examples and development

```bash
uv run --no-sync python examples/basic.py
uv run --no-sync python -m pytest
uv run --no-sync ruff check src tests scripts examples benchmarks
uv run --no-sync mypy
uv build --no-sources
uv run --no-sync python scripts/check_wheel.py
```

[The basic example](examples/basic.py) needs only NumPy. It can write a synthetic
overlay with `--output .test-results/demo.png` when Pillow is installed.
[The model example](examples/detect.py) and [decoder example](examples/video_frames.py)
show application-owned integrations with separately installed dependencies.
The opt-in [end-to-end suite](docs/validation.md#real-media-and-model-validation)
executes these examples and runs cube conversion and CPU detection on every frame
of two real panorama video encodings. It also checks two stitched photos and
photographic cube faces. Media and model weights are checksum-verified local fixtures.

See [API conventions](docs/api.md), [validation](docs/validation.md), and
[whole-panorama benchmarks](docs/benchmarks.md). The ordinary test suite never
downloads media or model weights.

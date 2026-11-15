# Ray360

Perspective views and prediction mapping for already-stitched 360 images.

Ray360 uses NumPy arrays and preserves the geometry through crop, resize, and
letterbox preprocessing. Map predicted points and boxes back into a panorama,
including its seam and poles. NumPy is the only required dependency; Pillow
provides optional image helpers. Python 3.11 or newer is required.

```bash
uv sync --locked --group dev --extra image
uv run --no-sync python -m pytest
uv run --no-sync mypy
uv build --no-sources
```

```python
import numpy as np
from ray360 import Box2D, Panorama, Perspective

panorama = Panorama(np.zeros((1024, 2048, 3), dtype=np.uint8))
view = panorama.view(Perspective(640, 360, yaw=180)).letterbox(640, 640)
regions = view.map_boxes([Box2D(240, 240, 400, 400, label="person")])
points, valid = view.transform.to_panorama([[320, 320]])
```

Use nearest interpolation for label arrays. Point mappings return validity masks;
box mappings clip padding and discard boxes entirely outside content. Applications
own model inference, video decoding, and dataset formats. Experimental sampled
spherical IoU and NMS are available under `ray360.experimental`.

The source is distributed under the MIT license.

## Cubemap conversion

```python
from ray360 import Equirectangular, cube_projections, from_cubemap, to_cubemap

faces = to_cubemap(panorama, size=512)
restored = from_cubemap(faces, Equirectangular(2048, 1024))
front = panorama.view(cube_projections(512)["front"])
```

Faces are named front, right, back, left, up, and down. All faces have 90-degree
fields of view and zero roll. Bilinear conversion samples adjacent faces at shared
edges. Use nearest interpolation in both directions for label arrays.

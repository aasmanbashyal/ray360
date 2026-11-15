"""A synthetic stitched panorama, preprocessing, and a seam-crossing prediction.

uv run --no-sync python examples/basic.py
uv run --no-sync python examples/basic.py --output .test-results/demo.png
"""

import argparse
import json

import numpy as np

from ray360 import Box2D, Equirectangular, Panorama, Perspective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional panorama overlay; requires the image extra")
    args = parser.parse_args()
    projection = Equirectangular(720, 360)
    rays, _ = projection.unproject(projection.pixel_grid())
    image = np.rint((rays + 1) * 127.5).astype(np.uint8)
    panorama = Panorama(image, source_id="synthetic")
    view = panorama.view(Perspective(320, 180, yaw=180)).letterbox(320, 320)
    region = view.map_boxes([Box2D(90, 100, 230, 220, label="example", score=0.9)])[0]
    print(
        json.dumps(
            {
                "view_shape": list(view.image.shape),
                "panorama_components": len(region.polygons(projection)),
                "provenance": view.provenance(),
            },
            indent=2,
        )
    )
    if args.output:
        from ray360.image import overlay, write_image

        write_image(args.output, overlay(image, [region], projection=projection))


if __name__ == "__main__":
    main()

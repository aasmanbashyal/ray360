"""Optional application example using an existing local Ultralytics model.

Install Ultralytics in the application environment. It is not a ray360 extra.
uv run --with ultralytics --extra image python examples/detect.py panorama.jpg local-model.pt output.png

Each result maps back independently; this example does not merge objects or claim
detector accuracy. Model loading and inference belong to the application.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ray360 import Box2D, MapCache, coverage_views
from ray360.image import overlay, read_panorama, write_image


def detect_panorama(
    panorama, model, targets, *, image_size=320, confidence=0.25, device="cpu", cache=None
):
    """Return mapped regions and auditable per-view predictions. The caller owns the model."""
    if model.task != "detect":
        raise ValueError("This example requires an object-detection checkpoint.")
    cache = MapCache() if cache is None else cache
    regions, records = [], []
    for view_index, view in enumerate(panorama.views(targets, cache=cache)):
        view = view.letterbox(image_size, image_size)
        # Ultralytics ndarray input is BGR; xyxy is in this input image's pixels.
        result = model.predict(
            np.ascontiguousarray(view.image[..., ::-1]),
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        for xyxy, score, category in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            strict=True,
        ):
            box = Box2D(*xyxy, label=result.names[int(category)], score=float(score))
            region = view.transform.map_box(box)
            if region is not None:
                regions.append(region)
                records.append(
                    {
                        "view_index": view_index,
                        "box": box.to_dict(),
                        "provenance": view.provenance(),
                        "region": region.to_dict(),
                    }
                )
    return regions, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panorama")
    parser.add_argument("weights", type=Path)
    parser.add_argument("output")
    parser.add_argument("--json", type=Path, help="Write predictions and their transforms")
    parser.add_argument("--view-size", type=int, default=640)
    parser.add_argument(
        "--view-height", type=int, help="Defaults to view-size; shorter views are letterboxed"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error("weights must be an existing local checkpoint")
    if args.view_size < 32 or args.view_size % 32:
        parser.error("view-size must be a positive multiple of 32")
    if not 0 <= args.confidence <= 1:
        parser.error("confidence must be between 0 and 1")
    if args.view_height is not None and args.view_height <= 0:
        parser.error("view-height must be positive")
    if Path(args.output).exists() or (args.json is not None and args.json.exists()):
        parser.error("output files already exist")
    if args.json is not None and args.json.resolve() == Path(args.output).resolve():
        parser.error("image and JSON outputs must be different files")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    panorama = read_panorama(args.panorama)
    if model.task != "detect":
        parser.error("weights must be an object-detection checkpoint")
    targets = coverage_views(width=args.view_size, height=args.view_height or args.view_size)
    regions, records = detect_panorama(
        panorama,
        model,
        targets,
        image_size=args.view_size,
        confidence=args.confidence,
        device=args.device,
    )
    write_image(args.output, overlay(panorama.image, regions, projection=panorama.projection))
    report = {
        "source": args.panorama,
        "view_count": len(targets),
        "detection_count": len(records),
        "predictions": records,
    }
    if args.json is not None:
        with args.json.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}))


if __name__ == "__main__":
    main()

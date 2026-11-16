"""Rebuild the README figures from the checksum-verified public panorama fixture.

Run scripts/fetch_test_data.py first, then:
uv run --no-sync python examples/readme_gallery.py

The outlined boxes illustrate geometry; they are not model detections.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ray360 import Box2D, Panorama, Perspective, to_cubemap_cross
from ray360.image import overlay

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#101a2b"
FOREGROUND = "#edf4ff"
SECONDARY = "#adbed5"
COLORS = [(86, 218, 219), (255, 190, 101), (197, 161, 255)]


def font(size):
    candidates = [
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text(canvas, xy, content, size=22, color=FOREGROUND):
    ImageDraw.Draw(canvas).text(xy, content, fill=color, font=font(size))


def canvas(height, title, subtitle):
    image = Image.new("RGB", (1200, height), BACKGROUND)
    text(image, (24, 16), title, 30)
    text(image, (24, 60), subtitle, 19, SECONDARY)
    return image


def photo(array, size=None):
    image = Image.fromarray(array)
    return image if size is None else image.resize(size, Image.Resampling.LANCZOS)


def save(image, output, filename):
    text(
        image,
        (24, image.height - 30),
        "Jordan Pond: Matthew Petroff / Pannellum. See docs/assets/README.md.",
        16,
        SECONDARY,
    )
    path = output / filename
    image.save(path, format="WEBP", quality=86, method=6)
    print(f"{path.name}: {image.width} x {image.height}, {path.stat().st_size:,} bytes")


def views_figure(panorama, output):
    image = canvas(
        1010,
        "One panorama, multiple perspectives",
        "Colored outlines mark the pixels used by each perspective view.",
    )
    mapped = panorama.image
    targets = [(180, "Across the seam"), (-90, "Looking left"), (0, "Looking forward")]
    for index, (yaw, label) in enumerate(targets):
        view = panorama.view(Perspective(368, 230, yaw=yaw, hfov=90))
        region = view.map_boxes([Box2D(0, 0, 368, 230)])[0]
        mapped = overlay(
            mapped, [region], projection=panorama.projection, color=COLORS[index], width=7
        )
        x = 24 + index * 392
        text(image, (x, 714), f"{label} / yaw {yaw}", 20, COLORS[index])
        image.paste(photo(view.image), (x, 746))
    image.paste(photo(mapped, (1152, 576)), (24, 112))
    save(image, output, "perspective-views.webp")


def seam_figure(panorama, output):
    image = canvas(
        1110,
        "One prediction, two panorama pieces",
        "A demonstration box mapped through a letterboxed view at yaw 180 degrees.",
    )
    view = panorama.view(Perspective(560, 315, yaw=180, hfov=90)).letterbox(560, 360)
    box = Box2D(160, 100, 400, 230, label="demo region")
    region = view.map_boxes([box])[0]
    assert len(region.polygons(panorama.projection)) == 2
    image.paste(photo(overlay(view.image, [box], color=COLORS[0], width=4)), (24, 108))
    text(image, (620, 132), "Perspective model input", 26)
    text(image, (620, 182), "The dark bands are letterbox padding.", 21, SECONDARY)
    text(image, (620, 242), "One Box2D becomes one SphericalRegion.", 21)
    text(image, (620, 286), "It renders on both sides of the seam.", 21, SECONDARY)
    text(image, (620, 358), "Illustrative box, not a model detection.", 19, COLORS[0])
    mapped = overlay(
        panorama.image, [region], projection=panorama.projection, color=COLORS[0], width=8
    )
    image.paste(photo(mapped, (1152, 576)), (24, 478))
    save(image, output, "seam-mapping.webp")


def cube_figure(panorama, output):
    size = 288
    x, y = 24, 106
    image = canvas(
        y + 3 * size + 48,
        "A panorama unfolded into a cube cross",
        "Each square is a 90-degree view of the same panorama.",
    )
    image.paste(photo(to_cubemap_cross(panorama, size=size)), (x, y))
    draw = ImageDraw.Draw(image)
    label_font = font(22)
    for name, row, col in (
        ("Top", 0, 1),
        ("Left", 1, 0),
        ("Front", 1, 1),
        ("Right", 1, 2),
        ("Back", 1, 3),
        ("Bottom", 2, 1),
    ):
        draw.text(
            (x + col * size + 12, y + row * size + 8),
            name,
            font=label_font,
            fill=FOREGROUND,
            stroke_width=2,
            stroke_fill=BACKGROUND,
        )
    save(image, output, "cubemap-faces.webp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / ".test-data")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "tests/data-sources.json").read_text())
    entry = next(item for item in manifest["entries"] if item["id"] == "pannellum-panorama")
    path = args.data / entry["path"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError("Panorama checksum does not match the documented source")
    with Image.open(path) as original:
        pixels = np.array(original.convert("RGB").resize((2304, 1152), Image.Resampling.LANCZOS))
    panorama = Panorama(pixels, source_id=entry["id"])
    args.output.mkdir(parents=True, exist_ok=True)
    views_figure(panorama, args.output)
    seam_figure(panorama, args.output)
    cube_figure(panorama, args.output)


if __name__ == "__main__":
    main()

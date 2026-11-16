# Changelog

## Unreleased

## 0.2.0 · 2026-11-16

- Added `from_cubemap_cross()` to reconstruct a panorama and map boxes across cube faces and the panorama seam.
- Added `to_cubemap_cross()` to render all six oriented faces in one image.
- Improved installation, examples, and navigation in the README.
- Added reproducible perspective, seam-mapping, and cubemap figures with source credits.
- Added contribution and security reporting guidance, issue forms, and a pull request template.

## 0.1.0 · 2026-11-15

First public release on [PyPI](https://pypi.org/project/ray360/0.1.0/).

- Perspective extraction and full-sphere view coverage for stitched equirectangular arrays.
- Crop, resize, and letterbox transforms with inverse point and box mapping.
- Spherical regions with seam-aware polygons, approximate boxes, and rasterized membership.
- Conversion between equirectangular panoramas and six named cube faces, including adjacent-face filtering.
- Bounded coordinate-map caching and JSON-compatible geometry persistence.
- Optional Pillow image loading, saving, and outline overlays.
- Application-owned model and video decoder examples with opt-in real-media validation.

Requires Python 3.11 or newer. NumPy is the only required runtime dependency;
Pillow is available through the `image` extra. Sampled spherical IoU and NMS live
in `ray360.experimental` and are outside the supported root API.

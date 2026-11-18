# Changelog

## 0.3.1 · 2026-11-18

- A failed image encode now preserves the previous file and leaves a new filename unused.
- Cubemap outlines on 16-bit RGB and RGBA images now render as 8-bit RGB previews.
- Reverse conversion of a contiguous `uint8` cubemap cross now uses the optional Rust sampler.
- Fixed incorrect batch ray rotations observed under Linux ARM64 emulation with OpenBLAS.

## 0.3.0 · 2026-11-17

- Added an optional Rust sampler for contiguous `uint8` images in the standard view, preprocessing, and cubemap APIs.
- The optional Rust extension is built from the repository. It also provides `PreparedView` for reusing a single sampling plan across frames.
- The PyPI package continues to work with NumPy alone and uses NumPy for inputs outside the Rust sampler's supported formats.
- The Python source distribution omits the README gallery images, which remain available on GitHub.

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

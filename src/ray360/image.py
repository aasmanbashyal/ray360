"""Optional Pillow loading, saving and outline overlays for stitched images."""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .annotations import Box2D, Polygon2D, SphericalRegion
from .panorama import Panorama
from .projections import Equirectangular, Projection


def _pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image helpers require Pillow; install 'ray360[image]'.") from exc
    return Image


def read_panorama(path: str | Path, *, projection: Equirectangular | None = None) -> Panorama:
    """Read uint8 RGB pixels. Calling this explicitly declares the input stitched.

    EXIF orientation is not applied: the supplied projection refers to stored pixels.
    Use arrays directly for label images or high-bit-depth data.
    """
    with _pillow().open(path) as image:
        array = np.array(image.convert("RGB"))
    return Panorama(array, projection, source_id=str(path))


def write_image(path: str | Path, image: NDArray, *, overwrite: bool = False) -> None:
    """Encode before replacing the destination. Existing files require overwrite=True."""
    encoded = _pillow().fromarray(np.asarray(image))
    path = Path(path)
    extension = path.suffix.lower()
    formats = _pillow().registered_extensions()
    if extension not in formats:
        raise ValueError(f"Unknown image extension {extension!r}.")
    if not overwrite and os.path.lexists(path):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
    if overwrite:
        path = path.resolve()
        if path.exists() and not path.is_file():
            # Pipes and devices must be opened in place, never replaced by a file.
            with path.open("wb") as stream:
                encoded.save(stream, format=formats[extension])
            return
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory, "image")
        with staged.open("wb") as stream:
            encoded.save(stream, format=formats[extension])
        if overwrite:
            if path.is_file():
                shutil.copymode(path, staged)
            staged.replace(path)
        elif os.name == "nt":
            staged.rename(path)  # Windows rename refuses an existing destination.
        else:
            try:
                os.link(staged, path)
            except OSError as exc:
                if exc.errno not in (errno.ENOTSUP, errno.EPERM, errno.ENOSYS):
                    raise
                # Filesystems without hard links still support exclusive creation.
                with path.open("xb") as output, staged.open("rb") as source:
                    shutil.copyfileobj(source, output)


def overlay(
    image: NDArray,
    annotations: Iterable[Box2D | Polygon2D | SphericalRegion],
    *,
    projection: Projection | None = None,
    color: tuple[int, int, int] = (255, 60, 40),
    width: int = 2,
) -> NDArray:
    """Return RGB pixels with outlines. Spherical regions require a target projection."""
    from .geometry import positive_int

    positive_int(width, "width")
    Image = _pillow()
    from PIL import ImageDraw

    pixels = np.asarray(image)
    if (
        pixels.ndim == 3
        and pixels.shape[-1] in (3, 4)
        and pixels.dtype.kind == "u"
        and pixels.dtype.itemsize == 2
    ):
        # Pillow's RGB display mode uses eight bits per channel.
        pixels = (pixels >> 8).astype(np.uint8)
    base = Image.fromarray(pixels).convert("RGB")
    if projection is not None and projection.size != base.size:
        raise ValueError("Overlay projection must match image dimensions.")
    draw = ImageDraw.Draw(base)
    for annotation in annotations:
        if isinstance(annotation, SphericalRegion):
            if projection is None:
                raise ValueError("Spherical overlays require a target projection.")
            polygons = annotation.polygons(projection)
        elif isinstance(annotation, Box2D):
            polygons = [annotation.polygon()]
        elif isinstance(annotation, Polygon2D):
            polygons = [annotation]
        else:
            raise TypeError("Expected a box, rendered polygon or spherical region.")
        for polygon in polygons:
            points = [tuple(p) for p in polygon.vertices.tolist()]
            draw.line(points + points[:1], fill=color, width=width)
    return np.array(base)

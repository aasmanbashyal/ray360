"""Optional Pillow loading, saving and outline overlays for stitched images."""

from __future__ import annotations

import errno
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable
from contextlib import suppress
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


if sys.platform == "win32":
    import msvcrt
    from typing import BinaryIO

    def _copy_in_place(source: BinaryIO, target: BinaryIO) -> None:
        target.seek(0)
        shutil.copyfileobj(source, target)
        if target.tell() != source.tell():
            raise OSError(errno.EIO, "Incomplete write", target.name)
        target.truncate()

    def _overwrite_open_file(staged: Path, path: Path) -> None:
        # A Windows reader can share write access while blocking a rename.
        # Unbuffered writes let us seek back even after a disk-full error.
        with staged.open("rb") as source, path.open("r+b", buffering=0) as target:
            # Serialize in-place writers. Closing the handle releases the lock.
            msvcrt.locking(target.fileno(), msvcrt.LK_NBLCK, 1)
            with staged.with_name("backup").open("w+b") as backup:
                shutil.copyfileobj(target, backup)
                backup.seek(0)
                try:
                    _copy_in_place(source, target)
                except BaseException:
                    _copy_in_place(backup, target)
                    raise


def _copy_exclusive(staged: Path, path: Path) -> None:
    with staged.open("rb") as source:
        output = path.open("xb")
        created = os.fstat(output.fileno())
        try:
            with output:
                shutil.copyfileobj(source, output)
        except BaseException:
            # A writer may have replaced our directory entry during the copy.
            with suppress(FileNotFoundError):
                if os.path.samestat(created, path.lstat()):
                    path.unlink()
            raise


def write_image(path: str | Path, image: NDArray, *, overwrite: bool = False) -> None:
    """Existing files require overwrite=True. Encode regular files before publishing.

    When a Windows reader blocks replacement but shares write access, update the
    open file in place. Readers can observe that update before it finishes.
    """
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
            try:
                staged.replace(path)
            except PermissionError as exc:
                if sys.platform == "win32" and exc.winerror in (5, 32) and path.is_file():
                    _overwrite_open_file(staged, path)
                else:
                    raise
        elif sys.platform == "win32":
            staged.rename(path)  # Windows rename refuses an existing destination.
        else:
            try:
                os.link(staged, path)
            except OSError as exc:
                if exc.errno not in (errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM, errno.ENOSYS):
                    raise
                _copy_exclusive(staged, path)


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

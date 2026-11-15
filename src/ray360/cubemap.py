"""Six named 90-degree cube faces, with filtering across shared edges."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .geometry import positive_int
from .panorama import Panorama
from .projections import Equirectangular, Perspective
from .remap import DEFAULT_CACHE, Interpolation, MapCache, sample, validate_image

_ORIENTATIONS = {
    "front": (0, 0),
    "right": (90, 0),
    "back": (180, 0),
    "left": (-90, 0),
    "up": (0, 90),
    "down": (0, -90),
}


def cube_projections(size: int) -> dict[str, Perspective]:
    """World-aligned square faces: front +Z, right +X, back -Z, left -X, up +Y, down -Y.

    All faces have zero roll. The top of the up face points toward -Z;
    the top of the down face points toward +Z. No atlas layout is assumed.
    """
    positive_int(size, "size")
    return {
        name: Perspective(size, size, yaw=yaw, pitch=pitch, hfov=90)
        for name, (yaw, pitch) in _ORIENTATIONS.items()
    }


def to_cubemap(
    panorama: Panorama,
    size: int = 512,
    *,
    interpolation: Interpolation = "bilinear",
    cache: MapCache | None = DEFAULT_CACHE,
) -> dict[str, NDArray]:
    """Convert a declared panorama into six named, world-aligned arrays.

    To retain prediction transforms, use panorama.view on cube_projections instead.
    """
    if not isinstance(panorama, Panorama):
        raise TypeError("to_cubemap requires a Panorama.")
    return {
        name: panorama.view(projection, interpolation=interpolation, cache=cache).image
        for name, projection in cube_projections(size).items()
    }


def _owners(rays: NDArray) -> NDArray:
    # Ties prefer X, then Y, then Z. Each ray belongs to exactly one face.
    axis = np.argmax(np.abs(rays), axis=-1)
    positive = np.take_along_axis(rays, axis[..., None], axis=-1)[..., 0] >= 0
    return np.where(positive, np.array([1, 4, 0])[axis], np.array([3, 5, 2])[axis])


def _nearest(
    faces: Mapping[str, NDArray], projections: Mapping[str, Perspective], rays: NDArray
) -> NDArray:
    first = faces["front"]
    output = np.empty((*rays.shape[:-1], *first.shape[2:]), dtype=first.dtype)
    owners = _owners(rays)
    for index, (name, projection) in enumerate(projections.items()):
        chosen = owners == index
        if chosen.any():
            xy, _ = projection.project(rays[chosen])
            output[chosen] = sample(faces[name], xy, interpolation="nearest")
    return output


def _face_sample(
    faces: Mapping[str, NDArray],
    projections: Mapping[str, Perspective],
    name: str,
    xy: NDArray,
    interpolation: Interpolation,
) -> NDArray:
    face = faces[name]
    output = sample(face, xy, interpolation=interpolation)
    if interpolation == "nearest":
        return output
    size = face.shape[0]
    border = ((xy < 0.5) | (xy > size - 0.5)).any(-1)
    if not border.any():
        return output
    # Only the edge samples need cross-face taps. Interior pixels use the common sampler.
    centers = xy[border] - 0.5
    lower = np.floor(centers)
    fraction = centers - lower
    dtype = np.result_type(face.dtype, np.float32)
    filtered = np.zeros((len(centers), *face.shape[2:]), dtype=dtype)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        taps = lower + [dx + 0.5, dy + 0.5]
        rays, _ = projections[name].unproject(taps)
        values = _nearest(faces, projections, rays).astype(dtype)
        weight = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (
            fraction[:, 1] if dy else 1 - fraction[:, 1]
        )
        if face.ndim == 3:
            weight = weight[:, None]
        filtered += values * weight.astype(dtype)
    if face.dtype.kind in "ui":
        limits = np.iinfo(face.dtype)
        filtered = np.clip(np.rint(filtered), limits.min, limits.max)
    output[border] = filtered.astype(face.dtype)
    return output


def from_cubemap(
    faces: Mapping[str, NDArray],
    target: Equirectangular,
    *,
    interpolation: Interpolation = "bilinear",
    chunk_rows: int = 128,
) -> NDArray:
    """Convert six matching square faces to a full equirectangular array.

    Bilinear taps crossing a face edge sample the adjacent face. Nearest sampling
    preserves class IDs. Face orientation follows cube_projections, including poles.
    """
    if not isinstance(target, Equirectangular):
        raise TypeError("from_cubemap requires an Equirectangular target.")
    positive_int(chunk_rows, "chunk_rows")
    if set(faces) != set(_ORIENTATIONS):
        raise ValueError("Expected exactly front, right, back, left, up, and down faces.")
    arrays = {name: validate_image(array) for name, array in faces.items()}
    first = arrays["front"]
    if first.shape[0] != first.shape[1]:
        raise ValueError("Cubemap faces must be square.")
    if any(array.shape != first.shape or array.dtype != first.dtype for array in arrays.values()):
        raise ValueError("Cubemap faces must have matching shapes and dtypes.")
    projections = cube_projections(first.shape[0])
    output = np.empty((target.height, target.width, *first.shape[2:]), dtype=first.dtype)
    for start in range(0, target.height, chunk_rows):
        stop = min(target.height, start + chunk_rows)
        rays, _ = target.unproject(target.pixel_grid(start, stop))
        owners = _owners(rays)
        for index, (name, projection) in enumerate(projections.items()):
            chosen = owners == index
            xy, _ = projection.project(rays[chosen])
            output[start:stop][chosen] = _face_sample(arrays, projections, name, xy, interpolation)
    return output

"""Spherical coordinates. Rays use x=right, y=up, z=forward."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def normalize(vectors: ArrayLike) -> FloatArray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("Expected vectors with shape (..., 3).")
    scale = np.max(np.abs(values), axis=-1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(scale == 0):
        raise ValueError("Rays must be finite and nonzero.")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def lonlat_to_rays(lonlat: ArrayLike) -> FloatArray:
    """Convert (..., 2) longitude/latitude in degrees to unit rays."""
    ll = np.asarray(lonlat, dtype=np.float64)
    if ll.ndim < 1 or ll.shape[-1] != 2 or not np.all(np.isfinite(ll)):
        raise ValueError("Expected finite longitude/latitude pairs.")
    if np.any(np.abs(ll[..., 1]) > 90):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    lon, lat = np.moveaxis(np.deg2rad(ll), -1, 0)
    return np.stack((np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)), -1)


def rays_to_lonlat(rays: ArrayLike) -> FloatArray:
    rays = normalize(rays)
    lon = np.rad2deg(np.arctan2(rays[..., 0], rays[..., 2]))
    lon = (lon + 180) % 360 - 180
    lat = np.rad2deg(np.arctan2(rays[..., 1], np.hypot(rays[..., 0], rays[..., 2])))
    return np.stack((lon, lat), -1)


def rotation_matrix(yaw: float = 0, pitch: float = 0, roll: float = 0) -> FloatArray:
    """Camera-to-world rotation: yaw right, pitch up, roll camera-right toward down."""
    if not np.all(np.isfinite([yaw, pitch, roll])):
        raise ValueError("Rotation angles must be finite.")
    y, p, r = np.deg2rad([yaw, pitch, roll])
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    rz = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


def angular_distance(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Great-circle angular distance in degrees, with NumPy broadcasting."""
    a, b = normalize(a), normalize(b)
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, -1)))


def fibonacci_sphere(samples: int = 65536) -> FloatArray:
    """Deterministic approximately equal-area samples, useful for spherical IoU."""
    positive_int(samples, "samples")
    index = np.arange(samples, dtype=np.float64)
    y = 1 - 2 * (index + 0.5) / samples
    lon = index * (np.pi * (3 - np.sqrt(5)))
    r = np.sqrt(1 - y * y)
    return np.stack((r * np.sin(lon), y, r * np.cos(lon)), -1)


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def points2d(points: ArrayLike) -> FloatArray:
    xy = np.asarray(points, dtype=np.float64)
    if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite coordinates with shape (..., 2).")
    return xy

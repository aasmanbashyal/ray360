"""Projection definitions. Continuous pixel edges span [0, width] × [0, height]."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .geometry import (
    FloatArray,
    _rotate_rays,
    lonlat_to_rays,
    normalize,
    points2d,
    positive_int,
    rays_to_lonlat,
    rotation_matrix,
)


@dataclass(frozen=True)
class _Projection(ABC):
    width: int
    height: int
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    kind: ClassVar[str] = "base"

    def __post_init__(self) -> None:
        positive_int(self.width, "width")
        positive_int(self.height, "height")
        rotation_matrix(self.yaw, self.pitch, self.roll)

    @property
    def rotation(self) -> FloatArray:
        return rotation_matrix(self.yaw, self.pitch, self.roll)

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def pixel_grid(self, start: int = 0, stop: int | None = None) -> FloatArray:
        stop = self.height if stop is None else stop
        x, y = np.meshgrid(np.arange(self.width) + 0.5, np.arange(start, stop) + 0.5)
        return np.stack((x, y), -1)

    def contains_pixels(self, xy: ArrayLike) -> np.ndarray:
        xy = np.asarray(xy)
        return (
            np.isfinite(xy).all(-1)
            & (xy[..., 0] >= -1e-8)
            & (xy[..., 0] <= self.width + 1e-8)
            & (xy[..., 1] >= -1e-8)
            & (xy[..., 1] <= self.height + 1e-8)
        )

    @abstractmethod
    def unproject(self, xy: ArrayLike) -> tuple[FloatArray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def project(self, rays: ArrayLike) -> tuple[FloatArray, np.ndarray]:
        raise NotImplementedError


@dataclass(frozen=True)
class Equirectangular(_Projection):
    kind: ClassVar[str] = "equirectangular"

    def unproject(self, xy: ArrayLike) -> tuple[FloatArray, np.ndarray]:
        xy = points2d(xy)
        ll = np.stack(
            ((xy[..., 0] / self.width - 0.5) * 360, (0.5 - xy[..., 1] / self.height) * 180), -1
        )
        valid = np.isfinite(ll).all(-1) & (np.abs(ll[..., 1]) <= 90 + 1e-8)
        safe = np.nan_to_num(ll)
        safe[..., 1] = np.clip(safe[..., 1], -90, 90)
        return _rotate_rays(lonlat_to_rays(safe), self.rotation.T), valid

    def project(self, rays: ArrayLike) -> tuple[FloatArray, np.ndarray]:
        ll = rays_to_lonlat(_rotate_rays(rays, self.rotation))
        xy = np.stack(
            ((ll[..., 0] / 360 + 0.5) * self.width, (0.5 - ll[..., 1] / 180) * self.height), -1
        )
        return xy, np.ones(xy.shape[:-1], dtype=bool)


@dataclass(frozen=True)
class Perspective(_Projection):
    hfov: float = 90.0
    kind: ClassVar[str] = "perspective"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.hfov < 180:
            raise ValueError("Perspective hfov must be between 0 and 180 degrees (exclusive).")

    @property
    def focal(self) -> float:
        return float(self.width / (2 * np.tan(np.deg2rad(self.hfov) / 2)))

    @property
    def vfov(self) -> float:
        return float(np.rad2deg(2 * np.arctan(self.height / (2 * self.focal))))

    def unproject(self, xy: ArrayLike) -> tuple[FloatArray, np.ndarray]:
        xy = points2d(xy)
        local = np.stack(
            (
                (xy[..., 0] - self.width / 2) / self.focal,
                (self.height / 2 - xy[..., 1]) / self.focal,
                np.ones(xy.shape[:-1]),
            ),
            -1,
        )
        return _rotate_rays(normalize(local), self.rotation.T), self.contains_pixels(xy)

    def project(self, rays: ArrayLike) -> tuple[FloatArray, np.ndarray]:
        local = _rotate_rays(normalize(rays), self.rotation)
        z = local[..., 2]
        safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
        xy = np.stack(
            (
                self.width / 2 + self.focal * local[..., 0] / safe_z,
                self.height / 2 - self.focal * local[..., 1] / safe_z,
            ),
            -1,
        )
        return xy, (z > 1e-12) & self.contains_pixels(xy)


def coverage_views(
    width: int = 640, height: int = 640, hfov: float = 90, overlap: float = 0.2
) -> list[Perspective]:
    """Cover the full sphere, including poles, using conservative inscribed view cones.

    overlap is a safety margin in angular center spacing; rectangular-view overlap
    varies with latitude. The default generates 24 views. This is geometric
    coverage, not a guarantee that an object detector sees every object.
    """
    view = Perspective(width, height, hfov=hfov)
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    step = min(hfov, view.vfov) / np.sqrt(2) * (1 - overlap)
    intervals = int(np.ceil(180 / step))
    lat_step = 180 / intervals
    result = []
    for pitch in np.linspace(-90, 90, intervals + 1):
        if abs(pitch) >= 90 - 1e-8:
            count = 1
        else:
            max_cos = np.cos(np.deg2rad(max(0, abs(pitch) - lat_step / 2)))
            count = int(np.ceil(360 * max_cos / step))
        for yaw in np.linspace(-180, 180, count, endpoint=False):
            result.append(Perspective(width, height, yaw=float(yaw), pitch=float(pitch), hfov=hfov))
    return result


Projection = Equirectangular | Perspective


def projection_to_dict(projection: Projection) -> dict[str, Any]:
    return {"kind": projection.kind, **asdict(projection)}


def projection_from_dict(value: dict[str, Any]) -> Projection:
    value = dict(value)
    kind = value.pop("kind", None)
    if kind == "equirectangular":
        return Equirectangular(**value)
    if kind == "perspective":
        return Perspective(**value)
    raise ValueError(f"Unknown projection {kind!r}.")

"""Perspective preprocessing with an explicit inverse into the original panorama."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .annotations import Box2D, SphericalRegion
from .geometry import points2d, positive_int
from .projections import Equirectangular, Perspective, projection_from_dict, projection_to_dict


@dataclass(frozen=True)
class ViewTransform:
    """A panorama-to-perspective mapping followed by positive scaling and translation."""

    source: Equirectangular
    view: Perspective
    scale: tuple[float, float] = (1.0, 1.0)
    offset: tuple[float, float] = (0.0, 0.0)
    output_size: tuple[int, int] | None = None
    content_bounds: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, Equirectangular) or not isinstance(self.view, Perspective):
            raise TypeError(
                "ViewTransform requires an Equirectangular source and Perspective view."
            )
        for name in ("scale", "offset"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain two finite values.")
            object.__setattr__(self, name, (float(values[0]), float(values[1])))
        if min(self.scale) <= 0:
            raise ValueError("scale must be positive.")
        if self.output_size is None:
            object.__setattr__(self, "output_size", self.view.size)
        if len(self.size) != 2:
            raise ValueError("output_size must contain width and height.")
        for value in self.size:
            positive_int(value, "output dimension")
        object.__setattr__(self, "output_size", tuple(self.size))
        bounds = np.asarray(self.bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("content_bounds must contain four finite values.")
        x1, y1, x2, y2 = bounds
        if not (0 <= x1 <= x2 <= self.view.width and 0 <= y1 <= y2 <= self.view.height):
            raise ValueError("content_bounds must lie within the original view.")
        object.__setattr__(self, "content_bounds", (float(x1), float(y1), float(x2), float(y2)))

    @property
    def size(self) -> tuple[int, int]:
        return self.view.size if self.output_size is None else self.output_size

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            (0, 0, self.view.width, self.view.height)
            if self.content_bounds is None
            else self.content_bounds
        )

    def _original(self, points: ArrayLike) -> NDArray:
        return (points2d(points) - self.offset) / self.scale

    def valid(self, points: ArrayLike) -> NDArray:
        """Whether output coordinates belong to retained image content."""
        points = points2d(points)
        xy = self._original(points)
        x1, y1, x2, y2 = self.bounds
        w, h = self.size
        return (
            (points[..., 0] >= -1e-8)
            & (points[..., 0] <= w + 1e-8)
            & (points[..., 1] >= -1e-8)
            & (points[..., 1] <= h + 1e-8)
            & (xy[..., 0] >= x1 - 1e-8)
            & (xy[..., 0] <= x2 + 1e-8)
            & (xy[..., 1] >= y1 - 1e-8)
            & (xy[..., 1] <= y2 + 1e-8)
            & (x2 > x1)
            & (y2 > y1)
        )

    def to_panorama(self, points: ArrayLike) -> tuple[NDArray, NDArray]:
        """Map output xy to panorama xy; ignore coordinates whose validity is false."""
        rays, _ = self.view.unproject(self._original(points))
        xy, _ = self.source.project(rays)
        return xy, self.valid(points)

    def from_panorama(self, points: ArrayLike) -> tuple[NDArray, NDArray]:
        rays, inside = self.source.unproject(points)
        xy, visible = self.view.project(rays)
        output = xy * self.scale + self.offset
        return output, inside & visible & self.valid(output)

    def _then(
        self, scale: tuple[float, float], offset: tuple[float, float], width: int, height: int
    ) -> ViewTransform:
        positive_int(width, "width")
        positive_int(height, "height")
        sx, sy = scale
        return replace(
            self,
            scale=(self.scale[0] * sx, self.scale[1] * sy),
            offset=(self.offset[0] * sx + offset[0], self.offset[1] * sy + offset[1]),
            output_size=(width, height),
        )

    def resize(self, width: int, height: int) -> ViewTransform:
        w, h = self.size
        return self._then((width / w, height / h), (0, 0), width, height)

    def letterbox(self, width: int, height: int) -> ViewTransform:
        w, h = self.size
        scale = min(width / w, height / h)
        return self._then(
            (scale, scale), ((width - w * scale) / 2, (height - h * scale) / 2), width, height
        )

    def crop(self, x: float, y: float, width: int, height: int) -> ViewTransform:
        positive_int(width, "width")
        positive_int(height, "height")
        w, h = self.size
        if not np.isfinite([x, y]).all() or x < 0 or y < 0 or x + width > w or y + height > h:
            raise ValueError("Crop must lie within the current image.")
        xy = self._original([[x, y], [x + width, y + height]])
        lo = np.maximum(xy[0], self.bounds[:2])
        hi = np.minimum(xy[1], self.bounds[2:])
        bounds = (
            (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
            if np.all(hi > lo)
            else (0.0, 0.0, 0.0, 0.0)
        )
        return replace(self._then((1, 1), (-x, -y), width, height), content_bounds=bounds)

    def map_box(self, box: Box2D) -> SphericalRegion | None:
        """Clip to retained content. Return None for boxes entirely in padding."""
        if not isinstance(box, Box2D):
            raise TypeError("map_box expects a Box2D.")
        lo = np.maximum(box.xyxy[:2], [0, 0])
        hi = np.minimum(box.xyxy[2:], self.size)
        if np.any(hi <= lo):
            return None
        xy = self._original([lo, hi])
        lo = np.maximum(xy[0], self.bounds[:2])
        hi = np.minimum(xy[1], self.bounds[2:])
        if np.any(hi <= lo):
            return None
        return SphericalRegion(Box2D(lo[0], lo[1], hi[0], hi[1], **box.metadata()), self.view)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": projection_to_dict(self.source),
            "view": projection_to_dict(self.view),
            "scale": list(self.scale),
            "offset": list(self.offset),
            "output_size": list(self.size),
            "content_bounds": list(self.bounds),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ViewTransform:
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported transform schema_version.")
        source = projection_from_dict(value["source"])
        view = projection_from_dict(value["view"])
        if not isinstance(source, Equirectangular) or not isinstance(view, Perspective):
            raise ValueError("Persisted transform requires a panorama source and perspective view.")
        return cls(
            source,
            view,
            tuple(value["scale"]),
            tuple(value["offset"]),
            tuple(value["output_size"]),
            tuple(value["content_bounds"]),
        )

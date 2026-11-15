"""Boxes and their spherical interiors. Polygon output may have seam pieces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import normalize, positive_int
from .projections import (
    Equirectangular,
    Perspective,
    Projection,
    projection_from_dict,
    projection_to_dict,
)


@dataclass(frozen=True, kw_only=True)
class _Metadata:
    label: str | int | None = None
    score: float | None = None
    track_id: str | int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None:
            if not 0 <= self.score <= 1:
                raise ValueError("score must be between 0 and 1.")
            object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "attributes", dict(self.attributes))

    def metadata(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "track_id": self.track_id,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class Box2D(_Metadata):
    """Positive-area pixel-edge coordinates, with optional prediction metadata."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.xyxy).all() or self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Boxes require finite coordinates and positive width and height.")
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @property
    def xyxy(self) -> NDArray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def polygon(self) -> Polygon2D:
        return Polygon2D(
            np.array(
                [[self.x1, self.y1], [self.x2, self.y1], [self.x2, self.y2], [self.x1, self.y2]]
            ),
            **self.metadata(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"xyxy": self.xyxy.tolist(), **self.metadata()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Box2D:
        return cls(
            *value["xyxy"],
            **{k: value[k] for k in ("label", "score", "track_id", "attributes") if k in value},
        )


@dataclass(frozen=True)
class Polygon2D(_Metadata):
    """A rendered boundary. Combine rings with even-odd fill; is_hole marks holes."""

    vertices: NDArray
    is_hole: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 2
            or len(vertices) < 3
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("A polygon needs at least three finite xy vertices.")
        vertices.flags.writeable = False
        object.__setattr__(self, "vertices", vertices)

    def bbox(self) -> Box2D:
        x1, y1 = self.vertices.min(0)
        x2, y2 = self.vertices.max(0)
        return Box2D(x1, y1, x2, y2, **self.metadata())


@dataclass(frozen=True)
class SphericalRegion:
    """One object defined by a box and its projection, including its intended interior.

    Equirectangular boxes may cross the seam using unwrapped x, such as 350..370
    in a 360-pixel panorama. Their width cannot exceed one revolution.
    """

    box: Box2D
    projection: Projection

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box2D) or not isinstance(
            self.projection, (Equirectangular, Perspective)
        ):
            raise TypeError(
                "A region requires a Box2D and an Equirectangular or Perspective projection."
            )
        b, p = self.box, self.projection
        if b.y1 < 0 or b.y2 > p.height:
            raise ValueError("Box y coordinates must lie within the projection.")
        if isinstance(p, Equirectangular):
            if b.x2 - b.x1 > p.width:
                raise ValueError("A panorama box cannot span more than one revolution.")
        elif b.x1 < 0 or b.x2 > p.width:
            raise ValueError("Box x coordinates must lie within the perspective view.")

    def contains(self, rays: ArrayLike) -> NDArray:
        xy, visible = self.projection.project(normalize(rays))
        b = self.box
        x = xy[..., 0]
        if isinstance(self.projection, Equirectangular):
            distance = (x - b.x1) % self.projection.width
            inside_x = (distance <= b.x2 - b.x1 + 1e-8) | (distance >= self.projection.width - 1e-8)
        else:
            inside_x = (x >= b.x1 - 1e-8) & (x <= b.x2 + 1e-8)
        return visible & inside_x & (xy[..., 1] >= b.y1 - 1e-8) & (xy[..., 1] <= b.y2 + 1e-8)

    def boundary(self, samples_per_edge: int = 32) -> NDArray:
        """Sample all four source edges and return world-space unit rays."""
        positive_int(samples_per_edge, "samples_per_edge")
        return self.projection.unproject(_densify(self.box.polygon().vertices, samples_per_edge))[0]

    def rasterize(self, target: Projection) -> NDArray:
        """Authoritative membership at target pixel centers."""
        mask = np.empty((target.height, target.width), bool)
        for start in range(0, target.height, 128):
            stop = min(start + 128, target.height)
            rays, valid = target.unproject(target.pixel_grid(start, stop))
            mask[start:stop] = valid & self.contains(rays)
        return mask

    def polygons(self, target: Projection, *, samples_per_edge: int = 32) -> list[Polygon2D]:
        """Render boundaries in the target image.

        Perspective-source edges are sampled approximations. Panorama-source
        regions use pixel-edge contours, preserving wide interiors and both poles
        under rotation. Subpixel regions may disappear in that raster fallback.
        """
        positive_int(samples_per_edge, "samples_per_edge")
        if not isinstance(target, (Equirectangular, Perspective)):
            raise TypeError("Unsupported target projection.")
        if isinstance(self.projection, Equirectangular):
            return [
                replace(p, **self.box.metadata()) for p in _mask_to_polygons(self.rasterize(target))
            ]
        rays = self.boundary(samples_per_edge)
        if isinstance(target, Perspective):
            local = rays @ target.rotation
            for normal in (
                [0, 0, 1],
                [1, 0, target.width / (2 * target.focal)],
                [-1, 0, target.width / (2 * target.focal)],
                [0, 1, target.height / (2 * target.focal)],
                [0, -1, target.height / (2 * target.focal)],
            ):
                local = _clip(local, np.asarray(normal))
            if len(local) < 3:
                return []
            xy, _ = target.project(normalize(local) @ target.rotation.T)
            return self._polygons([_clip_image(xy, target.width, target.height)])
        xy, _ = target.project(np.concatenate((rays, rays[:1])))
        x = np.unwrap(xy[:, 0] / target.width * (2 * np.pi)) / (2 * np.pi) * target.width
        boundary = np.column_stack((x, xy[:, 1]))
        if abs(x[-1] - x[0]) > target.width * 0.5:
            north = self.contains(np.array([0, 1, 0]) @ target.rotation.T)
            pole_y = 0 if north else target.height
            boundary = np.concatenate((boundary, [[x[-1], pole_y], [x[0], pole_y]]))
        lo, hi = int(np.floor(np.min(x) / target.width)), int(np.floor(np.max(x) / target.width))
        return self._polygons(
            [
                _clip_image(boundary - [shift * target.width, 0], target.width, target.height)
                for shift in range(lo, hi + 1)
            ]
        )

    def _polygons(self, pieces: list[NDArray]) -> list[Polygon2D]:
        return [
            Polygon2D(xy, **self.box.metadata())
            for xy in pieces
            if len(xy) >= 3 and abs(_signed_area(xy)) > 1e-8
        ]

    def boxes(self, target: Projection) -> list[Box2D]:
        """Approximate enclosing rectangles for outer components; holes are excluded."""
        return [p.bbox() for p in self.polygons(target) if not p.is_hole]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "box": self.box.to_dict(),
            "projection": projection_to_dict(self.projection),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SphericalRegion:
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported region schema_version.")
        return cls(Box2D.from_dict(value["box"]), projection_from_dict(value["projection"]))


def _signed_area(xy: NDArray) -> float:
    return float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)) / 2)


def _clip(vertices: NDArray, normal: NDArray, offset: float = 0) -> NDArray:
    """Sutherland-Hodgman against dot(vertex, normal) + offset >= 0."""
    if len(vertices) == 0:
        return vertices
    output = []
    a = vertices[-1]
    da = float(a @ normal + offset)
    for b in vertices:
        db = float(b @ normal + offset)
        if (da >= 0) != (db >= 0):
            output.append(a + da / (da - db) * (b - a))
        if db >= 0:
            output.append(b)
        a, da = b, db
    return np.asarray(output).reshape(-1, vertices.shape[-1])


def _clip_image(vertices: NDArray, width: float, height: float) -> NDArray:
    for normal, offset in (([1, 0], 0), ([-1, 0], width), ([0, 1], 0), ([0, -1], height)):
        vertices = _clip(vertices, np.asarray(normal), offset)
    return vertices


def _densify(vertices: NDArray, samples_per_edge: int) -> NDArray:
    t = np.arange(samples_per_edge) / samples_per_edge
    return np.concatenate(
        [
            a[None] * (1 - t[:, None]) + b[None] * t[:, None]
            for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True)
        ]
    )


def _mask_to_polygons(mask: NDArray) -> list[Polygon2D]:
    """Trace pixel-edge contours, including hole rings (opposite winding)."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2:
        raise ValueError("mask must be HW.")
    padded = np.pad(mask, 1)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for dy, dx, a, b in (
        (-1, 0, (0, 0), (1, 0)),
        (0, 1, (1, 0), (1, 1)),
        (1, 0, (1, 1), (0, 1)),
        (0, -1, (0, 1), (0, 0)),
    ):
        neighbor = padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
        ys, xs = np.nonzero(mask & ~neighbor)
        for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
            edges.setdefault((x + a[0], y + a[1]), []).append((x + b[0], y + b[1]))
    polygons = []
    while edges:
        first = next(iter(edges))
        point = first
        ring = []
        while True:
            ring.append(point)
            following = edges[point].pop()
            if not edges[point]:
                del edges[point]
            point = following
            if point == first:
                break
        if len(ring) >= 3:
            vertices = np.asarray(ring)
            polygons.append(Polygon2D(vertices, is_hole=_signed_area(vertices) < 0))
    return polygons

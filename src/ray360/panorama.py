"""Stitched panorama arrays and perspective views with retained transforms."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .annotations import Box2D, SphericalRegion
from .projections import Equirectangular, Perspective
from .remap import DEFAULT_CACHE, Interpolation, MapCache, ProjectionMap, sample, validate_image
from .transforms import ViewTransform


@dataclass(frozen=True, init=False)
class Panorama:
    """A caller-owned array explicitly declared to be a full stitched panorama.

    No pixel conversion or copy occurs. Do not mutate a shared input during use.
    The optional projection sets orientation; dimensions must match the array.
    """

    image: NDArray
    projection: Equirectangular
    source_id: str | None
    timestamp: float | None

    def __init__(
        self,
        image: NDArray,
        projection: Equirectangular | None = None,
        *,
        source_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        image = validate_image(image)
        if projection is None:
            projection = Equirectangular(image.shape[1], image.shape[0])
        if not isinstance(projection, Equirectangular):
            raise TypeError("Panorama accepts only already-stitched equirectangular images.")
        if projection.size != image.shape[:2][::-1]:
            raise ValueError("Image dimensions must match the panorama projection.")
        if timestamp is not None and not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite.")
        for name, value in (
            ("image", image),
            ("projection", projection),
            ("source_id", source_id),
            ("timestamp", timestamp),
        ):
            object.__setattr__(self, name, value)

    def view(
        self,
        target: Perspective,
        *,
        interpolation: Interpolation = "bilinear",
        cache: MapCache | None = DEFAULT_CACHE,
    ) -> View:
        mapping = (
            ProjectionMap.build(self.projection, target)
            if cache is None
            else cache.get(self.projection, target)
        )
        return View(
            mapping.apply(self.image, interpolation=interpolation),
            ViewTransform(self.projection, target),
            interpolation,
            self.source_id,
            self.timestamp,
        )

    def views(
        self,
        targets: Iterable[Perspective],
        *,
        interpolation: Interpolation = "bilinear",
        cache: MapCache | None = DEFAULT_CACHE,
    ) -> Iterator[View]:
        """Yield one view at a time, reusing maps across frames when a cache is supplied."""
        for target in targets:
            yield self.view(target, interpolation=interpolation, cache=cache)


@dataclass(frozen=True)
class View:
    image: NDArray
    transform: ViewTransform
    interpolation: Interpolation = "bilinear"
    source_id: str | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        image = validate_image(self.image)
        if image.shape[:2][::-1] != self.transform.size:
            raise ValueError("View dimensions must match its transform.")
        if self.interpolation not in ("nearest", "bilinear"):
            raise ValueError("interpolation must be nearest or bilinear.")
        if self.timestamp is not None and not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite.")
        object.__setattr__(self, "image", image)

    @property
    def valid(self) -> NDArray:
        """A fresh HW mask marking retained content rather than preprocessing padding."""
        w, h = self.transform.size
        output = np.empty((h, w), bool)
        for start in range(0, h, 128):
            stop = min(h, start + 128)
            x, y = np.meshgrid(np.arange(w) + 0.5, np.arange(start, stop) + 0.5)
            output[start:stop] = self.transform.valid(np.stack((x, y), -1))
        return output

    def _resample(self, transform: ViewTransform, interpolation: Interpolation | None) -> View:
        mode = self.interpolation if interpolation is None else interpolation
        w, h = transform.size
        output = np.empty((h, w, *self.image.shape[2:]), dtype=self.image.dtype)
        for start in range(0, h, 128):
            stop = min(h, start + 128)
            x, y = np.meshgrid(np.arange(w) + 0.5, np.arange(start, stop) + 0.5)
            points = np.stack((x, y), -1)
            xy = transform._original(points) * self.transform.scale + self.transform.offset
            output[start:stop] = sample(
                self.image, xy, valid=transform.valid(points), interpolation=mode
            )
        return replace(self, image=output, transform=transform, interpolation=mode)

    def resize(
        self, width: int, height: int, *, interpolation: Interpolation | None = None
    ) -> View:
        return self._resample(self.transform.resize(width, height), interpolation)

    def letterbox(
        self, width: int, height: int, *, interpolation: Interpolation | None = None
    ) -> View:
        """Scale with symmetric zero padding. Fractional padding stays in the transform."""
        return self._resample(self.transform.letterbox(width, height), interpolation)

    def crop(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        *,
        interpolation: Interpolation | None = None,
    ) -> View:
        return self._resample(self.transform.crop(x, y, width, height), interpolation)

    def map_boxes(self, boxes: Iterable[Box2D]) -> list[SphericalRegion]:
        """Map boxes in current output coordinates, discarding boxes wholly in padding."""
        return [region for box in boxes if (region := self.transform.map_box(box)) is not None]

    def provenance(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "interpolation": self.interpolation,
            "transform": self.transform.to_dict(),
        }

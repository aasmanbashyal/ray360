"""Bounded projection-map caching and chunked image resampling."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from threading import RLock
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .geometry import positive_int
from .projections import Equirectangular, Perspective

Interpolation = Literal["nearest", "bilinear"]


class _NativeSampler(Protocol):
    def __call__(
        self,
        image: NDArray,
        xy: NDArray,
        *,
        wrap: bool,
        interpolation: Interpolation,
        valid: NDArray | None,
    ) -> NDArray | None: ...


def sample(
    image: NDArray,
    xy: NDArray,
    *,
    wrap: bool = False,
    interpolation: Interpolation = "bilinear",
    valid: NDArray | None = None,
) -> NDArray:
    """Sample at edge-based coordinates, clamping vertically and optionally wrapping x."""
    return _sample(image, xy, wrap, interpolation, valid, _native_sampler())


@cache
def _native_sampler() -> _NativeSampler | None:
    try:
        from ray360_native._sampler import try_sample_v1
    except ImportError:
        return None
    return try_sample_v1


def _sample_numpy(
    image: NDArray,
    xy: NDArray,
    *,
    wrap: bool = False,
    interpolation: Interpolation = "bilinear",
    valid: NDArray | None = None,
) -> NDArray:
    return _sample(image, xy, wrap, interpolation, valid, None)


def _sample(
    image: NDArray,
    xy: NDArray,
    wrap: bool,
    interpolation: Interpolation,
    valid: NDArray | None,
    sampler: _NativeSampler | None,
) -> NDArray:
    image = validate_image(image)
    if interpolation not in ("nearest", "bilinear"):
        raise ValueError("interpolation must be 'nearest' or 'bilinear'.")
    if interpolation == "bilinear" and (
        image.dtype.kind == "b" or (image.dtype.kind in "ui" and image.dtype.itemsize >= 8)
    ):
        raise ValueError("Boolean and 64-bit integer arrays require nearest interpolation.")
    xy = np.asarray(xy)
    if sampler is not None:
        result = sampler(image, xy, wrap=wrap, interpolation=interpolation, valid=valid)
        if result is not None:
            return result

    h, w = image.shape[:2]
    finite = np.isfinite(xy).all(-1)
    x, y = np.moveaxis(np.where(finite[..., None], xy, 0), -1, 0)
    x = np.remainder(x, w) if wrap else np.clip(x, 0, w)
    y = np.clip(y, 0, h)
    x, y = x - 0.5, y - 0.5

    def ix(values):
        return values % w if wrap else np.clip(values, 0, w - 1)

    if interpolation == "nearest":
        result = image[
            np.clip(np.floor(y + 0.5).astype(np.int64), 0, h - 1),
            ix(np.floor(x + 0.5).astype(np.int64)),
        ].copy()
    else:
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        work_dtype = np.result_type(image.dtype, np.float32)
        fx, fy = (x - x0).astype(work_dtype), (y - y0).astype(work_dtype)
        if image.ndim == 3:
            fx, fy = fx[..., None], fy[..., None]
        a = image[np.clip(y0, 0, h - 1), ix(x0)].astype(work_dtype)
        b = image[np.clip(y0, 0, h - 1), ix(x0 + 1)].astype(work_dtype)
        c = image[np.clip(y0 + 1, 0, h - 1), ix(x0)].astype(work_dtype)
        d = image[np.clip(y0 + 1, 0, h - 1), ix(x0 + 1)].astype(work_dtype)
        result = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
        if image.dtype.kind in "ui":
            limits = np.iinfo(image.dtype)
            result = np.clip(np.rint(result), limits.min, limits.max)
        result = result.astype(image.dtype)
    # A single coordinate on an HW image can produce a NumPy scalar.
    result = np.asarray(result)
    mask = finite if valid is None else finite & valid
    result[~mask] = 0
    return result


@dataclass(frozen=True)
class ProjectionMap:
    source: Equirectangular
    target: Perspective
    xy: NDArray
    valid: NDArray

    @classmethod
    def build(
        cls, source: Equirectangular, target: Perspective, *, chunk_rows: int = 128
    ) -> ProjectionMap:
        positive_int(chunk_rows, "chunk_rows")
        if not isinstance(source, Equirectangular) or not isinstance(target, Perspective):
            raise TypeError(
                "ProjectionMap requires an Equirectangular source and Perspective target."
            )
        xy = np.empty((target.height, target.width, 2), dtype=np.float32)
        valid = np.empty((target.height, target.width), dtype=bool)
        for start in range(0, target.height, chunk_rows):
            stop = min(target.height, start + chunk_rows)
            rays, visible = target.unproject(target.pixel_grid(start, stop))
            coords, inside = source.project(rays)
            xy[start:stop] = coords
            valid[start:stop] = visible & inside
        xy.flags.writeable = valid.flags.writeable = False
        return cls(source, target, xy, valid)

    @property
    def nbytes(self) -> int:
        return self.xy.nbytes + self.valid.nbytes

    def apply(
        self, image: NDArray, *, interpolation: Interpolation = "bilinear", chunk_rows: int = 128
    ) -> NDArray:
        image = validate_image(image)
        if image.shape[:2] != (self.source.height, self.source.width):
            raise ValueError("Image dimensions do not match the map's source projection.")
        positive_int(chunk_rows, "chunk_rows")
        output = np.empty(
            (self.target.height, self.target.width, *image.shape[2:]), dtype=image.dtype
        )
        self._sample_into(image, output, interpolation=interpolation, chunk_rows=chunk_rows)
        return output

    def _sample_into(
        self,
        image: NDArray,
        output: NDArray,
        *,
        interpolation: Interpolation,
        chunk_rows: int,
    ) -> None:
        """Fill a matching destination, which may be a slice of a larger image."""
        for start in range(0, self.target.height, chunk_rows):
            stop = start + chunk_rows
            output[start:stop] = sample(
                image,
                self.xy[start:stop],
                valid=self.valid[start:stop],
                wrap=isinstance(self.source, Equirectangular),
                interpolation=interpolation,
            )


class MapCache:
    """Thread-safe LRU with a byte budget. Oversized maps are returned without caching."""

    def __init__(self, max_bytes: int = 128 * 1024 * 1024):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative.")
        self._max_bytes = max_bytes
        self._maps: OrderedDict[tuple[Equirectangular, Perspective], ProjectionMap] = OrderedDict()
        self._lock = RLock()

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(value.nbytes for value in self._maps.values())

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def clear(self) -> None:
        with self._lock:
            self._maps.clear()

    def get(self, source: Equirectangular, target: Perspective) -> ProjectionMap:
        key = source, target
        with self._lock:
            if key in self._maps:
                self._maps.move_to_end(key)
                return self._maps[key]
        result = ProjectionMap.build(source, target)
        with self._lock:
            if result.nbytes <= self.max_bytes:
                self._maps[key] = result
                self._maps.move_to_end(key)
                while self.nbytes > self.max_bytes:
                    self._maps.popitem(last=False)
        return result


DEFAULT_CACHE = MapCache()


def validate_image(image: NDArray) -> NDArray:
    image = np.asarray(image)
    if image.ndim not in (2, 3) or any(n == 0 for n in image.shape):
        raise ValueError("Expected a nonempty HW or HWC array.")
    if image.dtype.kind not in "buif":
        raise ValueError("Images must have boolean, integer, or floating dtype.")
    return image

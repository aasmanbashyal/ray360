"""Reusable single-pass perspective views."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from ray360 import Panorama, View, ViewTransform

from . import _native
from ._sampler import _MAX_FLOAT32_SOURCE_DIM, _sample_numpy


@dataclass(frozen=True, init=False)
class PreparedView:
    """Compose projection/crop/resize/letterbox geometry and sample once per frame.

    This explicitly changes successive-resampling semantics. Geometry and
    metadata are retained, but pixels need not equal an eager resize chain.
    Preparation owns its interpolation plan. Coordinate maps are generated lazily
    for inspection or NumPy fallback. nbytes includes any cached maps. No global cache.
    """

    transform: ViewTransform
    _plan: _native.PreparedBilinear | None = field(repr=False)

    def __init__(self, transform):
        if not isinstance(transform, ViewTransform):
            raise TypeError("Expected a ViewTransform")
        plan = None
        if max(transform.source.size) <= _MAX_FLOAT32_SOURCE_DIM:
            xy, valid = self._coordinates(transform)
            plan = _native.PreparedBilinear(*transform.source.size, xy, valid.view(np.uint8), True)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "_plan", plan)

    @staticmethod
    def _coordinates(transform):
        rotation = transform.view.rotation.T @ transform.source.rotation
        xy, mask = _native.fused_coordinates(
            transform.source.size,
            transform.view.size,
            transform.size,
            transform.view.focal,
            rotation.ravel().tolist(),
            transform.scale,
            transform.offset,
            transform.bounds,
        )
        valid = mask.view(bool)
        xy.flags.writeable = mask.flags.writeable = valid.flags.writeable = False
        return xy, valid

    @cached_property
    def _maps(self):
        return self._coordinates(self.transform)

    @property
    def xy(self):
        return self._maps[0]

    @property
    def valid(self):
        return self._maps[1]

    @property
    def nbytes(self):
        map_bytes = sum(m.nbytes for m in self.__dict__.get("_maps", ()))
        return map_bytes if self._plan is None else map_bytes + self._plan.nbytes

    def apply(self, panorama, *, engine="auto"):
        if not isinstance(panorama, Panorama):
            raise TypeError("Expected a Panorama")
        if panorama.projection != self.transform.source:
            raise ValueError("Panorama projection must match the prepared source")
        if engine not in ("auto", "scalar", "avx2"):
            raise ValueError("engine must be auto, scalar, or avx2")
        image = panorama.image
        if self._plan is not None and image.dtype == np.uint8 and image.flags.c_contiguous:
            output = self._plan.apply(image, engine)
        else:
            if engine != "auto":
                if self._plan is None:
                    raise ValueError(
                        "Explicit Rust engines require source dimensions at most 65,535."
                    )
                raise ValueError("Explicit Rust engines require contiguous uint8 input")
            output = _sample_numpy(image, self.xy, wrap=True, valid=self.valid)
        return View(output, self.transform, "bilinear", panorama.source_id, panorama.timestamp)

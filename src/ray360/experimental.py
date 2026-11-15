"""Experimental sampled overlap and NMS; excluded from the stable root API.

Small regions can miss all samples. These methods neither fuse partial detections
nor establish detector accuracy. Increase samples and validate on the target data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .annotations import SphericalRegion
from .geometry import fibonacci_sphere


def spherical_iou(a: SphericalRegion, b: SphericalRegion, *, samples: int = 65536) -> float:
    """Approximate solid-angle IoU using equal-area samples, not panorama pixel area.

    Increase samples for small objects; a union missed by all samples yields 0.
    """
    rays = fibonacci_sphere(samples)
    aa, bb = a.contains(rays), b.contains(rays)
    union = int(np.count_nonzero(aa | bb))
    return int(np.count_nonzero(aa & bb)) / union if union else 0.0


def spherical_nms(
    regions: list[SphericalRegion],
    threshold: float = 0.5,
    *,
    samples: int = 65536,
    class_agnostic: bool = False,
) -> list[SphericalRegion]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    rays = fibonacci_sphere(samples)
    # Evaluate each region once, rather than resampling every pair.
    ordered = sorted(
        regions, key=lambda r: r.box.score if r.box.score is not None else 1.0, reverse=True
    )
    kept: list[tuple[SphericalRegion, NDArray]] = []
    for region in ordered:
        mask = region.contains(rays)
        suppressed = False
        for other, other_mask in kept:
            if not class_agnostic and region.box.label != other.box.label:
                continue
            union = int(np.count_nonzero(mask | other_mask))
            overlap = int(np.count_nonzero(mask & other_mask)) / union if union else 0
            if overlap > threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append((region, mask))
    return [region for region, _ in kept]

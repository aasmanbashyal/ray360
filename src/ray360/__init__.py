"""Perspective views, cubemap crosses and seam-aware annotation mapping for 360 images."""

from .annotations import Box2D, Polygon2D, SphericalRegion
from .cubemap import (
    cube_projections,
    from_cubemap,
    from_cubemap_cross,
    to_cubemap,
    to_cubemap_cross,
)
from .panorama import Panorama, View
from .projections import Equirectangular, Perspective, coverage_views
from .remap import MapCache, ProjectionMap
from .transforms import ViewTransform

__version__ = "0.1.0"

__all__ = [
    "Box2D",
    "Equirectangular",
    "MapCache",
    "Panorama",
    "Perspective",
    "Polygon2D",
    "ProjectionMap",
    "SphericalRegion",
    "View",
    "ViewTransform",
    "coverage_views",
    "cube_projections",
    "from_cubemap",
    "from_cubemap_cross",
    "to_cubemap",
    "to_cubemap_cross",
]

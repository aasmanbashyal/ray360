import numpy as np
import pytest

from ray360 import Box2D, Equirectangular, Perspective, SphericalRegion
from ray360.experimental import spherical_iou, spherical_nms
from ray360.geometry import lonlat_to_rays


def test_distinct_overlapping_views_merge_a_shared_seam_object():
    corners = lonlat_to_rays([[172, -10], [188, -10], [188, 10], [172, 10]])
    regions = []
    for yaw, score in [(165, 0.9), (195, 0.7)]:
        projection = Perspective(200, 160, yaw=yaw, hfov=90)
        xy, valid = projection.project(corners)
        assert valid.all()
        lo, hi = xy.min(0), xy.max(0)
        regions.append(SphericalRegion(Box2D(*lo, *hi, label="object", score=score), projection))
    assert regions[0].projection != regions[1].projection
    assert spherical_iou(*regions, samples=65536) > 0.75
    assert spherical_nms(regions, 0.5)[0] is regions[0]
    assert len(spherical_nms(regions, 0.5)) == 1
    other = SphericalRegion(
        Box2D(*regions[1].box.xyxy, label="other", score=0.8), regions[1].projection
    )
    assert len(spherical_nms([*regions, other], 0.5)) == 2


def test_overlap_against_analytic_solid_angle_and_small_object_limit():
    projection = Equirectangular(360, 180)
    a = SphericalRegion(Box2D(170, 70, 190, 110), projection)
    b = SphericalRegion(Box2D(180, 70, 200, 110), projection)
    assert spherical_iou(a, b, samples=131072) == pytest.approx(1 / 3, abs=0.01)
    tiny = SphericalRegion(Box2D(180, 90, 180.000001, 90.000001), projection)
    assert spherical_iou(tiny, tiny, samples=128) == 0
    assert len(spherical_nms([tiny, tiny], samples=128)) == 2
    with pytest.raises(ValueError):
        spherical_nms([], threshold=np.nan)

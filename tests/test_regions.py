import json

import numpy as np
import pytest

from ray360 import Box2D, Equirectangular, Perspective, SphericalRegion
from ray360.annotations import _mask_to_polygons
from ray360.geometry import lonlat_to_rays


def fill(polygons, projection):
    """Independent scan-line even-odd fill at target pixel centers."""
    xy = projection.pixel_grid()
    x, y = xy[..., 0], xy[..., 1]
    mask = np.zeros(x.shape, bool)
    for polygon in polygons:
        vertices = polygon.vertices
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            if abs(b[1] - a[1]) > 1e-12:
                mask ^= ((a[1] > y) != (b[1] > y)) & (
                    x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
                )
    return mask


def test_seam_is_two_pieces_of_one_object_with_metadata():
    box = Box2D(50, 50, 150, 150, label="car", score=0.8, track_id=17)
    region = SphericalRegion(box, Perspective(200, 200, yaw=180))
    target = Equirectangular(360, 180)
    pieces = region.polygons(target)
    assert len(pieces) == 2
    assert all(p.metadata() == box.metadata() for p in pieces)
    assert region.contains(lonlat_to_rays([[179, 0], [-179, 0], [0, 0]])).tolist() == [
        True,
        True,
        False,
    ]
    assert np.mean(fill(pieces, target) != region.rasterize(target)) < 0.001
    restored = SphericalRegion.from_dict(json.loads(json.dumps(region.to_dict())))
    np.testing.assert_array_equal(restored.rasterize(target), region.rasterize(target))


@pytest.mark.parametrize("pitch,pole_y", [(90, 0), (-90, 180)])
def test_pole_caps_and_curved_edges(pitch, pole_y):
    region = SphericalRegion(Box2D(20, 20, 180, 180), Perspective(200, 200, pitch=pitch))
    target = Equirectangular(360, 180)
    mask = region.rasterize(target)
    assert mask[0 if pitch > 0 else -1].all()
    polygons = region.polygons(target)
    assert any(np.any(np.isclose(p.vertices[:, 1], pole_y)) for p in polygons)
    assert np.mean(fill(polygons, target) != mask) < 0.005
    front = SphericalRegion(Box2D(0, 0, 200, 200), Perspective(200, 200))
    assert front.polygons(target)[0].vertices[:, 1].min() == pytest.approx(45)


@pytest.mark.parametrize(
    "box",
    [
        Box2D(0, 0, 360, 180),
        Box2D(0, 20, 300, 160),
        Box2D(0, 50, 360, 130),
        Box2D(350, 70, 370, 110),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        Equirectangular(120, 60),
        Equirectangular(120, 60, pitch=45),
        Equirectangular(120, 60, yaw=50, roll=40),
        Perspective(60, 60, yaw=180),
    ],
)
def test_wide_and_seam_panorama_regions_preserve_rotated_interiors(box, target):
    region = SphericalRegion(box, Equirectangular(360, 180))
    expected = region.rasterize(target)
    polygons = region.polygons(target)
    np.testing.assert_array_equal(fill(polygons, target), expected)
    if box.area == 360 * 180:
        assert expected.all() and polygons and region.boxes(target)


def test_panorama_seam_membership_and_whole_sphere_are_analytic():
    projection = Equirectangular(360, 180)
    region = SphericalRegion(Box2D(350, 70, 370, 110), projection)
    rays = lonlat_to_rays([[179, 0], [-179, 0], [0, 0], [179, 30]])
    assert region.contains(rays).tolist() == [True, True, False, False]
    full = SphericalRegion(Box2D(0, 0, 360, 180), projection)
    assert full.contains(rays).all()
    assert full.rasterize(Equirectangular(360, 180, yaw=50, roll=40)).all()


def test_perspective_frustum_clipping_and_behind_camera():
    region = SphericalRegion(Box2D(0, 0, 200, 200), Perspective(200, 200, hfov=140))
    target = Perspective(100, 100, hfov=50)
    np.testing.assert_allclose(region.boxes(target)[0].xyxy, [0, 0, 100, 100], atol=1e-7)
    assert region.polygons(Perspective(100, 100, yaw=180)) == []


@pytest.mark.parametrize("left", [1, 7, 350])
def test_panorama_region_includes_exact_edges_after_rotation(left):
    projection = Equirectangular(360, 180, yaw=31, pitch=22, roll=14)
    region = SphericalRegion(Box2D(left, 20, left + 20, 160), projection)
    edges = [[x, y] for x in (left, left + 20) for y in (20, 50, 90, 130, 160)]
    rays, _ = projection.unproject(edges)
    assert region.contains(rays).all()
    outside, _ = projection.unproject([[left - 1e-5, 90], [left + 20 + 1e-5, 90]])
    assert not region.contains(outside).any()


def test_contours_retain_holes_and_diagonal_components():
    mask = np.zeros((20, 20), bool)
    mask[2:18, 2:18] = True
    mask[7:12, 7:12] = False
    mask[0, 0] = mask[1, 1] = True
    polygons = _mask_to_polygons(mask)
    assert any(p.is_hole for p in polygons)
    np.testing.assert_array_equal(fill(polygons, Equirectangular(20, 20)), mask)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Box2D(0, 0, 0, 1),
        lambda: Box2D(0, 0, 1, 1, score=np.nan),
        lambda: SphericalRegion(Box2D(-1, 0, 5, 5), Perspective(10, 10)),
        lambda: SphericalRegion(Box2D(0, -1, 5, 5), Equirectangular(10, 10)),
        lambda: SphericalRegion(Box2D(0, 0, 11, 5), Equirectangular(10, 10)),
        lambda: SphericalRegion.from_dict({"schema_version": 99}),
    ],
)
def test_invalid_region_contracts(factory):
    with pytest.raises(ValueError):
        factory()

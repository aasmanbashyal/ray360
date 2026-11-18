import json

import numpy as np
import pytest

from ray360 import Equirectangular, Perspective, coverage_views, geometry
from ray360.geometry import (
    angular_distance,
    fibonacci_sphere,
    lonlat_to_rays,
    normalize,
    rays_to_lonlat,
)
from ray360.projections import projection_from_dict, projection_to_dict


@pytest.mark.parametrize(
    "projection",
    [
        Equirectangular(800, 400),
        Equirectangular(800, 400, yaw=21, pitch=-13, roll=8),
        Perspective(641, 479, hfov=110),
        Perspective(320, 240, yaw=179, pitch=85, roll=34),
    ],
)
def test_roundtrip_and_serialization(projection):
    xy = np.random.default_rng(42).uniform(0.01, 0.99, (300, 2)) * projection.size
    rays, valid = projection.unproject(xy)
    actual, visible = projection.project(rays)
    np.testing.assert_allclose(actual, xy, atol=1e-8)
    assert valid.all() and visible.all()
    assert (
        projection_from_dict(json.loads(json.dumps(projection_to_dict(projection)))) == projection
    )


def test_coordinate_conventions_and_visibility():
    ll = np.array([[0, 0], [90, 0], [-90, 0], [-180, 0], [0, 90], [0, -90]])
    rays = lonlat_to_rays(ll)
    np.testing.assert_allclose(rays[:4], [[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 0, -1]], atol=1e-12)
    np.testing.assert_allclose(rays_to_lonlat(rays), ll, atol=1e-12)
    ray, _ = Perspective(100, 100, yaw=90, pitch=45).unproject([50, 50])
    np.testing.assert_allclose(rays_to_lonlat(ray), [90, 45], atol=1e-12)
    assert Perspective(100, 100).project([[0, 0, -1], [0, 0, 1], [100, 0, 1]])[1].tolist() == [
        False,
        True,
        False,
    ]
    assert Equirectangular(360, 180).unproject([[370, 90], [5, -1]])[1].tolist() == [True, False]
    np.testing.assert_allclose(normalize([1e300, 0, 0]), [1, 0, 0])
    assert angular_distance([0, 0, 1], [0, 0, -1]) == pytest.approx(180)


def test_arm64_batch_rotation_matches_known_longitudes(monkeypatch):
    monkeypatch.setattr(geometry, "_LINUX_ARM64", True)
    longitude = np.linspace(-150, 150, 300).reshape(2, 150)
    latitude = np.linspace(-75, 75, 300).reshape(longitude.shape)
    world_rays = lonlat_to_rays(np.stack((longitude, latitude), axis=-1))
    target = Equirectangular(360, 180, yaw=23)
    pixels, valid = target.project(world_rays)
    expected = np.stack((longitude - 23 + 180, 90 - latitude), axis=-1)
    np.testing.assert_allclose(pixels, expected, atol=1e-12, rtol=0)
    assert valid.all()
    np.testing.assert_allclose(target.unproject(expected)[0], world_rays, atol=1e-14, rtol=0)
    with pytest.raises(ValueError):
        target.project([[1]])


@pytest.mark.parametrize(
    "width,height,hfov,overlap", [(640, 640, 90, 0.2), (960, 540, 100, 0), (200, 300, 70, 0.3)]
)
def test_coverage_has_no_gaps_including_poles(width, height, hfov, overlap):
    rays = np.concatenate((fibonacci_sphere(10000), [[0, 1, 0], [0, -1, 0]]))
    covered = np.zeros(len(rays), bool)
    for projection in coverage_views(width, height, hfov, overlap):
        covered |= projection.project(rays)[1]
    assert covered.all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Perspective(2, 2, hfov=180),
        lambda: Equirectangular(0, 1),
        lambda: Equirectangular(True, 1),
        lambda: Equirectangular(2.0, 1),
        lambda: Perspective(1, 1, yaw=np.nan),
        lambda: coverage_views(overlap=1),
        lambda: lonlat_to_rays([0, 91]),
        lambda: normalize([0, 0, 0]),
        lambda: Perspective(2, 2).unproject([np.nan, 0]),
        lambda: Equirectangular(2, 2).unproject([0, 0, 0]),
        lambda: fibonacci_sphere(0),
        lambda: fibonacci_sphere(1.5),
        lambda: projection_from_dict({"kind": "fisheye"}),
    ],
)
def test_invalid_inputs(factory):
    with pytest.raises(ValueError):
        factory()

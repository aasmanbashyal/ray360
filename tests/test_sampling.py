from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from ray360 import Equirectangular, MapCache, Panorama, Perspective, ProjectionMap
from ray360.remap import sample


def test_independent_perspective_reference():
    """An analytic latitude/longitude field gives an independent pixel oracle."""
    h, w = 360, 720
    yy, xx = np.mgrid[:h, :w]
    image = np.stack(((xx + 0.5) / w * 360 - 180, 90 - (yy + 0.5) / h * 180), -1)
    view = Panorama(image).view(Perspective(81, 61, hfov=100), cache=None)
    y, x = np.mgrid[:61, :81]
    focal = 81 / (2 * np.tan(np.deg2rad(50)))
    dx, dy = (x + 0.5 - 81 / 2) / focal, (61 / 2 - y - 0.5) / focal
    expected = np.stack(
        (np.rad2deg(np.arctan(dx)), np.rad2deg(np.arctan2(dy, np.sqrt(1 + dx**2)))), -1
    )
    np.testing.assert_allclose(view.image, expected, atol=3e-5)


def test_seam_and_vertical_clamping():
    image = np.array([[0, 10, 20, 30]], np.float32)
    np.testing.assert_allclose(sample(image, np.array([[0, 0.5], [4, 0.5]]), wrap=True), [15, 15])
    np.testing.assert_allclose(sample(image, np.array([[0.5, -10], [3.5, 10]])), [0, 30])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32, np.float64])
def test_dtype_and_chunked_output(dtype):
    image = np.random.default_rng(2).integers(0, 100, (30, 60, 3)).astype(dtype)
    mapping = ProjectionMap.build(
        Equirectangular(60, 30), Perspective(25, 23, yaw=179), chunk_rows=3
    )
    expected = mapping.apply(image, chunk_rows=1)
    np.testing.assert_array_equal(mapping.apply(image, chunk_rows=100), expected)
    assert expected.dtype == dtype
    with pytest.raises(ValueError):
        mapping.xy[0, 0] = 0


@pytest.mark.parametrize("dtype", [bool, np.int64, np.uint64])
def test_nearest_preserves_boolean_and_large_integer_labels(dtype):
    image = np.array([[0, 1], [1, 0]], dtype=dtype)
    if dtype is not bool:
        image[image == 1] = np.iinfo(dtype).max
    grid = np.array([[[0.5, 0.5], [1.5, 0.5]], [[0.5, 1.5], [1.5, 1.5]]])
    np.testing.assert_array_equal(sample(image, grid, interpolation="nearest"), image)
    with pytest.raises(ValueError, match="nearest"):
        sample(image, grid)


def test_cache_budget_disable_eviction_and_concurrency():
    source = Equirectangular(40, 20)
    cache = MapCache(1000)
    a = cache.get(source, Perspective(10, 10))
    assert cache.get(source, Perspective(10, 10)) is a
    assert cache.nbytes == 900
    cache.get(source, Perspective(10, 10, yaw=90))
    assert cache.get(source, Perspective(10, 10)) is not a
    cache.get(source, Perspective(20, 20))
    assert cache.nbytes <= 1000
    disabled = MapCache(0)
    disabled.get(source, Perspective(10, 10))
    assert disabled.nbytes == 0
    image = np.arange(800, dtype=np.uint16).reshape(20, 40)
    targets = [Perspective(10, 10, yaw=yaw) for yaw in (-180, -90, 0, 90)] * 4
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda p: cache.get(source, p).apply(image), targets))
    for target, output in zip(targets, results, strict=True):
        np.testing.assert_array_equal(output, ProjectionMap.build(source, target).apply(image))
    assert cache.nbytes <= cache.max_bytes
    cache.clear()
    assert cache.nbytes == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MapCache(-1),
        lambda: MapCache(2.5),
        lambda: ProjectionMap.build(Equirectangular(4, 2), Perspective(2, 2), chunk_rows=0),
        lambda: sample(np.zeros((0, 3)), np.zeros((1, 2))),
        lambda: sample(np.zeros((2, 2)), np.zeros((1, 2)), interpolation="cubic"),
        lambda: Panorama(np.zeros((2, 2, 0))),
        lambda: Panorama(np.full((2, 2), "bad")),
    ],
)
def test_invalid_sampling(factory):
    with pytest.raises(ValueError):
        factory()


def test_map_rejects_wrong_source_target_and_image_size():
    with pytest.raises(TypeError):
        ProjectionMap.build(Perspective(2, 2), Perspective(2, 2))
    mapping = ProjectionMap.build(Equirectangular(4, 2), Perspective(2, 2))
    with pytest.raises(ValueError):
        mapping.apply(np.zeros((4, 2)))

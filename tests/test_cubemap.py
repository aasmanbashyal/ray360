import numpy as np
import pytest

from ray360 import (
    Equirectangular,
    MapCache,
    Panorama,
    cube_projections,
    from_cubemap,
    to_cubemap,
)


def direction_faces(size):
    """Independent XYZ color fields, including the pole faces' image orientation."""
    v, u = np.mgrid[:size, :size].astype(float)
    u, v = (u + 0.5) * 2 / size - 1, (v + 0.5) * 2 / size - 1
    one = np.ones_like(u)
    fields = {
        "front": (u, -v, one),
        "right": (one, -v, -u),
        "back": (-u, -v, -one),
        "left": (-one, -v, u),
        "up": (u, one, v),
        "down": (u, -one, -v),
    }
    return {
        name: (a := np.stack(channels, -1)) / np.linalg.norm(a, axis=-1, keepdims=True)
        for name, channels in fields.items()
    }


def test_face_orientation_and_panorama_conversion_match_independent_xyz_fields():
    projection = Equirectangular(1024, 512, yaw=17, pitch=-23, roll=11)
    rays, _ = projection.unproject(projection.pixel_grid())
    cache = MapCache(200_000)
    actual = to_cubemap(Panorama(rays, projection), 33, cache=cache)
    expected = direction_faces(33)
    assert list(actual) == ["front", "right", "back", "left", "up", "down"]
    for name, image in actual.items():
        np.testing.assert_allclose(image, expected[name], atol=2e-5)
        independent, _ = cube_projections(33)[name].unproject(
            cube_projections(33)[name].pixel_grid()
        )
        np.testing.assert_allclose(independent, expected[name], atol=1e-14)
    assert 0 < cache.nbytes <= cache.max_bytes


@pytest.mark.parametrize("rotation", [(0, 0, 0), (19, -37, 52)])
def test_inverse_xyz_including_cube_edges_and_poles(rotation):
    target = Equirectangular(360, 180, yaw=rotation[0], pitch=rotation[1], roll=rotation[2])
    expected, _ = target.unproject(target.pixel_grid())
    faces = direction_faces(32)
    result = from_cubemap(faces, target, chunk_rows=7)
    np.testing.assert_allclose(result, expected, atol=0.006)
    np.testing.assert_array_equal(result, from_cubemap(faces, target, chunk_rows=180))
    # Rotating by half a pixel places target centers exactly on 45-degree cube edges.
    edges = Equirectangular(360, 180, yaw=0.5, pitch=0.5)
    rays, _ = edges.unproject(edges.pixel_grid())
    np.testing.assert_allclose(from_cubemap(faces, edges), rays, atol=0.006)


def test_bilinear_uses_adjacent_face_instead_of_clamping_at_edge():
    faces = {name: np.zeros((8, 8), np.float32) for name in cube_projections(8)}
    faces["right"][:] = 100
    # Center pixel faces precisely along the front/right boundary.
    target = Equirectangular(1, 1, yaw=45)
    assert from_cubemap(faces, target)[0, 0] == pytest.approx(50, abs=1e-5)
    assert from_cubemap(faces, target, interpolation="nearest")[0, 0] in (0, 100)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.int64, np.uint64, bool])
def test_nearest_roundtrip_preserves_label_ids_and_dtype(dtype):
    labels = np.zeros((32, 64), dtype=dtype)
    value = (
        True if dtype is bool else (np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 4)
    )
    labels[:, 32:] = value
    faces = to_cubemap(Panorama(labels), 32, interpolation="nearest", cache=None)
    restored = from_cubemap(faces, Equirectangular(64, 32), interpolation="nearest")
    assert restored.dtype == labels.dtype
    assert set(np.unique(restored)) == {0, value}
    assert np.mean(restored != labels) < 0.01


def test_invalid_cube_faces_and_modes():
    good = {name: np.zeros((8, 8), np.uint8) for name in cube_projections(8)}
    target = Equirectangular(16, 8)
    for invalid in (
        {},
        {**good, "extra": good["front"]},
        {**good, "up": np.zeros((8, 7))},
        {**good, "up": np.zeros((8, 8), np.float32)},
        {**good, "up": np.zeros((8, 8, 3), np.uint8)},
    ):
        with pytest.raises(ValueError):
            from_cubemap(invalid, target)
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            cube_projections(size)
    with pytest.raises(TypeError):
        to_cubemap(np.zeros((8, 16)))
    with pytest.raises(TypeError):
        from_cubemap(good, cube_projections(8)["front"])
    with pytest.raises(ValueError):
        from_cubemap(good, target, chunk_rows=0)
    with pytest.raises(ValueError):
        from_cubemap(good, target, interpolation="cubic")
    with pytest.raises(ValueError, match="nearest"):
        from_cubemap({name: face.astype(np.int64) for name, face in good.items()}, target)

import numpy as np
import pytest

from ray360 import (
    Box2D,
    Equirectangular,
    MapCache,
    Panorama,
    cube_projections,
    from_cubemap,
    from_cubemap_cross,
    to_cubemap,
    to_cubemap_cross,
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


@pytest.mark.parametrize("size", [1, 33])
def test_cross_positions_and_orientations_match_independent_xyz_fields(size):
    projection = Equirectangular(1024, 512, yaw=17, pitch=-23, roll=11)
    rays, _ = projection.unproject(projection.pixel_grid())
    faces = direction_faces(size)
    blank = np.zeros((size, size, 3))
    expected = np.concatenate(
        [
            np.concatenate((blank, faces["up"], blank, blank), axis=1),
            np.concatenate(
                tuple(faces[name] for name in ("left", "front", "right", "back")), axis=1
            ),
            np.concatenate((blank, faces["down"], blank, blank), axis=1),
        ]
    )
    actual = to_cubemap_cross(Panorama(rays, projection), size, cache=None)
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert actual.dtype == rays.dtype
    assert actual.flags.c_contiguous


def test_cross_matches_separate_faces_with_strided_images_and_reused_maps():
    size = 271  # Wide enough to exercise reduced chunk rows and a partial final chunk.
    rgb = np.random.default_rng(0).integers(0, 256, (48, 192, 3), dtype=np.uint8)[:, ::2]
    max_label = np.uint64(np.iinfo(np.uint64).max)
    labels = np.where(rgb[..., 0] < 128, np.uint64(0), max_label)
    cache = MapCache()
    for image, interpolation in ((rgb, "bilinear"), (labels, "nearest")):
        panorama = Panorama(image)
        faces = to_cubemap(panorama, size, interpolation=interpolation, cache=cache)
        cross = to_cubemap_cross(panorama, size, interpolation=interpolation, cache=cache)
        assert cross.dtype == image.dtype
        assert cross.shape == (3 * size, 4 * size, *image.shape[2:])
        expected = np.zeros_like(cross)
        expected[:size, size : 2 * size] = faces["up"]
        expected[size : 2 * size] = np.concatenate(
            tuple(faces[name] for name in ("left", "front", "right", "back")), axis=1
        )
        expected[2 * size :, size : 2 * size] = faces["down"]
        np.testing.assert_array_equal(cross, expected)
    assert cache.nbytes == 6 * size * size * 9
    assert cross.max() == max_label


def test_cross_validates_panorama_size_and_interpolation():
    panorama = Panorama(np.zeros((8, 16), dtype=bool))
    with pytest.raises(TypeError, match="Panorama"):
        to_cubemap_cross(panorama.image)
    with pytest.raises(ValueError, match="size"):
        to_cubemap_cross(panorama, 0)
    with pytest.raises(ValueError, match="nearest"):
        to_cubemap_cross(panorama, 1)
    with pytest.raises(ValueError, match="interpolation"):
        to_cubemap_cross(panorama, 1, interpolation="cubic")


def test_reverse_cross_uses_only_faces_and_matches_rotated_cube_reconstruction():
    size = 16
    panorama = Panorama(np.arange(16 * 32 * 3, dtype=np.uint16).reshape(16, 32, 3))
    cross = to_cubemap_cross(panorama, size)
    cross[:size, :size] = 65535
    target = Equirectangular(73, 37, yaw=28, pitch=-17, roll=8)
    actual, groups = from_cubemap_cross(cross, target, chunk_rows=7)
    expected = from_cubemap(to_cubemap(panorama, size), target)
    np.testing.assert_array_equal(actual, expected)
    assert groups == []


def test_reverse_cross_seam_groups_metadata_and_padding():
    cross = np.zeros((96, 128), np.uint64)
    cross[32:64, 96:] = np.iinfo(np.uint64).max
    box = Box2D(103, 40, 121, 57, label="back", score=0.9, track_id=4, attributes={"frame": 7})
    pixels, (seam, padding) = from_cubemap_cross(
        cross, boxes=[box, (1, 1, 10, 10)], interpolation="nearest"
    )
    assert pixels.shape == (64, 128) and pixels.dtype == np.uint64
    assert {int(v) for v in np.unique(pixels)} == {0, np.iinfo(np.uint64).max}
    assert len(seam) == 2 and padding == []
    assert any(polygon.bbox().x1 == 0 for polygon in seam)
    assert any(polygon.bbox().x2 == 128 for polygon in seam)
    assert all(polygon.metadata() == box.metadata() for polygon in seam)


def test_reverse_cross_joins_faces_and_splits_only_at_the_target_seam():
    cross = np.zeros((96, 128, 3), np.uint8)
    # Straddle the Right/Back join at world longitude 135 degrees.
    box = np.array([86.5, 41.5, 105.5, 55.5])
    _, (joined,) = from_cubemap_cross(cross, Equirectangular(360, 180), boxes=[box])
    assert len(joined) == 1
    bounds = joined[0].bbox()
    assert bounds.x1 < 315 < bounds.x2
    # Rotate so the panorama seam coincides with that same cube edge.
    target = Equirectangular(360, 180, yaw=-45)
    _, (wrapped,) = from_cubemap_cross(cross, target, boxes=[box])
    assert len(wrapped) == 2
    left, right = sorted((polygon.bbox() for polygon in wrapped), key=lambda bbox: bbox.x1)
    assert left.x1 == 0 and right.x2 == target.width
    assert left.x2 < 30 and right.x1 > 330


def test_reverse_cross_validates_layout_and_boxes():
    with pytest.raises(ValueError, match="horizontal cubemap cross"):
        from_cubemap_cross(np.zeros((96, 127, 3), np.uint8))
    cross = np.zeros((24, 32, 3), np.uint8)
    with pytest.raises(TypeError, match="Equirectangular"):
        from_cubemap_cross(cross, cube_projections(8)["front"])
    with pytest.raises(ValueError, match="four xyxy"):
        from_cubemap_cross(cross, boxes=[(1, 2, 3)])
    with pytest.raises(ValueError, match="finite coordinates and positive"):
        from_cubemap_cross(cross, boxes=[(3, 1, 2, 4)])


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

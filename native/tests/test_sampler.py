import numpy as np
import pytest
from ray360 import (
    Equirectangular,
    Panorama,
    Perspective,
    ProjectionMap,
    from_cubemap,
    from_cubemap_cross,
    remap,
    to_cubemap,
)
from ray360.remap import _sample_numpy as reference
from ray360_native import _native, sample


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_reverse_cross_uses_rust_for_every_face_sample(monkeypatch, interpolation):
    cross = np.random.default_rng(71).integers(0, 256, (48, 64, 3), np.uint8)
    target = Equirectangular(95, 47, yaw=17, pitch=-7, roll=4)
    with monkeypatch.context() as numpy_backend:
        numpy_backend.setattr(remap, "_native_sampler", lambda: None)
        expected, _ = from_cubemap_cross(cross, target, interpolation=interpolation, chunk_rows=7)

    provider = remap._native_sampler()
    calls = []

    def require_native(image, xy, **kwargs):
        pixels = provider(image, xy, **kwargs)
        assert pixels is not None
        calls.append(image)
        return pixels

    monkeypatch.setattr(remap, "_native_sampler", lambda: require_native)
    actual, _ = from_cubemap_cross(cross, target, interpolation=interpolation, chunk_rows=7)
    assert len({id(face) for face in calls}) == 6
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("coordinate_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(7, 11), (7, 11, 1), (7, 11, 3), (7, 11, 5)])
@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
def test_random_differential(coordinate_dtype, shape, wrap, interpolation):
    rng = np.random.default_rng(246)
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    xy = rng.uniform(-50, 50, (19, 23, 2)).astype(coordinate_dtype)
    xy[0, :4] = [[np.nan, 0], [0, np.inf], [-np.inf, 0], [0, -np.inf]]
    valid = rng.random((19, 23)) > 0.2
    expected = reference(image, xy, wrap=wrap, interpolation=interpolation, valid=valid)
    actual = sample(image, xy, wrap=wrap, interpolation=interpolation, valid=valid)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == image.dtype
    assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_independent_seam_poles_and_rounding(dtype):
    image = np.array([[0, 10, 20, 30]], np.uint8)
    xy = np.array([[0, 0.5], [4, 0.5], [0.5, -10], [3.5, 10]], dtype)
    np.testing.assert_array_equal(sample(image, xy, wrap=True), [15, 15, 0, 30])
    halves = np.array([[0, 1, 2, 3]], np.uint8)
    # NumPy rounds ties to even, not away from zero.
    np.testing.assert_array_equal(
        sample(halves, np.array([[1, 0.5], [2, 0.5], [3, 0.5]], dtype)), [0, 2, 2]
    )
    boundary = np.array([[0.99999994, 0.5], [1, 0.5], [1.0000001, 0.5]], dtype)
    np.testing.assert_array_equal(
        sample(image, boundary, interpolation="nearest"),
        reference(image, boundary, interpolation="nearest"),
    )


@pytest.mark.parametrize(
    "dtype", [bool, np.uint16, np.int32, np.int64, np.uint64, np.float32, np.float64]
)
def test_fallback_preserves_labels_and_dtypes(dtype):
    image = np.array([[0, 1], [1, 0]], dtype=dtype)
    if dtype in (np.int64, np.uint64):
        image[image == 1] = np.iinfo(dtype).max
    xy = np.array([[[0.5, 0.5], [1.5, 0.5]], [[0.5, 1.5], [1.5, 1.5]]])
    np.testing.assert_array_equal(sample(image, xy, interpolation="nearest"), image)
    if dtype in (bool, np.int64, np.uint64):
        with pytest.raises(ValueError, match="nearest"):
            sample(image, xy)


def test_strides_empty_and_scalar_maps():
    image = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    xy = np.array([[0.5, 0.5], [2.5, 1.5], [3.3, 2.8]])
    for pixels, coords in [
        (image[:, ::-1], xy),
        (image, xy[::-1]),
        (np.asfortranarray(image), xy),
        (image, xy[:0]),
        (image, xy[0]),
    ]:
        np.testing.assert_array_equal(sample(pixels, coords), reference(pixels, coords))
    np.testing.assert_array_equal(sample(image, xy, valid=False), reference(image, xy, valid=False))


@pytest.mark.parametrize("interpolation, value", [("nearest", 20), ("bilinear", 15)])
@pytest.mark.parametrize("channels", [None, 3])
def test_single_coordinate_returns_array_and_respects_masks(interpolation, value, channels):
    image = np.array([[10, 20]], np.uint8)
    if channels is not None:
        image = np.repeat(image[..., None], channels, axis=2)
    shape = () if channels is None else (channels,)
    for x, valid, expected in [(1.0, True, value), (1.0, False, 0), (np.nan, True, 0)]:
        xy = np.array([x, 0.5], np.float32)
        mask = np.array(valid)
        for sampler in (sample, reference):
            result = sampler(image, xy, interpolation=interpolation, valid=mask)
            assert isinstance(result, np.ndarray) and result.shape == shape
            np.testing.assert_array_equal(result, np.full(shape, expected, dtype=np.uint8))


def test_native_rejects_invalid_shapes_and_layouts():
    image = np.zeros((3, 4, 3), np.uint8)
    xy = np.zeros((2, 2, 2), np.float32)
    for pixels, coords, mask in [
        (image[:, ::-1], xy, None),
        (image, xy, np.ones(2, np.uint8)),
        (image, np.zeros((4, 3), np.float32), None),
        (image[:0], xy, None),
    ]:
        with pytest.raises(ValueError):
            _native.sample_u8_f32(pixels, coords, mask)


def test_unaligned_and_big_endian_coordinates_fall_back():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    xy = np.ndarray((2, 2), dtype=np.float32, buffer=bytearray(17), offset=1)
    xy[:] = [[0.5, 0.5], [2.25, 1.25]]
    assert not xy.flags.aligned
    np.testing.assert_array_equal(sample(image, xy), reference(image, xy))
    with pytest.raises(ValueError, match="aligned"):
        _native.sample_u8_f32(image, xy)
    swapped = xy.astype(">f4")
    np.testing.assert_array_equal(sample(image, swapped), reference(image, swapped))


def test_noncanonical_boolean_mask_bytes():
    image = np.array([[10, 20, 30]], np.uint8)
    xy = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
    mask = np.array([0, 2, 255], np.uint8).view(bool)
    np.testing.assert_array_equal(sample(image, xy, valid=mask), [0, 20, 30])


def test_float32_coordinates_on_wide_images_match_numpy():
    image = np.arange(65537, dtype=np.uint8)[None, :]
    xy = np.array([[-0.00075, 0.5], [65536.5, 0.5], [65537.25, 0.5]], np.float32)
    for interpolation in ("nearest", "bilinear"):
        kwargs = {"wrap": True, "interpolation": interpolation}
        np.testing.assert_array_equal(sample(image, xy, **kwargs), reference(image, xy, **kwargs))


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (180, 0), (45, 90), (-70, -90)])
def test_views_preprocessing_and_cubemap(monkeypatch, yaw, pitch):
    image = np.random.default_rng(41).integers(0, 256, (64, 128, 3), np.uint8)
    pano = Panorama(image)
    target = Perspective(31, 23, yaw=yaw, pitch=pitch)
    mapping = ProjectionMap.build(pano.projection, target)
    with monkeypatch.context() as numpy_backend:
        numpy_backend.setattr(remap, "_native_sampler", lambda: None)
        expected = pano.view(target).crop(0.25, 0.75, 29, 21).resize(45, 33).letterbox(48, 48)
        faces = to_cubemap(pano, size=17)
        reconstructed = from_cubemap(faces, Equirectangular(96, 48))

    calls = []
    for name in ("sample_u8_f32", "sample_u8_f64"):
        kernel = getattr(_native, name)

        def track(*args, name=name, kernel=kernel):
            calls.append(name)
            return kernel(*args)

        monkeypatch.setattr(_native, name, track)
    actual = pano.view(target).crop(0.25, 0.75, 29, 21).resize(45, 33).letterbox(48, 48)
    assert calls == ["sample_u8_f32", *["sample_u8_f64"] * 3]
    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.valid, expected.valid)
    assert actual.transform.to_dict() == expected.transform.to_dict()
    np.testing.assert_array_equal(
        mapping.apply(image, chunk_rows=1), mapping.apply(image, chunk_rows=50)
    )
    for name, pixels in to_cubemap(pano, size=17).items():
        np.testing.assert_array_equal(pixels, faces[name])
    np.testing.assert_array_equal(from_cubemap(faces, Equirectangular(96, 48)), reconstructed)

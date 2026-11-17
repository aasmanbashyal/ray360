import numpy as np
import pytest
from ray360 import Equirectangular, Panorama, Perspective, ViewTransform
from ray360.remap import _sample_numpy as reference
from ray360_native import PreparedView, _native


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 2, 3), (3, 1, 4), (7, 11), (7, 11, 3), (7, 11, 4), (7, 11, 5)]
)
@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 16, 33])
@pytest.mark.parametrize("wrap", [False, True])
def test_prepared_scalar_simd_and_tail_match_numpy(shape, count, wrap):
    rng = np.random.default_rng(192)
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    xy = rng.uniform(-20, 30, (count, 2)).astype(np.float32)
    if count:
        # Exercise final source pixel, where a SIMD 4-byte RGB load would overread.
        xy[-1] = [shape[1] - 0.5, shape[0] - 0.5]
        xy[0] = [np.nan, np.inf]
    valid = rng.integers(0, 3, count, dtype=np.uint8)
    plan = _native.PreparedBilinear(shape[1], shape[0], xy, valid, wrap)
    expected = reference(image, xy, valid=valid != 0, wrap=wrap)
    np.testing.assert_array_equal(plan.apply(image, "scalar"), expected)
    np.testing.assert_array_equal(plan.apply(image), expected)
    if _native.avx2_available() and (len(shape) == 2 or shape[2] in (1, 3, 4)):
        np.testing.assert_array_equal(plan.apply(image, "avx2"), expected)
    # Preparation owns the indices/weights/mask, rather than borrowing mutable inputs.
    xy[:] = 0
    valid[:] = 0
    np.testing.assert_array_equal(plan.apply(image), expected)


def test_simd_ties_and_invalid_lanes():
    image = np.array([[0, 1, 2, 3]], np.uint8)
    xy = np.tile(np.array([[1, 0.5], [2, 0.5], [3, 0.5]], np.float32), (9, 1))
    plan = _native.PreparedBilinear(4, 1, xy)
    np.testing.assert_array_equal(plan.apply(image), np.tile([0, 2, 2], 9))


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (180, 0, 0), (27, 90, 15), (-63, -90, -20)])
def test_fused_geometry_and_single_pass_reference(yaw, pitch, roll):
    source = Equirectangular(192, 96, yaw=17, pitch=-13, roll=9)
    view = Perspective(65, 47, yaw=yaw, pitch=pitch, roll=roll, hfov=112)
    transform = (
        ViewTransform(source, view).crop(0.25, 0.75, 61, 43).resize(53, 31).letterbox(57, 49)
    )
    prepared = PreparedView(transform)
    h, w = prepared.valid.shape
    y, x = np.mgrid[:h, :w]
    points = np.stack((x + 0.5, y + 0.5), -1)
    expected_xy, valid = transform.to_panorama(points)
    np.testing.assert_array_equal(prepared.valid, valid)
    # Longitude comparison is circular, particularly at the seam.
    dx = (
        prepared.xy[..., 0] - expected_xy[..., 0] + source.width / 2
    ) % source.width - source.width / 2
    np.testing.assert_allclose(dx[valid], 0, atol=2e-5)
    np.testing.assert_allclose(prepared.xy[..., 1][valid], expected_xy[..., 1][valid], atol=1e-5)
    pano = Panorama(
        np.random.default_rng(7).integers(0, 256, (96, 192, 3), np.uint8),
        source,
        source_id="test",
        timestamp=1.25,
    )
    result = prepared.apply(pano)
    # Validate against the float32 single-pass map, not a lossy eager resize chain.
    expected = reference(pano.image, expected_xy.astype(np.float32), wrap=True, valid=valid)
    np.testing.assert_array_equal(result.image, expected)
    np.testing.assert_array_equal(result.image, prepared.apply(pano, engine="scalar").image)
    assert result.transform is transform
    assert result.source_id == "test" and result.timestamp == 1.25
    assert prepared.nbytes > prepared.xy.nbytes
    np.testing.assert_array_equal(result.valid, valid)


def test_padding_only_is_exact_and_non_native_fallback():
    pano = Panorama(np.random.default_rng(42).integers(0, 256, (64, 128, 3), np.uint8))
    projection = Perspective(32, 32, yaw=179)
    prepared = PreparedView(ViewTransform(pano.projection, projection).letterbox(32, 48))
    np.testing.assert_array_equal(
        prepared.apply(pano).image, pano.view(projection).letterbox(32, 48).image
    )
    floats = Panorama(pano.image.astype(np.float32))
    np.testing.assert_array_equal(
        prepared.apply(floats).image,
        reference(floats.image, prepared.xy, wrap=True, valid=prepared.valid),
    )
    with pytest.raises(ValueError, match="uint8"):
        prepared.apply(floats, engine="avx2")
    with pytest.raises(ValueError, match="projection"):
        prepared.apply(Panorama(pano.image, Equirectangular(128, 64, yaw=10)))


def test_wide_prepared_panorama_keeps_numpy_pixel_parity():
    # NumPy 1.x promotes float32 coordinate arithmetic on a source this wide.
    # Keeping the math in float32 changes the rounding of this seam pixel.
    width = 65_537
    pixels = np.zeros((1, width), np.uint8)
    pixels[0, 0] = 255
    seam_x = np.float32(0.06274505704641342)
    yaw = float(seam_x) / width * 360 - 180
    pano = Panorama(pixels)
    prepared = PreparedView(ViewTransform(pano.projection, Perspective(1, 1, yaw=yaw)))

    assert prepared.nbytes == 0
    expected = reference(pixels, prepared.xy, wrap=True, valid=prepared.valid)
    np.testing.assert_array_equal(prepared.apply(pano).image, expected)
    with pytest.raises(ValueError, match="source dimensions"):
        prepared.apply(pano, engine="scalar")


def test_native_validation_and_unsupported_simd():
    xy = np.zeros((8, 2), np.float32)
    with pytest.raises(ValueError):
        _native.PreparedBilinear(0, 1, xy)
    with pytest.raises(ValueError):
        _native.PreparedBilinear(2, 2, xy, np.ones(3, np.uint8))
    plan = _native.PreparedBilinear(2, 2, xy)
    with pytest.raises(ValueError):
        plan.apply(np.zeros((3, 3), np.uint8))
    with pytest.raises(ValueError):
        plan.apply(np.zeros((2, 2), np.uint8), "invalid")
    with pytest.raises(ValueError, match="AVX2"):
        plan.apply(np.zeros((2, 2, 5), np.uint8), "avx2")


def test_entirely_padding_has_no_content():
    pano = Panorama(np.full((16, 32, 3), 255, np.uint8))
    transform = (
        ViewTransform(pano.projection, Perspective(16, 16)).letterbox(16, 32).crop(0, 0, 16, 4)
    )
    prepared = PreparedView(transform)
    assert prepared.nbytes == 0
    assert not prepared.valid.any()
    assert not prepared.apply(pano).image.any()
    assert not prepared.apply(pano, engine="scalar").image.any()


def test_maps_are_lazy_and_memory_is_accounted():
    pano = Panorama(np.random.default_rng(4).integers(0, 256, (64, 128, 3), np.uint8))
    transform = ViewTransform(pano.projection, Perspective(32, 24)).letterbox(32, 32)
    prepared = PreparedView(transform)
    # Full weights and exact indices fit within 14 bytes per active pixel on 64-bit.
    resident = prepared.nbytes
    assert 0 < resident <= 32 * 24 * 14
    output = prepared.apply(pano).image
    assert prepared.nbytes == resident
    assert "_maps" not in prepared.__dict__
    xy, valid = prepared.xy, prepared.valid
    assert prepared.xy is xy and prepared.valid is valid
    assert not xy.flags.writeable and not valid.flags.writeable
    assert prepared.nbytes == resident + xy.nbytes + valid.nbytes
    np.testing.assert_array_equal(output, reference(pano.image, xy, wrap=True, valid=valid))


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
@pytest.mark.parametrize("wrap", [False, True])
def test_compact_blocks_keep_sparse_positions_borders_and_tail(channels, wrap):
    rng = np.random.default_rng(178)
    image = rng.integers(0, 256, (13, 17, channels), dtype=np.uint8)
    xy = np.zeros((9, 17, 2), np.float32)
    valid = np.zeros((9, 17), np.uint8)
    border = np.array(
        [
            [0, 0],
            [17, 13],
            [-0.2, 0.2],
            [16.8, 12.8],
            [17.2, -0.2],
            [0.5, 12.5],
            [16.5, 0.5],
            [3.2, 4.7],
        ],
        np.float32,
    )
    # Complete interior block, mixed border block, skipped blocks, and a one-lane tail.
    xy.reshape(-1, 2)[8:16] = [3.2, 4.7]
    valid.ravel()[8:16] = 255
    xy.reshape(-1, 2)[32:40] = border
    valid.ravel()[32:40] = [1, 0, 255, 1, 0, 1, 1, 1]
    xy[-1, -1] = [16.5, 12.5]
    valid[-1, -1] = 1
    plan = _native.PreparedBilinear(17, 13, xy, valid, wrap)
    assert 0 < plan.nbytes <= 3 * 8 * 14
    expected = reference(image, xy, valid=valid != 0, wrap=wrap)
    for engine in ("scalar", "auto"):
        np.testing.assert_array_equal(plan.apply(image, engine), expected)
    if _native.avx2_available() and channels in (1, 3, 4):
        np.testing.assert_array_equal(plan.apply(image, "avx2"), expected)


@pytest.mark.parametrize("wrap", [False, True])
def test_compact_weights_remain_full_precision(wrap):
    rng = np.random.default_rng(9762)
    image = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    xy = rng.uniform([-2, -2], [55, 39], (101, 103, 2)).astype(np.float32)
    valid = rng.integers(0, 3, xy.shape[:-1], dtype=np.uint8)
    plan = _native.PreparedBilinear(53, 37, xy, valid, wrap)
    expected = reference(image, xy, valid=valid != 0, wrap=wrap)
    for engine in ("scalar", "auto"):
        np.testing.assert_array_equal(plan.apply(image, engine), expected)


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
@pytest.mark.parametrize("wrap", [False, True])
def test_lossless_packing_and_wide_exceptions(channels, wrap):
    rng = np.random.default_rng(55)
    image = rng.integers(0, 256, (256, 512, channels), dtype=np.uint8)
    xy = np.tile(np.array([256.75, 64.125], np.float32), (49, 1))
    valid = np.ones(49, np.uint8)
    # Tiny fractions need more than 16 fractional bits, despite small source indices.
    xy[8:16, 0] = np.nextafter(np.float32(0.5), np.float32(1))
    # Individually representable fractions, but index offsets exceed u16.
    xy[16:24, 1] = [1.5, 200.5] * 4
    # Mixed validity retains a compact block and correct output placement.
    valid[24:32:2] = 0
    # Float32 rounding can yield a weight of exactly 1.0; Q16 cannot store 65536.
    xy[32:40, 0] = np.nextafter(np.float32(0.5), np.float32(0))
    valid[40:48] = 0
    plan = _native.PreparedBilinear(512, 256, xy, valid, wrap)
    assert plan.packed_blocks == 3
    assert plan.wide_blocks == 3
    assert plan.nbytes <= 3 * 64 + 3 * 112
    expected = reference(image, xy, valid=valid != 0, wrap=wrap)
    for engine in ("scalar", "auto"):
        np.testing.assert_array_equal(plan.apply(image, engine), expected)
    if _native.avx2_available() and channels in (1, 3, 4):
        np.testing.assert_array_equal(plan.apply(image, "avx2"), expected)


@pytest.mark.parametrize("channels", [3, 4])
def test_packed_fractions_use_every_bit_without_quantization(channels):
    # Sweep all 16-bit fractions on either axis. The half-pixel subtraction is exact.
    fraction = np.arange(65536, dtype=np.float32) / np.float32(65536)
    xy = np.stack((fraction + np.float32(0.5), fraction[::-1] + np.float32(0.5)), -1)
    image = np.zeros((3, 3, channels), np.uint8)
    image[:2, :2, :3] = [[[0, 255, 67], [1, 123, 0]], [[255, 0, 141], [2, 75, 255]]]
    if channels == 4:
        image[:2, :2, 3] = [[1, 255], [0, 2]]
    plan = _native.PreparedBilinear(3, 3, xy)
    assert plan.packed_blocks == len(fraction) // 8 and plan.wide_blocks == 0
    expected = reference(image, xy, wrap=True)
    np.testing.assert_array_equal(plan.apply(image, "scalar"), expected)
    np.testing.assert_array_equal(plan.apply(image), expected)

import importlib.util

import numpy as np
import pytest

from ray360 import Box2D, Equirectangular, Perspective, SphericalRegion, from_cubemap_cross
from ray360.image import overlay, read_panorama, write_image

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="image extra is not installed"
)


def test_file_roundtrip_exact_names_and_overwrite_protection(tmp_path):
    image = np.full((20, 40, 3), 120, np.uint8)
    a, b = tmp_path / "capture.v1_0000.png", tmp_path / "capture.v1_0001.png"
    write_image(a, image)
    write_image(b, image + 10)
    np.testing.assert_array_equal(read_panorama(a).image, image)
    np.testing.assert_array_equal(read_panorama(b).image, image + 10)
    with pytest.raises(FileExistsError):
        write_image(a, image + 1)
    np.testing.assert_array_equal(read_panorama(a).image, image)
    write_image(a, image + 1, overwrite=True)
    np.testing.assert_array_equal(read_panorama(a).image, image + 1)
    with pytest.raises(ValueError):
        write_image(tmp_path / "bad.not_a_format", image)


@pytest.mark.parametrize("overwrite", [False, True])
def test_failed_jpeg_save_leaves_directory_untouched(tmp_path, overwrite):
    path = tmp_path / "capture.jpg"
    rgba = np.full((8, 16, 4), 120, np.uint8)
    if overwrite:
        write_image(path, rgba[:, :, :3])
        original = path.read_bytes()
    with pytest.raises(OSError, match="RGBA"):
        write_image(path, rgba, overwrite=overwrite)
    if overwrite:
        assert path.read_bytes() == original
        assert list(tmp_path.iterdir()) == [path]
    else:
        assert list(tmp_path.iterdir()) == []


def test_default_save_preserves_file_created_during_encoding(tmp_path, monkeypatch):
    from PIL import Image

    path = tmp_path / "shared.png"
    save = Image.Image.save

    def competing_save(image, *args, **kwargs):
        path.write_bytes(b"another writer")
        return save(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", competing_save)
    with pytest.raises(FileExistsError):
        write_image(path, np.zeros((2, 3), np.uint8))
    assert path.read_bytes() == b"another writer"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("dtype", ["<u2", ">u2"])
def test_reverse_cross_draws_uint16_rgb_as_an_eight_bit_preview(dtype):
    cross = np.full((48, 64, 3), [0x1745, 0x8A76, 0xFE34], dtype=dtype)
    boxes = [(19, 20, 27, 29)]
    original, (plain_group,) = from_cubemap_cross(cross, boxes=boxes)
    assert original.dtype == cross.dtype
    drawn, (drawn_group,) = from_cubemap_cross(cross, boxes=boxes, draw_boxes=True)
    assert drawn.dtype == np.uint8
    np.testing.assert_array_equal(drawn_group[0].vertices, plain_group[0].vertices)
    np.testing.assert_array_equal(drawn, overlay((original >> 8).astype(np.uint8), plain_group))
    assert (drawn == [255, 60, 40]).all(-1).any()
    assert (drawn == [0x17, 0x8A, 0xFE]).all(-1).any()


def test_outlines_do_not_mutate_input_or_connect_across_seam():
    image = np.zeros((90, 180, 3), np.uint8)
    region = SphericalRegion(Box2D(20, 20, 80, 80), Perspective(100, 100, yaw=180))
    result = overlay(image, [region], projection=Equirectangular(180, 90))
    assert result.any() and not image.any()
    assert not result[:, 60:120].any()
    assert overlay(image, [Box2D(10, 10, 30, 30)]).any()
    with pytest.raises(ValueError):
        overlay(image, [region])
    with pytest.raises(ValueError):
        overlay(image, [region], projection=Equirectangular(20, 10))


def test_reverse_cross_draws_joined_outlines_without_an_internal_cube_edge():
    cross = np.zeros((96, 128, 3), np.uint8)
    target = Equirectangular(360, 180)
    image, (joined, seam) = from_cubemap_cross(
        cross, target, boxes=[(54, 40, 74, 56), (103, 40, 121, 57)], draw_boxes=True
    )
    assert len(joined) == 1 and len(seam) == 2
    assert image.any() and not cross.any()
    # The Front/Right cube boundary at yaw 45 is inside the first box.
    assert not image[90, 225].any()
    assert image[:, :24].any() and image[:, -24:].any()


@pytest.mark.public_data
def test_actual_stitched_photos_through_full_geometry_workflow(public_photos, tmp_path):
    for index, path in enumerate(public_photos):
        panorama = read_panorama(path)
        view = (
            panorama.view(Perspective(160, 90, yaw=179, pitch=15))
            .crop(10, 5, 140, 80)
            .letterbox(160, 160)
        )
        regions = view.map_boxes([Box2D(40, 50, 120, 110, label="example")])
        assert len(regions) == 1
        points = np.array([[40, 50], [120, 110]])
        source_xy, valid = view.transform.to_panorama(points)
        recovered, visible = view.transform.from_panorama(source_xy)
        assert valid.all() and visible.all()
        np.testing.assert_allclose(recovered, points, atol=1e-7)
        result = overlay(panorama.image, regions, projection=panorama.projection)
        assert np.any(result != panorama.image)
        output = tmp_path / f"photo.v1_{index:04d}.png"
        write_image(output, view.image)
        assert output.stat().st_size > 0

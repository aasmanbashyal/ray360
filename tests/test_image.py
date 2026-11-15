import importlib.util

import numpy as np
import pytest

from ray360 import Box2D, Equirectangular, Perspective, SphericalRegion
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

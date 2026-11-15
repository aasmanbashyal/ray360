import json
from dataclasses import replace

import numpy as np
import pytest

from ray360 import Box2D, Equirectangular, Panorama, Perspective, ViewTransform


def panorama():
    return Panorama(np.ones((100, 200, 3), np.uint8), source_id="capture", timestamp=1.25)


@pytest.mark.parametrize(
    "yaw,pitch,roll", [(0, 0, 0), (179, 25, 30), (-179, -30, -25), (30, 90, 45)]
)
def test_preprocessing_point_inverse_and_persistence(yaw, pitch, roll):
    view = panorama().view(Perspective(200, 100, yaw=yaw, pitch=pitch, roll=roll))
    view = view.crop(20, 10, 160, 80).resize(320, 160).letterbox(640, 640)
    points = np.random.default_rng(3).uniform([20, 200], [620, 430], (100, 2))
    xy, valid = view.transform.to_panorama(points)
    recovered, visible = view.transform.from_panorama(xy)
    assert valid.all() and visible.all()
    np.testing.assert_allclose(recovered, points, atol=1e-8)
    stored = json.loads(json.dumps(view.provenance()))
    restored = ViewTransform.from_dict(stored["transform"])
    np.testing.assert_allclose(restored.to_panorama(points)[0], xy)
    assert stored["source_id"] == "capture" and stored["timestamp"] == 1.25
    assert not view.valid[:100].any() and view.valid[320].all()


def test_padding_boxes_and_points_clip_or_discard():
    view = panorama().view(Perspective(100, 50)).letterbox(100, 100)
    assert view.map_boxes([Box2D(10, 0, 20, 10)]) == []
    partial = Box2D(
        10, 10, 20, 40, label="person", score=0.9, track_id=7, attributes={"camera": "a"}
    )
    result = view.map_boxes([partial])[0]
    np.testing.assert_allclose(result.box.xyxy, [10, 0, 20, 15])
    assert result.box.metadata() == partial.metadata()
    assert view.transform.to_panorama([[10, 0], [50, 50], [-10, 50]])[1].tolist() == [
        False,
        True,
        False,
    ]
    assert not view.image[~view.valid].any()


def test_exact_content_edges_remain_valid_after_roundtrip():
    transform = ViewTransform(
        Equirectangular(360, 180, yaw=-41, roll=17),
        Perspective(100, 100, yaw=31, pitch=22, roll=14),
    )
    for current, corners in (
        (transform, [[0, 0], [100, 0], [100, 100], [0, 100]]),
        (
            transform.crop(10, 20, 80, 60).resize(160, 120).letterbox(200, 200),
            [[0, 25], [200, 25], [200, 175], [0, 175]],
        ),
    ):
        xy, valid = current.to_panorama(corners)
        recovered, visible = current.from_panorama(xy)
        assert valid.all() and visible.all()
        np.testing.assert_allclose(recovered, corners, atol=1e-8)
        assert not current.valid([[-1e-5, 50], [current.size[0] + 1e-5, 50]]).any()


def test_crop_history_cannot_restore_removed_content():
    view = panorama().view(Perspective(200, 100)).crop(50, 0, 100, 100).letterbox(200, 100)
    assert not view.image[:, :50].any()
    assert view.map_boxes([Box2D(10, 10, 20, 20)]) == []
    assert len(view.map_boxes([Box2D(40, 10, 60, 20)])) == 1
    restored = ViewTransform.from_dict(json.loads(json.dumps(view.transform.to_dict())))
    assert restored.bounds == (50, 0, 150, 100)
    assert not restored.to_panorama([[10, 20]])[1].any()
    empty = panorama().view(Perspective(100, 50)).letterbox(100, 100).crop(0, 0, 100, 10)
    assert not empty.valid.any() and not empty.image.any()
    assert empty.letterbox(200, 200).map_boxes([Box2D(0, 0, 200, 200)]) == []


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int64, bool])
def test_nearest_mode_survives_all_preprocessing(dtype):
    labels = np.zeros((40, 80), dtype=dtype)
    labels[:, 40:] = True if dtype is bool else 4
    view = Panorama(labels).view(Perspective(20, 10), interpolation="nearest")
    view = view.resize(37, 19).crop(0.5, 0.5, 35, 18).letterbox(43, 41)
    assert view.interpolation == "nearest"
    assert set(np.unique(view.image)) <= set(np.unique(labels))
    assert view.image.dtype == labels.dtype
    assert not view.image[~view.valid].any()


def test_interpolation_can_be_overridden_and_arrays_remain_caller_owned():
    pixels = np.ones((100, 200, 3), np.uint8)
    p = Panorama(pixels)
    assert np.shares_memory(p.image, pixels)
    view = p.view(Perspective(12, 10))
    nearest = view.resize(30, 30, interpolation="nearest")
    assert nearest.interpolation == "nearest" and view.interpolation == "bilinear"
    before = p.image.copy()
    list(p.views([Perspective(10, 10), Perspective(10, 10, yaw=90)], cache=None))
    np.testing.assert_array_equal(p.image, before)


def test_box_clips_at_all_edges_and_outside_boxes_are_empty():
    view = panorama().view(Perspective(100, 100))
    region = view.map_boxes([Box2D(-10, -10, 110, 110)])[0]
    np.testing.assert_array_equal(region.box.xyxy, [0, 0, 100, 100])
    assert view.map_boxes([Box2D(-20, -20, -10, -10), Box2D(101, 5, 110, 20)]) == []
    assert not view.transform.from_panorama([[0, 50]])[1].any()


@pytest.mark.parametrize(
    "change",
    [
        {"scale": (0, 1)},
        {"offset": (np.nan, 0)},
        {"output_size": (0, 2)},
        {"content_bounds": (-1, 0, 5, 5)},
        {"content_bounds": (0, 0, 1, np.inf)},
    ],
)
def test_invalid_persisted_transform(change):
    with pytest.raises(ValueError):
        replace(ViewTransform(Equirectangular(20, 10), Perspective(10, 10)), **change)


def test_invalid_panorama_and_preprocessing_contracts():
    with pytest.raises(TypeError):
        Panorama(np.zeros((10, 10)), Perspective(10, 10))
    with pytest.raises(ValueError):
        Panorama(np.zeros((10, 10)), Equirectangular(20, 10))
    with pytest.raises(ValueError):
        Panorama(np.zeros((10, 10)), timestamp=np.nan)
    view = panorama().view(Perspective(10, 10))
    for operation in [
        lambda: view.resize(0, 10),
        lambda: view.letterbox(10, -1),
        lambda: view.crop(np.nan, 0, 1, 1),
        lambda: view.crop(9, 9, 5, 5),
        lambda: ViewTransform.from_dict({"schema_version": 99}),
    ]:
        with pytest.raises(ValueError):
            operation()

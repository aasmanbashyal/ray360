import errno
import importlib.util
import io
import os
import shutil
import stat
from pathlib import Path

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


def test_overwrite_with_open_reader_grows_and_truncates(tmp_path):
    path = tmp_path / "held.png"
    expected = tmp_path / "expected.png"
    small = np.zeros((2, 4, 3), np.uint8)
    large = np.random.default_rng(42).integers(256, size=(96, 192, 3), dtype=np.uint8)
    write_image(path, small)
    with path.open("rb"):
        for pixels in (large, small):
            write_image(expected, pixels, overwrite=True)
            write_image(path, pixels, overwrite=True)
            assert path.read_bytes() == expected.read_bytes()
    assert set(tmp_path.iterdir()) == {path, expected}


@pytest.mark.skipif(os.name != "nt", reason="Windows open-file overwrite")
def test_failed_open_file_overwrite_restores_original_and_can_retry(tmp_path, monkeypatch):
    path = tmp_path / "held.png"
    image = np.zeros((8, 16, 3), np.uint8)
    new_image = np.random.default_rng(7).integers(256, size=(32, 64, 3), dtype=np.uint8)
    write_image(path, image)
    original = path.read_bytes()
    copy = shutil.copyfileobj
    failed = False

    def fail_once(source, output):
        nonlocal failed
        if output.name == str(path) and not failed:
            failed = True
            # Grow the file as well as overwriting its original PNG header.
            output.write(source.read(len(original) + 37))
            raise OSError(errno.ENOSPC, "injected disk full")
        copy(source, output)

    with path.open("rb"):
        monkeypatch.setattr(shutil, "copyfileobj", fail_once)
        with pytest.raises(OSError) as error:
            write_image(path, new_image, overwrite=True)
        assert error.value.errno == errno.ENOSPC
        assert path.read_bytes() == original
        write_image(path, new_image, overwrite=True)
        np.testing.assert_array_equal(read_panorama(path).image, new_image)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only attribute")
def test_read_only_overwrite_preserves_file(tmp_path):
    path = tmp_path / "protected.png"
    path.write_bytes(b"original")
    path.chmod(stat.S_IREAD)
    try:
        with pytest.raises(PermissionError):
            write_image(path, np.zeros((2, 3), np.uint8), overwrite=True)
        assert path.read_bytes() == b"original"
        assert list(tmp_path.iterdir()) == [path]
    finally:
        path.chmod(stat.S_IWRITE)


@pytest.mark.skipif(os.name != "nt", reason="Windows byte-range lock")
def test_locked_writer_preserves_original_until_unlock(tmp_path):
    import msvcrt

    path = tmp_path / "locked.png"
    image = np.zeros((8, 16, 3), np.uint8)
    path.write_bytes(b"original")
    with path.open("r+b", buffering=0) as holder:
        msvcrt.locking(holder.fileno(), msvcrt.LK_NBLCK, 1)
        with pytest.raises(PermissionError):
            write_image(path, image, overwrite=True)
        assert holder.read() == b"original"
        holder.seek(0)
        msvcrt.locking(holder.fileno(), msvcrt.LK_UNLCK, 1)
        write_image(path, image, overwrite=True)
    np.testing.assert_array_equal(read_panorama(path).image, image)


@pytest.fixture
def no_hard_links(monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX exclusive-create fallback")

    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    monkeypatch.setattr(os, "link", unsupported)


def test_no_hard_links_disk_full_allows_retry(no_hard_links, tmp_path, monkeypatch):
    path = tmp_path / "retry.png"
    image = np.zeros((8, 16, 3), np.uint8)

    def disk_full(source, output):
        output.write(source.read(37))
        raise OSError(errno.ENOSPC, "injected disk full")

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfileobj", disk_full)
        with pytest.raises(OSError) as error:
            write_image(path, image)
        assert error.value.errno == errno.ENOSPC
    assert list(tmp_path.iterdir()) == []
    write_image(path, image)
    np.testing.assert_array_equal(read_panorama(path).image, image)


def test_no_hard_links_delayed_close_error_removes_output(no_hard_links, tmp_path, monkeypatch):
    path = tmp_path / "late.png"
    normal_open = Path.open

    class FailedClose(io.FileIO):
        def close(self):
            super().close()
            raise OSError(errno.ENOSPC, "injected close error")

    def open_file(self, mode, *args, **kwargs):
        if self == path and mode == "xb":
            return FailedClose(self, mode)
        return normal_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError) as error:
        write_image(path, np.zeros((8, 16, 3), np.uint8))
    assert error.value.errno == errno.ENOSPC
    assert list(tmp_path.iterdir()) == []


def test_no_hard_links_cleanup_preserves_concurrent_replacement(
    no_hard_links, tmp_path, monkeypatch
):
    path = tmp_path / "shared.png"

    def replace_during_copy(source, output):
        output.write(source.read(37))
        other = tmp_path / "other"
        other.write_bytes(b"another writer")
        other.replace(path)
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(shutil, "copyfileobj", replace_during_copy)
    with pytest.raises(OSError) as error:
        write_image(path, np.zeros((8, 16, 3), np.uint8))
    assert error.value.errno == errno.ENOSPC
    assert path.read_bytes() == b"another writer"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name == "nt", reason="POSIX link errors")
@pytest.mark.parametrize("link_error", [errno.EPERM, errno.ENOSYS, errno.EIO])
def test_link_failure_preserves_competing_file(tmp_path, monkeypatch, link_error):
    path = tmp_path / "shared.png"

    def competing_create(*args):
        path.write_bytes(b"another writer")
        raise OSError(link_error, "injected link error")

    monkeypatch.setattr(os, "link", competing_create)
    with pytest.raises(OSError) as error:
        write_image(path, np.zeros((2, 3), np.uint8))
    assert error.value.errno == (errno.EIO if link_error == errno.EIO else errno.EEXIST)
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

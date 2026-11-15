"""Downloader integrity checks run without touching the network."""

import hashlib
import zipfile

import pytest

from scripts.fetch_test_data import extract, fetch


def test_verify_only_detects_modified_extracted_data_without_repairing_it(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("dataset/labels.txt", "original labels")
    extract(archive, tmp_path, 1024)
    labels = tmp_path / "dataset/labels.txt"
    entry = {
        "id": "local",
        "path": "dataset.zip",
        "url": "https://unused.invalid/",
        "extract": True,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    fetch(entry, tmp_path, lock=False, verify_only=True, limit=1024)
    labels.write_text("changed labels")
    with pytest.raises(ValueError, match="modified unpacked"):
        fetch(entry, tmp_path, lock=False, verify_only=True, limit=1024)
    assert labels.read_text() == "changed labels"


@pytest.mark.parametrize("name", ["../escape.txt", "/escape.txt"])
def test_archive_paths_cannot_escape_the_cache(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(name, "not extracted")
    with pytest.raises(ValueError, match="Unsafe archive"):
        extract(archive, tmp_path, 1024)


def test_archive_expansion_budget_is_enforced(tmp_path):
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        stream.writestr("large.txt", "x" * 10000)
    with pytest.raises(ValueError, match="Expanded archive"):
        extract(archive, tmp_path, 1000)
    assert not (tmp_path / "large.txt").exists()

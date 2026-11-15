"""Optional photo fixtures are explicit and verified before use."""

import hashlib
import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--public-data",
        nargs="?",
        const=".test-data",
        default=None,
        help="Run optional image tests against the downloaded panorama corpus",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "public_data: optional downloaded stitched panoramas")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--public-data") is None:
        for item in items:
            if "public_data" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason="Pass --public-data after fetching the optional photos")
                )


@pytest.fixture(scope="session")
def public_photos(pytestconfig):
    root = Path(pytestconfig.getoption("--public-data"))
    manifest = json.loads((Path(__file__).parent / "data-sources.json").read_text(encoding="utf-8"))
    paths = []
    for entry in manifest["entries"]:
        path = root / entry["path"]
        assert path.is_file(), f"Run scripts/fetch_test_data.py; missing {path}"
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        paths.append(path)
    return paths

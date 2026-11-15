"""Optional photo fixtures are explicit and verified before use."""

import hashlib
import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--e2e", action="store_true", help="Run real cube, decoder and model integration tests"
    )
    parser.addoption(
        "--e2e-output", default=None, help="New directory for end-to-end reports and images"
    )
    parser.addoption(
        "--public-data",
        nargs="?",
        const=".test-data",
        default=None,
        help="Run optional image tests against the downloaded panorama corpus",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "public_data: optional downloaded stitched panoramas")
    config.addinivalue_line(
        "markers", "e2e: optional real media and external model/decoder integration"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--e2e"):
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Pass --e2e with the external application dependencies installed"
                    )
                )
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
        if entry.get("group") == "e2e":
            continue
        path = root / entry["path"]
        assert path.is_file(), f"Run scripts/fetch_test_data.py; missing {path}"
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        paths.append(path)
    return paths

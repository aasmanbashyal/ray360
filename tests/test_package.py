import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import ray360


def test_public_package_contract():
    assert set(ray360.__all__) == {
        "Box2D",
        "Equirectangular",
        "MapCache",
        "Panorama",
        "Perspective",
        "Polygon2D",
        "ProjectionMap",
        "SphericalRegion",
        "View",
        "ViewTransform",
        "coverage_views",
        "cube_projections",
        "from_cubemap",
        "from_cubemap_cross",
        "to_cubemap",
        "to_cubemap_cross",
    }
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["name"] == "ray360"
    assert version("ray360") == ray360.__version__
    assert config["project"]["version"] == ray360.__version__
    assert config["project"]["dependencies"] == ["numpy>=1.24"]
    assert set(config["project"]["optional-dependencies"]) == {"image"}
    assert "scripts" not in config["project"]
    assert Path(ray360.__file__).with_name("py.typed").is_file()


def test_core_import_and_processing_never_import_optional_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.util
import sys
import numpy as np
from ray360 import Panorama, Perspective, from_cubemap_cross
assert importlib.util.find_spec('spherekit') is None
assert Panorama(np.zeros((10, 20, 3), np.uint8)).view(Perspective(8, 8)).image.shape == (8, 8, 3)
image, [polygons] = from_cubemap_cross(np.zeros((24, 32, 3), np.uint8), boxes=[(27, 10, 29, 14)])
assert image.shape == (16, 32, 3) and len(polygons) == 2
assert not any(name in sys.modules for name in ('PIL', 'av', 'torch', 'ultralytics'))
for name in ('media', 'cameras', 'datasets', 'ai', 'torch', 'cli'):
    assert importlib.util.find_spec('ray360.' + name) is None
""",
        ],
        check=True,
    )


def test_image_helper_names_missing_extra_without_importing_pillow():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import numpy as np
from ray360.image import read_panorama
sys.modules['PIL'] = None
try:
    read_panorama('unused.png')
except ImportError as error:
    assert 'ray360[image]' in str(error)
else:
    raise AssertionError('Expected a missing-backend error')
""",
        ],
        check=True,
    )

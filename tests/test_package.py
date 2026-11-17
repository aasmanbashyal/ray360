import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

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
assert 'ray360_native' not in sys.modules
sys.modules['ray360_native'] = None
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


@pytest.mark.parametrize("provider_state", ["legacy", "unloadable", "init_error", "kernel_error"])
def test_optional_sampler_discovery_and_kernel_errors(provider_state):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from importlib.abc import MetaPathFinder
import sys
from types import ModuleType
import numpy as np
from ray360 import remap

state = sys.argv[1]
package = ModuleType('ray360_native')
package.__path__ = []
sys.modules[package.__name__] = package
assert 'ray360_native._sampler' not in sys.modules

def fail(*args, **kwargs):
    raise RuntimeError('sampling kernel failed')

if state == 'legacy':
    package._sample_with_fallback = fail
elif state in ('unloadable', 'init_error'):
    class UnloadableExtension(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'ray360_native._sampler':
                if state == 'init_error':
                    raise OSError('provider initialization failed')
                raise ImportError('the compiled extension failed to load')

    sys.meta_path.insert(0, UnloadableExtension())
else:
    provider = ModuleType('ray360_native._sampler')
    provider.try_sample_v1 = fail
    sys.modules[provider.__name__] = provider

pixels = np.array([[10, 20]], np.uint8)
xy = np.array([[1.0, 0.5]], np.float32)
if state in ('init_error', 'kernel_error'):
    expected = (OSError, 'provider initialization failed') if state == 'init_error' else (
        RuntimeError, 'sampling kernel failed'
    )
    try:
        remap.sample(pixels, xy)
    except expected[0] as error:
        assert str(error) == expected[1]
    else:
        raise AssertionError('The provider error was swallowed')
else:
    np.testing.assert_array_equal(remap.sample(pixels, xy), [15])
""",
            provider_state,
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

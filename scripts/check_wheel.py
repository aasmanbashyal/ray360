"""Install the built wheel in a clean environment and test it outside the checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SMOKE = """
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import ray360
from ray360 import Box2D, Equirectangular, Panorama, Perspective, SphericalRegion, ViewTransform
from ray360.remap import sample
assert importlib.util.find_spec('spherekit') is None
assert importlib.metadata.version('ray360') == ray360.__version__
expected = {'numpy', 'ray360'} | ({'pillow'} if sys.argv[1] == 'image' else set())
installed = {d.metadata['Name'].lower() for d in importlib.metadata.distributions()}
assert installed == expected, installed
assert Path(ray360.__file__).is_relative_to(Path(sys.prefix)), ray360.__file__
assert Path(ray360.__file__).with_name('py.typed').is_file()
pixel = sample(np.array([[10, 20]], np.uint8), np.array([1.0, 0.5], np.float32))
assert isinstance(pixel, np.ndarray) and pixel.shape == () and pixel.item() == 15
assert 'to_cubemap_cross' in ray360.__all__
for name in ('ai', 'cameras', 'cli', 'datasets', 'frame', 'media', 'torch', 'visualize'):
    assert importlib.util.find_spec('ray360.' + name) is None, name
panorama = Panorama(np.zeros((90, 180, 3), np.uint8))
faces = ray360.to_cubemap(panorama, 32)
np.testing.assert_array_equal(ray360.from_cubemap(faces, panorama.projection), panorama.image)
cross = ray360.to_cubemap_cross(panorama, 32)
assert cross.shape == (96, 128, 3) and cross.dtype == np.uint8
np.testing.assert_array_equal(cross[32:64, 32:64], faces['front'])
rebuilt, [seam] = ray360.from_cubemap_cross(cross, panorama.projection, boxes=[(106, 40, 118, 56)])
np.testing.assert_array_equal(rebuilt, panorama.image)
assert len(seam) == 2
view = panorama.view(Perspective(100, 50, yaw=180)).letterbox(100, 100)
region = view.map_boxes([Box2D(25, 30, 75, 70, label='object')])[0]
assert len(region.polygons(panorama.projection)) == 2
assert view.map_boxes([Box2D(1, 1, 10, 10)]) == []
chain = ViewTransform.from_dict(json.loads(json.dumps(view.transform.to_dict())))
xy, valid = chain.to_panorama([[50, 50]])
assert valid.all()
np.testing.assert_allclose(chain.from_panorama(xy)[0], [[50, 50]], atol=1e-8)
restored = SphericalRegion.from_dict(json.loads(json.dumps(region.to_dict())))
np.testing.assert_array_equal(restored.rasterize(panorama.projection), region.rasterize(panorama.projection))
if sys.argv[1] == 'image':
    from ray360.image import read_panorama, write_image
    write_image('capture.v1.png', panorama.image)
    np.testing.assert_array_equal(read_panorama('capture.v1.png').image, panorama.image)
    write_image('cubemap-cross.png', cross)
    np.testing.assert_array_equal(read_panorama('cubemap-cross.png').image, cross)
    annotated, _ = ray360.from_cubemap_cross(cross, boxes=[(106, 40, 118, 56)], draw_boxes=True)
    assert annotated.any()
else:
    from ray360.image import read_panorama
    try:
        read_panorama('unused.png')
    except ImportError as error:
        assert 'ray360[image]' in str(error)
    else:
        raise AssertionError('Pillow unexpectedly available')
print('Installed wheel passed:', ray360.__version__, sorted(installed))
"""

NATIVE_SMOKE = """
import platform
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import ray360
assert 'ray360_native' not in sys.modules
import ray360_native
from ray360 import remap
from ray360_native import _native

for module in (ray360, ray360_native, _native):
    assert Path(module.__file__).is_relative_to(Path(sys.prefix)), module.__file__
pixels = np.array([[10, 20]], dtype=np.uint8)
xy = np.array([[1.0, 0.5]], dtype=np.float32)
with patch.object(_native, 'sample_u8_f32', wraps=_native.sample_u8_f32) as kernel:
    np.testing.assert_array_equal(remap.sample(pixels, xy), [15])
    kernel.assert_called_once()
    np.testing.assert_array_equal(remap.sample(pixels.astype(np.uint16), xy), [15])
    kernel.assert_called_once()
assert 'ray360_native._prepared' not in sys.modules
print('Installed Rust wheel:', Path(_native.__file__).name)
print('Platform:', platform.system(), platform.machine(), 'NumPy', np.__version__)
print('AVX2 available:', _native.avx2_available())
"""

NATIVE_ONLY_SMOKE = """
import importlib.metadata
import sys
import numpy as np
import ray360_native

installed = {d.metadata['Name'].lower() for d in importlib.metadata.distributions()}
assert installed == {'numpy', 'ray360-native'}, installed
pixels = np.array([[10, 20]], dtype=np.uint8)
xy = np.array([[1.0, 0.5]], dtype=np.float32)
np.testing.assert_array_equal(ray360_native.sample(pixels, xy), [15])
assert 'ray360' not in sys.modules
assert 'ray360_native._prepared' not in sys.modules
for use_ray360 in (
    lambda: ray360_native.PreparedView,
    lambda: ray360_native.sample(pixels.astype(np.uint16), xy),
):
    try:
        use_ray360()
    except ImportError as error:
        assert 'Install ray360' in str(error), error
    else:
        raise AssertionError('Expected an error naming the optional Ray360 dependency')
"""


def make_venv(environment: Path) -> Path:
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install_wheels(python: Path, constraints: Path) -> list[str]:
    return [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--only-binary",
        ":all:",
        "--constraint",
        str(constraints),
    ]


def check_sdist(archive: Path, wheel: Path, work: Path) -> None:
    print(f"{archive.name}: {archive.stat().st_size:,} bytes", flush=True)
    with tarfile.open(archive, "r:gz") as source:
        if any("/docs/assets/" in name and name.endswith(".webp") for name in source.getnames()):
            raise SystemExit("The source archive includes README gallery images.")

    rebuilt_dir = work / "rebuilt"
    subprocess.run(
        ["uv", "build", "--wheel", "--no-sources", "--out-dir", str(rebuilt_dir), str(archive)],
        cwd=work,
        check=True,
    )
    if (rebuilt_dir / wheel.name).read_bytes() != wheel.read_bytes():
        raise SystemExit("The wheel rebuilt from the source archive differs from the original.")
    print("Source archive rebuilt a byte-identical core wheel.", flush=True)


def check_native(work: Path, core_wheel: Path, native_wheel: Path, constraints: Path) -> None:
    print(f"{native_wheel.name}: {native_wheel.stat().st_size:,} bytes", flush=True)
    python = make_venv(work / "native-environment")
    install = install_wheels(python, constraints)
    subprocess.run([*install, str(native_wheel)], check=True)
    subprocess.run([str(python), "-I", "-c", NATIVE_ONLY_SMOKE], cwd=work, check=True)
    subprocess.run(
        [
            *install,
            "--no-index",
            "--find-links",
            str(core_wheel.parent),
            f"ray360-native[ray360] @ {native_wheel.as_uri()}",
        ],
        check=True,
    )
    subprocess.run([*install, "pytest==" + importlib.metadata.version("pytest")], check=True)
    subprocess.run([str(python), "-I", "-c", NATIVE_SMOKE], cwd=work, check=True)
    subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "pytest",
            "-c",
            str(ROOT / "native" / "pyproject.toml"),
            str(ROOT / "native" / "tests"),
            "-q",
        ],
        cwd=work,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", action="store_true", help="Also exercise the optional image extra"
    )
    parser.add_argument("--native", type=Path, metavar="WHEEL", help="Check a built Rust wheel")
    parser.add_argument(
        "--native-max-bytes", type=int, metavar="BYTES", help="Limit a native wheel's size"
    )
    parser.add_argument(
        "--sdist", action="store_true", help="Check the core source archive and rebuild its wheel"
    )
    args = parser.parse_args()
    if args.native and (args.native.suffix != ".whl" or not args.native.is_file()):
        parser.error("--native requires an existing .whl file")
    if args.native_max_bytes is not None:
        if args.native is None or args.native_max_bytes <= 0:
            parser.error("--native-max-bytes requires --native and a positive limit")
        size = args.native.stat().st_size
        if size > args.native_max_bytes:
            parser.error(
                f"Native wheel is {size:,} bytes, above the {args.native_max_bytes:,}-byte limit"
            )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = config["project"]["version"]
    wheel = ROOT / "dist" / f"ray360-{version}-py3-none-any.whl"
    if not wheel.is_file():
        parser.error("Run uv build --no-sources first")
    archive = ROOT / "dist" / f"ray360-{version}.tar.gz"
    if args.sdist and not archive.is_file():
        parser.error("Run uv build --no-sources to create the source archive")
    print(f"{wheel.name}: {wheel.stat().st_size:,} bytes", flush=True)
    with tempfile.TemporaryDirectory(prefix="ray360-wheel-") as directory:
        work = Path(directory)
        python = make_venv(work / "environment")
        versions = ["numpy==" + importlib.metadata.version("numpy")]
        if args.image:
            versions.append("Pillow==" + importlib.metadata.version("Pillow"))
        constraints = work / "constraints.txt"
        constraints.write_text("\n".join(versions) + "\n", encoding="utf-8")
        project = "ray360[image]" if args.image else "ray360"
        requirement = f"{project} @ {wheel.resolve().as_uri()}"
        subprocess.run([*install_wheels(python, constraints), requirement], check=True)
        subprocess.run(
            [str(python), "-I", "-c", SMOKE, "image" if args.image else "core"],
            cwd=work,
            check=True,
        )
        if args.sdist:
            check_sdist(archive, wheel, work)
        if args.native:
            check_native(work, wheel, args.native.resolve(), constraints)


if __name__ == "__main__":
    main()

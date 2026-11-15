"""Opt-in actual application runs. No downloads, mocked models or synthetic videos."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from examples.detect import detect_panorama
from examples.video_frames import iter_panoramas
from ray360 import (
    Equirectangular,
    MapCache,
    Panorama,
    SphericalRegion,
    ViewTransform,
    coverage_views,
    cube_projections,
    from_cubemap,
    to_cubemap,
)
from ray360.image import overlay, read_panorama, write_image

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def assets(pytestconfig):
    root = Path(pytestconfig.getoption("--public-data") or ROOT / ".test-data")
    manifest = json.loads((ROOT / "tests/data-sources.json").read_text(encoding="utf-8"))
    result = {}
    for entry in manifest["entries"]:
        path = root / entry["path"]
        assert path.is_file(), f"Run scripts/fetch_test_data.py --include-e2e; missing {path}"
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"], path
        result[entry["id"]] = (path, entry)
    return result


@pytest.fixture(scope="module")
def report(pytestconfig, tmp_path_factory, assets):
    chosen = pytestconfig.getoption("--e2e-output")
    output = Path(chosen).resolve() if chosen else tmp_path_factory.mktemp("ray360-e2e")
    if chosen:
        output.mkdir(parents=True, exist_ok=False)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("ray360", "numpy", "Pillow", "av", "ultralytics", "torch", "torchvision")
    }
    data = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "device": "cpu",
        "workflows": {},
        "fixtures": {name: entry["sha256"] for name, (_, entry) in assets.items()},
    }
    yield output, data
    (output / "report.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"End-to-end artifacts: {output}", flush=True)


@pytest.fixture(scope="module")
def detector(assets, report):
    os.environ["YOLO_CONFIG_DIR"] = str(report[0] / "yolo-settings")
    os.environ["YOLO_OFFLINE"] = "true"
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    return YOLO(str(assets["yolo11n"][0]))


def check_predictions(records):
    for record in records:
        region = SphericalRegion.from_dict(record["region"])
        transform = ViewTransform.from_dict(record["provenance"]["transform"])
        center = (region.box.xyxy[:2] + region.box.xyxy[2:]) / 2
        output_point = center * transform.scale + transform.offset
        xy, valid = transform.to_panorama(output_point)
        recovered, visible = transform.from_panorama(xy)
        assert valid and visible
        np.testing.assert_allclose(recovered, output_point, atol=1e-7)
        rays, _ = transform.source.unproject(xy)
        assert region.contains(rays)
        assert region.box.label == record["box"]["label"]
        assert region.box.score == record["box"]["score"]


def psnr(a, b):
    mse = float(np.mean((a.astype(float) - b.astype(float)) ** 2))
    return float(10 * np.log10(255**2 / max(mse, 1e-12)))


@pytest.mark.parametrize("asset_id", ["pannellum-panorama", "ricoh-theta-panorama"])
def test_photo_conversion_and_actual_model(asset_id, assets, report, detector):
    from PIL import Image

    output, data = report
    panorama = read_panorama(assets[asset_id][0])
    start = time.perf_counter()
    faces = to_cubemap(panorama, 512)
    restored = from_cubemap(faces, Equirectangular(1024, 512))
    reference = np.array(
        Image.fromarray(panorama.image).resize((1024, 512), Image.Resampling.BILINEAR)
    )
    quality = psnr(reference, restored)
    assert quality > 25, quality
    write_image(output / f"{asset_id}-roundtrip.png", restored)
    for name, face in faces.items():
        write_image(output / f"{asset_id}-{name}.png", face)
    # Non-square inputs exercise actual letterbox prediction coordinates.
    targets = coverage_views(width=320, height=240)
    regions, records = detect_panorama(panorama, detector, targets)
    check_predictions(records)
    if asset_id == "ricoh-theta-panorama":
        assert records
    write_image(
        output / f"{asset_id}-detections.png",
        overlay(panorama.image, regions, projection=panorama.projection),
    )
    (output / f"{asset_id}-predictions.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    data["workflows"][asset_id] = {
        "source_size": list(panorama.projection.size),
        "cube_size": 512,
        "roundtrip_size": [1024, 512],
        "roundtrip_psnr_db": quality,
        "view_count": len(targets),
        "detections": len(records),
        "labels": sorted({r.box.label for r in regions}),
        "elapsed_seconds": time.perf_counter() - start,
    }


def test_photographic_cube_faces(assets, report):
    from PIL import Image

    output, data = report
    faces = {
        name: np.array(Image.open(assets[f"pannellum-cube-{name}"][0]).convert("RGB"))
        for name in cube_projections(1)
    }
    panorama = Panorama(from_cubemap(faces, Equirectangular(2048, 1024)))
    write_image(output / "photographic-cube-panorama.png", panorama.image)
    # Compare independent upstream representations, so a round-trip cannot hide flips.
    with Image.open(assets["pannellum-panorama"][0]) as original:
        reference_panorama = np.array(
            original.convert("RGB").resize((2048, 1024), Image.Resampling.BILINEAR)
        )
    cross_format_quality = psnr(reference_panorama, panorama.image)
    assert cross_format_quality > 25, cross_format_quality
    recovered = to_cubemap(panorama, 256)
    quality = {}
    for name, face in faces.items():
        reference = np.array(Image.fromarray(face).resize((256, 256), Image.Resampling.BILINEAR))
        quality[name] = psnr(reference, recovered[name])
        assert quality[name] > 24, (name, quality[name])
    data["workflows"]["photographic-cube"] = {
        "input_face_size": faces["front"].shape[0],
        "panorama_size": list(panorama.projection.size),
        "recovered_face_size": 256,
        "face_psnr_db": quality,
        "independent_panorama_psnr_db": cross_format_quality,
    }


@pytest.mark.parametrize("asset_id", ["threejs-pano-mp4", "threejs-pano-webm"])
def test_complete_video_decode_cube_and_model(asset_id, assets, report, detector):
    output, data = report
    path, metadata = assets[asset_id]
    targets = list(cube_projections(320).values())
    cache = MapCache(32 * 1024 * 1024)
    frames, detections, start = [], 0, time.perf_counter()
    for index, panorama in enumerate(iter_panoramas(path)):
        faces = to_cubemap(panorama, 64, cache=cache)
        restored = from_cubemap(faces, Equirectangular(256, 128))
        assert restored.shape == (128, 256, 3) and restored.dtype == np.uint8
        regions, records = detect_panorama(panorama, detector, targets, cache=cache)
        check_predictions(records)
        assert all(record["provenance"]["timestamp"] == panorama.timestamp for record in records)
        detections += len(records)
        frames.append({"index": index, "timestamp": panorama.timestamp, "detections": len(records)})
        if index in (0, metadata["expected_frames"] - 1):
            write_image(output / f"{asset_id}-{index}-roundtrip.png", restored)
            write_image(
                output / f"{asset_id}-{index}-detections.png",
                overlay(panorama.image, regions, projection=panorama.projection),
            )
            (output / f"{asset_id}-{index}-predictions.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
        if index % 25 == 0:
            print(f"{asset_id}: decoded, converted and inferred frame {index}", flush=True)
    assert len(frames) == metadata["expected_frames"]
    times = [frame["timestamp"] for frame in frames]
    assert times[0] == pytest.approx(metadata["first_timestamp"])
    assert times[-1] == pytest.approx(metadata["last_timestamp"])
    assert np.all(np.diff(times) > 0)
    assert detections > 0 and cache.nbytes <= cache.max_bytes
    data["workflows"][asset_id] = {
        "frames": frames,
        "model_view_count": len(targets),
        "model_input_size": 320,
        "total_model_views": len(frames) * len(targets),
        "detections": detections,
        "cache_bytes": cache.nbytes,
        "elapsed_seconds": time.perf_counter() - start,
    }


def test_example_command_lines(assets, report):
    output, data = report
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    detected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/detect.py"),
            str(assets["ricoh-theta-panorama"][0]),
            str(assets["yolo11n"][0]),
            str(output / "cli-detections.png"),
            "--view-size",
            "320",
            "--view-height",
            "240",
            "--json",
            str(output / "cli-detections.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    predictions = json.loads((output / "cli-detections.json").read_text())
    assert predictions["detection_count"] > 0
    check_predictions(predictions["predictions"])
    for asset_id in ("threejs-pano-mp4", "threejs-pano-webm"):
        path, entry = assets[asset_id]
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples/video_frames.py"),
                str(path),
                "--json",
                str(output / f"cli-{asset_id}.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        frames = json.loads((output / f"cli-{asset_id}.json").read_text())
        assert len(frames) == entry["expected_frames"]
        assert frames[-1]["timestamp"] == pytest.approx(entry["last_timestamp"])
        assert all(frame["shape"] == [180, 320, 3] for frame in frames)
    data["workflows"]["example-clis"] = {
        "image_detections": predictions["detection_count"],
        "decoder_frames": [188, 189],
        "detector_stdout": detected.stdout.strip(),
    }

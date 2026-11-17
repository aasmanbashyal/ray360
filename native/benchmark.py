"""Compare Ray360's Rust and NumPy sampling with optional saved Rust and OpenCV baselines."""

import argparse
import importlib.util
import json
import platform
import sys
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ray360 import Box2D, MapCache, Panorama, Perspective, coverage_views, remap
from ray360_native._sampler import try_sample_v1

NUMPY_SAMPLE = remap._sample_numpy


def load_baseline(path, name="ray360_native_baseline"):
    """Load a saved package and its native binary under a distinct package name."""
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(operations, iterations, *, include_samples=False, balanced=False):
    """Interleave backends and alternate order to reduce clock/load drift."""
    for operation in operations.values():
        operation()
    times = {name: [] for name in operations}
    entries = list(operations.items())
    for index in range(iterations):
        if balanced:
            shift = index % len(entries)
            order = entries[shift:] + entries[:shift]
            if index // len(entries) % 2:
                order.reverse()
        else:
            order = entries[:: 1 if index % 2 == 0 else -1]
        for name, operation in order:
            start = time.perf_counter()
            operation()
            times[name].append((time.perf_counter() - start) * 1000)
    return {
        name: {
            "p50_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            **({"samples_ms": values} if include_samples else {}),
        }
        for name, values in times.items()
    }


def with_sampler(sampler, operation):
    # A benchmark-only override in this single-threaded process.
    provider = None if sampler is NUMPY_SAMPLE else sampler
    with patch.object(remap, "_native_sampler", lambda: provider):
        return operation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="Optional real panorama; otherwise seeded noise")
    parser.add_argument(
        "--baseline-package", type=Path, help="Optional saved ray360_native package"
    )
    parser.add_argument("--opencv", action="store_true", help="Also compare an installed OpenCV")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if args.image:
        from ray360.image import read_panorama

        pano = read_panorama(args.image)
    else:
        pano = Panorama(np.random.default_rng(42).integers(0, 256, (2048, 4096, 3), np.uint8))
    if remap._native_sampler() is not try_sample_v1:
        parser.error("Install the current ray360_native package before running this benchmark")
    backends = {"numpy": NUMPY_SAMPLE, "rust": None}
    if args.baseline_package:
        backends["rust_previous"] = load_baseline(args.baseline_package).sample
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "source_shape": list(pano.image.shape),
        "input": str(args.image) if args.image else "seeded uniform uint8 noise, seed 42",
        "iterations": args.iterations,
        "schedule": "Backends rotate through all measurement positions in both directions",
        "baseline_package": str(args.baseline_package) if args.baseline_package else None,
        "scope": "Warm geometry only. Map setup, decoding, inference, I/O and compilation excluded. "
        "Rust holds the GIL and uses one thread. No target-cpu=native or fast-math. "
        "Both pipelines use the same preprocessing, with no padding/crop shortcuts.",
        "timings": {},
        "accuracy": {},
    }
    cache = MapCache()
    cv2 = None
    if args.opencv:
        import cv2

        cv2.setNumThreads(1)
        report["opencv"] = cv2.__version__
        report["opencv_threads"] = cv2.getNumThreads()
    for size in (320, 640):
        projection = Perspective(size, size, yaw=179, pitch=20)
        mapping = cache.get(pano.projection, projection)
        for interpolation in ("bilinear", "nearest"):
            outputs = {}
            operations = {}
            operation = partial(mapping.apply, pano.image, interpolation=interpolation)
            for label, sampler in backends.items():
                run = operation if sampler is None else partial(with_sampler, sampler, operation)
                outputs[label] = run()
                operations[f"{label}_{size}_{interpolation}"] = run
            if cv2 is not None and interpolation == "bilinear":
                coords = mapping.xy.copy() - np.float32(0.5)
                coords[..., 1] = np.clip(coords[..., 1], 0, pano.projection.height - 1)
                operations[f"opencv_{size}_bilinear"] = lambda coords=coords: cv2.remap(
                    pano.image, coords, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
                )
            timings = measure(operations, args.iterations, balanced=True)
            report["timings"].update(timings)
            print(json.dumps(timings), flush=True)
            difference = np.abs(outputs["numpy"].astype(float) - outputs["rust"].astype(float))
            report["accuracy"][f"{size}_{interpolation}"] = {
                "max_abs": float(difference.max()),
                "mae": float(difference.mean()),
            }
            for output in outputs.values():
                np.testing.assert_array_equal(output, outputs["numpy"])

    targets = coverage_views(320, 320)
    for target in targets:
        cache.get(pano.projection, target)
    box = Box2D(80, 140, 240, 260)

    def workflow():
        for view in pano.views(targets, cache=cache):
            view = view.letterbox(320, 400)
            view.transform.to_panorama([[160, 200]])
            for region in view.map_boxes([box]):
                region.polygons(pano.projection)

    timings = measure(
        {
            f"{label}_24_view_workflow": workflow
            if sampler is None
            else partial(with_sampler, sampler, workflow)
            for label, sampler in backends.items()
        },
        args.iterations,
        balanced=True,
    )
    report["timings"].update(timings)
    print(json.dumps(timings), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

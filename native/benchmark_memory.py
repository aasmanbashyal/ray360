"""Compare compact prepared plans with OpenCV and an optional saved Rust package."""

import argparse
import hashlib
import json
import platform
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from ray360 import Box2D, Panorama, Perspective, coverage_views
from ray360.image import read_panorama
from ray360_native import PreparedView, _native

from benchmark import NUMPY_SAMPLE, load_baseline
from benchmark import measure as _measure
from benchmark_fused import CvPrepared, delta, transform_for


def measure(operations, iterations):
    # Retain every round so timing variability can be inspected independently.
    return _measure(operations, iterations, include_samples=True, balanced=True)


def compact_summary(value):
    """Keep console output short; complete per-round timings are saved in JSON."""
    if isinstance(value, dict):
        return {k: compact_summary(v) for k, v in value.items() if k != "samples_ms"}
    return value


def binary_info(plan_class):
    binary = Path(sys.modules[plan_class.__module__]._native.__file__)
    return {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--baseline-package", type=Path)
    parser.add_argument("--extra-baseline-package", type=Path, help="Optional second saved variant")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--workflow-iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iterations, args.workflow_iterations) < 1:
        parser.error("iterations must be positive")
    baseline = load_baseline(args.baseline_package).PreparedView if args.baseline_package else None
    extra = (
        load_baseline(args.extra_baseline_package, "ray360_native_extra_baseline").PreparedView
        if args.extra_baseline_package
        else None
    )
    pano = (
        read_panorama(args.image)
        if args.image
        else Panorama(np.random.default_rng(42).integers(0, 256, (2048, 4096, 3), np.uint8))
    )
    cv2.setNumThreads(1)

    def plans_for(transform):
        # Use a separate geometry holder: inspecting xy/valid intentionally caches
        # diagnostic arrays in PreparedView. Compact production plans never need them.
        geometry = PreparedView(transform)
        plans = {"rust_compact": PreparedView(transform)}
        if baseline:
            plans["rust_previous"] = baseline(transform)
        if extra:
            plans["rust_extra_baseline"] = extra(transform)
        plans["opencv_float_roi"] = CvPrepared(geometry)
        plans["opencv_fixed_roi"] = CvPrepared(geometry, fixed=True)
        return plans, geometry

    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "avx2_available": _native.avx2_available(),
        "source_shape": list(pano.image.shape),
        "image": str(args.image) if args.image else "seeded noise 42",
        "baseline_package": str(args.baseline_package) if baseline else None,
        "extra_baseline_package": str(args.extra_baseline_package) if extra else None,
        "binaries": {
            "rust_compact": binary_info(PreparedView),
            **({"rust_previous": binary_info(baseline)} if baseline else {}),
            **({"rust_extra_baseline": binary_info(extra)} if extra else {}),
        },
        "iterations": args.iterations,
        "workflow_iterations": args.workflow_iterations,
        "scope": "One CPU thread; warmed plans; balanced rotating/reversing backend order each round. "
        "Includes output allocation; excludes setup, decoding, inference, and I/O. "
        "Memory is owned retained array storage, excluding object/allocator headers, "
        "input/output frames and transient setup buffers. This is not peak process RSS. "
        "Compact plans do not retain diagnostic coordinate maps unless accessed. "
        "All backends use identical fused float32 geometry; OpenCV processes only the valid ROI.",
        "cases": {},
    }
    for size in (320, 640):
        target = Perspective(size * 2, size * 3 // 2, yaw=179, pitch=20, roll=7)
        transform = transform_for(pano.projection, target, size)
        plans, geometry = plans_for(transform)
        reference = NUMPY_SAMPLE(pano.image, geometry.xy, wrap=True, valid=geometry.valid)
        ops = {name: partial(plan.apply, pano) for name, plan in plans.items()}
        accuracy = {name: delta(op().image, reference) for name, op in ops.items()}
        for name in ops:
            if name.startswith("rust_"):
                np.testing.assert_array_equal(ops[name]().image, reference)
                np.testing.assert_array_equal(
                    plans[name].apply(pano, engine="scalar").image, reference
                )
        assert "_maps" not in plans["rust_compact"].__dict__
        setup_ops = {"rust_compact_geometry_and_plan": partial(PreparedView, transform)}
        if baseline:
            setup_ops["rust_previous_geometry_and_plan"] = partial(baseline, transform)
        setup_ops["opencv_float_from_shared_geometry"] = partial(CvPrepared, geometry)
        inspected = PreparedView(transform)
        diagnostic_bytes = inspected.xy.nbytes + inspected.valid.nbytes
        case = {
            "timings": measure(ops, args.iterations),
            "retained_plan_bytes": {name: plan.nbytes for name, plan in plans.items()},
            "block_storage": {
                "packed_blocks": plans["rust_compact"]._plan.packed_blocks,
                "wide_blocks": plans["rust_compact"]._plan.wide_blocks,
            },
            "compact_with_diagnostic_maps_bytes": inspected.nbytes,
            "optional_diagnostic_map_bytes": diagnostic_bytes,
            "accuracy_vs_single_pass_numpy": accuracy,
            "scalar_timings": measure(
                {
                    name: partial(plan.apply, pano, engine="scalar")
                    for name, plan in plans.items()
                    if name.startswith("rust_")
                },
                args.iterations,
            ),
            "setup_timings": measure(setup_ops, max(5, args.iterations // 4)),
        }
        report["cases"][str(size)] = case
        print(size, json.dumps(compact_summary(case)), flush=True)

    targets = [Perspective(640, 480, yaw=t.yaw, pitch=t.pitch) for t in coverage_views(320, 320)]
    groups = {}
    accuracy = {}
    for target in targets:
        plans, geometry = plans_for(transform_for(pano.projection, target, 320))
        reference = NUMPY_SAMPLE(pano.image, geometry.xy, wrap=True, valid=geometry.valid)
        for name, plan in plans.items():
            error = delta(plan.apply(pano).image, reference)
            if name.startswith("rust_"):
                np.testing.assert_array_equal(plan.apply(pano).image, reference)
            previous = accuracy.setdefault(name, {"max_abs": 0, "different_values": 0})
            previous["max_abs"] = max(previous["max_abs"], error["max_abs"])
            previous["different_values"] += error["different_values"]
            groups.setdefault(name, []).append(plan)
    box = Box2D(80, 100, 240, 220, label="example", score=0.9)

    def workflow(plans, *, geometry):
        for plan in plans:
            view = plan.apply(pano)
            if geometry:
                view.transform.to_panorama([[160, 160]])
                for region in view.map_boxes([box]):
                    region.polygons(pano.projection)

    report["workflow"] = {
        "view_count": len(targets),
        "sampling_only_timings": measure(
            {name: partial(workflow, plans, geometry=False) for name, plans in groups.items()},
            args.workflow_iterations,
        ),
        "complete_timings": measure(
            {name: partial(workflow, plans, geometry=True) for name, plans in groups.items()},
            args.workflow_iterations,
        ),
        "retained_plan_bytes": {
            name: sum(plan.nbytes for plan in plans) for name, plans in groups.items()
        },
        "block_storage": {
            "packed_blocks": sum(p._plan.packed_blocks for p in groups["rust_compact"]),
            "wide_blocks": sum(p._plan.wide_blocks for p in groups["rust_compact"]),
        },
        "accuracy_vs_single_pass_numpy": accuracy,
        "note": "Complete timings include one point/box mapping and polygon rendering per view.",
    }
    print("workflow", json.dumps(compact_summary(report["workflow"])), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

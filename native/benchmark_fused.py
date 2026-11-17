"""Interleaved CPU comparisons of fused Rust scalar/AVX2 and OpenCV remapping."""

import argparse
import json
import platform
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from ray360 import Box2D, Panorama, Perspective, View, ViewTransform, coverage_views
from ray360.image import read_panorama
from ray360_native import PreparedView, _native
from ray360_native import sample as rust_sample

from benchmark import NUMPY_SAMPLE, measure, with_sampler


class CvPrepared:
    """Cache maps and remap only the valid rectangle, writing into a zeroed output."""

    def __init__(self, prepared, *, fixed=False):
        self.transform = prepared.transform
        self.output_shape = prepared.valid.shape
        ys, xs = np.nonzero(prepared.valid)
        self.roi = None
        self.map1 = self.map2 = None
        if len(xs):
            y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            rectangle = np.zeros(prepared.valid.shape, bool)
            rectangle[y1:y2, x1:x2] = True
            np.testing.assert_array_equal(rectangle, prepared.valid)
            self.roi = (slice(y1, y2), slice(x1, x2))
            coords = prepared.xy[self.roi].copy() - np.float32(0.5)
            coords[..., 1] = np.clip(coords[..., 1], 0, prepared.transform.source.height - 1)
            if fixed:
                self.map1, self.map2 = cv2.convertMaps(coords, None, cv2.CV_16SC2)
            else:
                self.map1 = coords

    @property
    def nbytes(self):
        return sum(m.nbytes for m in (self.map1, self.map2) if m is not None)

    def apply(self, pano):
        output = np.zeros((*self.output_shape, *pano.image.shape[2:]), pano.image.dtype)
        if self.roi is not None:
            result = cv2.remap(
                pano.image,
                self.map1,
                self.map2,
                cv2.INTER_LINEAR,
                dst=output[self.roi],
                borderMode=cv2.BORDER_WRAP,
            )
            assert np.shares_memory(result, output), (
                "OpenCV unexpectedly allocated a separate output"
            )
        return View(output, self.transform, "bilinear", pano.source_id, pano.timestamp)


def delta(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs": float(d.max()),
        "mae": float(d.mean()),
        "different_values": int(np.count_nonzero(d)),
    }


def transform_for(source, target, size):
    return (
        ViewTransform(source, target)
        .crop(8.25, 6.5, target.width - 20, target.height - 16)
        .resize(size, size * 3 // 4)
        .letterbox(size, size)
    )


def eager(pano, target, size):
    return (
        pano.view(target)
        .crop(8.25, 6.5, target.width - 20, target.height - 16)
        .resize(size, size * 3 // 4)
        .letterbox(size, size)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--workflow-iterations", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iterations, args.workflow_iterations) < 1:
        parser.error("iterations must be positive")
    pano = (
        read_panorama(args.image)
        if args.image
        else Panorama(np.random.default_rng(42).integers(0, 256, (2048, 4096, 3), np.uint8))
    )
    cv2.setNumThreads(1)
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "avx2_available": _native.avx2_available(),
        "source_shape": list(pano.image.shape),
        "image": str(args.image) if args.image else "seeded noise 42",
        "iterations": args.iterations,
        "workflow_iterations": args.workflow_iterations,
        "scope": "One CPU thread. Alternating backend order. All fused samplers use the same float32 coordinates. "
        "OpenCV skips the padding ROI; Rust skips wholly invalid SIMD blocks. Output allocation included. "
        "Warm timings exclude plan setup, decoding, inference, file I/O. Eager chains intentionally resample repeatedly.",
        "cases": {},
        "workflow": {},
    }
    for size in (320, 640):
        target = Perspective(size * 2, size * 3 // 2, yaw=179, pitch=20, roll=7)
        transform = transform_for(pano.projection, target, size)
        prepared = PreparedView(transform)
        cvfloat = CvPrepared(prepared)
        cvfixed = CvPrepared(prepared, fixed=True)
        y, x = np.mgrid[:size, :size]
        points = np.stack((x + 0.5, y + 0.5), -1)
        independent_xy, valid = transform.to_panorama(points)
        independent = NUMPY_SAMPLE(
            pano.image, independent_xy.astype(np.float32), wrap=True, valid=valid
        )
        reference = NUMPY_SAMPLE(pano.image, prepared.xy, wrap=True, valid=prepared.valid)
        ops = {
            "numpy_fused": lambda p=prepared: NUMPY_SAMPLE(
                pano.image, p.xy, wrap=True, valid=p.valid
            ),
            "rust_unprepared_fused": lambda p=prepared: rust_sample(
                pano.image, p.xy, wrap=True, valid=p.valid
            ),
            "rust_prepared_scalar": lambda p=prepared: p.apply(pano, engine="scalar").image,
            "rust_prepared_auto": lambda p=prepared: p.apply(pano).image,
            "opencv_float_roi": lambda p=cvfloat: p.apply(pano).image,
            "opencv_fixed_roi": lambda p=cvfixed: p.apply(pano).image,
        }
        if _native.avx2_available():
            ops["rust_prepared_avx2"] = lambda p=prepared: p.apply(pano, engine="avx2").image
        accuracy = {name: delta(op(), reference) for name, op in ops.items()}
        for name in ("rust_unprepared_fused", "rust_prepared_scalar", "rust_prepared_auto"):
            np.testing.assert_array_equal(ops[name](), reference)
        np.testing.assert_array_equal(valid, prepared.valid)
        dx = (
            prepared.xy[..., 0] - independent_xy[..., 0] + pano.projection.width / 2
        ) % pano.projection.width - pano.projection.width / 2
        eager_output = eager(pano, target, size).image
        case = {
            "timings": measure(ops, args.iterations),
            "accuracy_vs_single_pass_numpy": accuracy,
            "independent_geometry_pixel_error": delta(reference, independent),
            "max_wrapped_x_coordinate_error": float(np.abs(dx[valid]).max()),
            "max_y_coordinate_error": float(
                np.abs(prepared.xy[..., 1][valid] - independent_xy[..., 1][valid]).max()
            ),
            "semantic_difference_from_eager_chain": delta(reference, eager_output),
            "map_bytes": {
                "rust_including_coordinates": prepared.nbytes,
                "opencv_float": cvfloat.nbytes,
                "opencv_fixed": cvfixed.nbytes,
            },
            "setup": measure(
                {
                    "rust_geometry_and_plan": lambda t=transform: PreparedView(t),
                    "opencv_float_from_shared_geometry": lambda p=prepared: CvPrepared(p),
                    "opencv_fixed_from_shared_geometry": lambda p=prepared: CvPrepared(
                        p, fixed=True
                    ),
                },
                5,
            ),
        }
        report["cases"][str(size)] = case
        print(size, json.dumps(case), flush=True)

    # Use the known 24 square-view directions with a common rectangular camera.
    targets = [Perspective(640, 480, yaw=t.yaw, pitch=t.pitch) for t in coverage_views(320, 320)]
    plans = [PreparedView(transform_for(pano.projection, t, 320)) for t in targets]
    cvplans = [CvPrepared(p) for p in plans]
    box = Box2D(80, 100, 240, 220, label="example", score=0.9)

    def finish(view):
        view.transform.to_panorama([[160, 160]])
        for region in view.map_boxes([box]):
            region.polygons(pano.projection)

    def workflow_native(engine):
        for plan in plans:
            finish(plan.apply(pano, engine=engine))

    def workflow_cv():
        for plan in cvplans:
            finish(plan.apply(pano))

    def workflow_eager():
        for target in targets:
            finish(eager(pano, target, 320))

    ops = {
        "rust_prepared_scalar": partial(workflow_native, "scalar"),
        "rust_prepared_auto": partial(workflow_native, "auto"),
        "opencv_float_roi": workflow_cv,
        "numpy_eager": lambda: with_sampler(NUMPY_SAMPLE, workflow_eager),
        "rust_eager": lambda: with_sampler(rust_sample, workflow_eager),
    }
    report["workflow"] = {
        "view_count": len(plans),
        "timings": measure(ops, args.workflow_iterations),
        "rust_plan_bytes": sum(p.nbytes for p in plans),
        "opencv_map_bytes": sum(p.nbytes for p in cvplans),
        "note": "Fused paths have identical transforms; eager paths have different pixel filtering. Includes one point/box mapping and polygon rendering per view.",
    }
    print("workflow", json.dumps(report["workflow"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

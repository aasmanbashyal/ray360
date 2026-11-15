"""Measure one view and a complete geometry workflow; inference and I/O are excluded."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from ray360 import Box2D, MapCache, Panorama, Perspective, ProjectionMap, coverage_views


def measure(operation, iterations):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def peak_rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if platform.system() == "Darwin" else rss * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--view-size", type=int, default=320)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.width, args.height, args.view_size, args.iterations) < 1:
        parser.error("dimensions and iterations must be positive")
    image = np.random.default_rng(42).integers(0, 256, (args.height, args.width, 3), np.uint8)
    panorama = Panorama(image)
    target = Perspective(args.view_size, args.view_size, yaw=179, pitch=20)
    cache = MapCache()
    targets = coverage_views(args.view_size, args.view_size)
    padded_size = args.view_size + max(1, args.view_size // 4)
    box = Box2D(
        args.view_size * 0.25, padded_size * 0.35, args.view_size * 0.75, padded_size * 0.65
    )

    def full_workflow():
        for view in panorama.views(targets, cache=cache):
            view = view.letterbox(args.view_size, padded_size)
            view.transform.to_panorama([[args.view_size / 2, padded_size / 2]])
            for region in view.map_boxes([box]):
                region.polygons(panorama.projection)

    panorama.view(target, cache=cache)
    full_workflow()
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "source_size": list(panorama.projection.size),
        "view_size": list(target.size),
        "coverage_view_count": len(targets),
        "iterations": args.iterations,
        "map_build": measure(
            lambda: ProjectionMap.build(panorama.projection, target), args.iterations
        ),
        "cached_single_view": measure(lambda: panorama.view(target, cache=cache), args.iterations),
        "full_panorama_geometry": measure(full_workflow, args.iterations),
        "cache_bytes": cache.nbytes,
        "cache_budget_bytes": cache.max_bytes,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "scope": "Already-decoded pixels; all coverage views, letterboxing, one point and box mapping per view, polygon rendering. Excludes decoding, inference, NMS and file output.",
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

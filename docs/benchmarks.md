# Geometry workflow benchmark

```bash
uv sync --locked
uv run --no-sync python benchmarks/benchmark.py --width 4096 --height 2048 --iterations 5
uv run --no-sync python benchmarks/benchmark.py --width 8192 --height 4096 --view-size 640 --iterations 5
```

The script runs on Windows, Linux, and macOS. It reports runtime versions, platform,
CPU count, input/view dimensions, view count, cached map bytes, process peak resident
memory, and observed p50/p95 timings after warmup.

The full-panorama measurement sequentially extracts every coverage view, letterboxes
each one, maps a point and synthetic box, and renders the resulting spherical
boundary into the panorama. Only one view is retained at a time. Inputs are already
decoded. Decoding, model inference, NMS, disk I/O, and encoding are excluded.

Use `--output report.json` to retain a result. These measurements describe the
specified hardware and workload. They do not establish detector accuracy or a
real-time throughput guarantee. A five-iteration p95 is a small-sample observation.
Measure the consuming application's complete pipeline before choosing view size,
overlap, sampling, or concurrency.

The earlier conversion/tensor measurements were removed with the broader API.

## Observed Windows run

[The saved report](benchmark-cpu.json) uses Python 3.12.13, NumPy 2.5.4, a
4096 by 2048 uint8 RGB input, and 24 views of 320 by 320 pixels. The environment
reports 16 logical CPUs. Five iterations followed warmup.

| Operation | p50 | p95 |
|---|---:|---:|
| Build one coordinate map | 31.4 ms | 33.4 ms |
| Extract one cached view | 29.4 ms | 30.8 ms |
| Complete panorama geometry workflow | 1,851.6 ms | 1,950.0 ms |

Cached maps occupied 22.0 MiB and process peak resident memory was 95.6 MiB.
The full workflow measurement includes preprocessing and mapping, as described
above. It excludes model inference. This configuration is intended for offline
processing; the measurements do not support a real-time claim.

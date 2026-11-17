# Rust sampling for Ray360

Build this optional package from the checkout to accelerate Ray360's regular
view, preprocessing, and cubemap calls. The Ray360 core from this checkout
uses Rust automatically when the native package is installed.

The kernel implements nearest and bilinear sampling for C-contiguous `uint8`
HW/HWC images with contiguous `float32` or `float64` coordinates. It retains
horizontal wrapping, vertical clamping, ties-to-even rounding, nonfinite
coordinate masking, and explicit validity masks. Other dtypes and layouts use
NumPy, including boolean and 64-bit integer label images. Float32 coordinates
for source dimensions above 65,535 also use NumPy to preserve NumPy 1.x promotion.

PyO3 and rust-numpy borrow the input buffers without copying. The Rust output
allocation becomes the returned NumPy array without a second pixel copy. Both
paths use one CPU thread and hold the Python GIL while reading borrowed arrays.
The direct sampler uses bounds-checked Rust and enables SSE4.1 rounding on
supported x86-64 processors. The prepared sampler confines
unsafe AVX2 intrinsics to a runtime-dispatched function with checked gather
bounds and masked tail lanes. There is no OpenCV code, Rayon pool, global
`target-cpu=native` build flag, FMA interpolation, or fast-math setting.

## Build and use

Install a Rust toolchain and the platform's native linker first. Windows uses
the Visual Studio C++ build tools. Run one of these commands from the repository
root to install into your active Python environment:

```sh
# Direct native sampling, with NumPy as the only runtime dependency.
uv pip install ./native

# Resolve a released Ray360 for PreparedView and fallback on unsupported inputs.
uv pip install "./native[ray360]"

# Install the core and native package from this checkout for automatic dispatch.
uv pip install . ./native
```

For development and benchmarking, first follow the setup in `CONTRIBUTING.md`.

```sh
uv pip install ./native
uv run --no-sync python -m pytest tests native/tests -q
uv run --no-sync python native/benchmark.py --output .test-results/rust-benchmark.json
```

To build the same type of locked wheel as CI, use the development environment's
Python executable. On Windows, use `.venv/Scripts/python.exe` in the command.

```sh
uvx maturin@1.16.0 build --release --locked --manifest-path native/Cargo.toml --interpreter .venv/bin/python --out native/target/wheels
```

The wheel is in `native/target/wheels`. The backend has its own `pyproject.toml`
and Cargo lockfile. Use `uv sync --inexact` to retain the backend during a sync.

Use the standard API after installing the backend:

```python
import numpy as np
from ray360 import Panorama, Perspective

pixels = np.zeros((360, 720, 3), dtype=np.uint8)
panorama = Panorama(pixels)
result = panorama.view(Perspective(320, 240)).image
```

`ray360_native.sample` is also available for callers with their own coordinate
maps. Its Rust sampling path only requires NumPy. Install `ray360` for
`PreparedView` or for sampling inputs outside the Rust kernel's contract.
Installing `./native[ray360]` selects that dependency automatically. With a
released Ray360 version that predates automatic dispatch, call the native API
directly.

## Fused preprocessing and SIMD

`PreparedView` composes the existing `ViewTransform` operations without sampling
intermediate images. Rust computes the final panorama coordinates and builds
an owned interpolation plan. It retains one source index and two exact float32
weights per active lane, using lossless packing where possible. Packed flags
recover the other three indices exactly, including
wrapping and clamping. Each frame samples the original panorama into the final output.

```python
from ray360 import Panorama, Perspective, ViewTransform
from ray360_native import PreparedView

panorama = Panorama(pixels, source_id="capture", timestamp=1.25)
transform = (
    ViewTransform(panorama.projection, Perspective(640, 480, yaw=180))
    .crop(8.25, 6.5, 620, 464)
    .resize(320, 240)
    .letterbox(320, 320)
)
prepared = PreparedView(transform)
view = prepared.apply(panorama)
print(prepared.nbytes)  # Owned map/plan array bytes, not total process memory
```

The returned `View` retains its transform, source ID, timestamp, content mask,
and point/box mapping. Reuse the plan for frames with the same source projection.
Changing orientation or dimensions requires a new plan.

The prepared path currently uses bilinear sampling. Contiguous `uint8` images
with 1, 3, or 4 channels use eight-pixel AVX2 batches on supported x86-64 CPUs;
other CPU architectures and channel counts use scalar Rust. Image buffers above
2 GiB also use scalar Rust. `engine="scalar"` forces the portable path, and
`engine="avx2"` requires supported inputs and hardware. Other image dtypes or
layouts use NumPy when `engine="auto"`; bilinear restrictions on label dtypes
remain in effect. If either source dimension exceeds 65,535 pixels, `auto`
uses the NumPy sampler to preserve its coordinate arithmetic. Explicit Rust
engines require dimensions within that limit. The original `sample()` supports
nearest interpolation.

Input buffers and prepared indices are bounds-checked. SIMD gathers that would
read beyond the final RGB/grayscale pixel fall back to scalar sampling. Invalid
blocks remain zero without reading source pixels. The SIMD calculation preserves
the scalar multiplication/addition order and explicitly rounds ties to even.

Fusing operations preserves the coordinate transform, but **does not promise the
same pixels as repeated resampling**. It samples once, so intermediate blurring
and aliasing differ. Use the eager API if successive-resampling behavior is
required. Fusing does not add antialiasing or establish detector accuracy.

Plans own their arrays and are not inserted into Ray360's global cache. Completely
invalid eight-pixel blocks have no stored indices or weights. Ordinary interior
blocks use constant neighbor offsets in AVX2; boundary blocks decode packed flags.
Each block attempts to encode indices as unsigned 16-bit offsets from a shared
origin and weights as unsigned 16-bit binary fractions. Every weight must reproduce
its original float32 bit pattern exactly after decoding. An offset overflow or
any inexact fraction sends the entire block to the wider format. Weights of 1.0,
very small fractions, and widely separated source indices can require that fallback.
Every packed block preserves the original interpolation weights.

Packed blocks occupy 64 aligned bytes per eight lanes, or eight bytes per lane,
including their output position, validity/border flags, and source-end marker.
Wide blocks occupy 112 bytes on a 64-bit build. Separate boxed arrays avoid paying
the wide allocation size for packed blocks and retain no spare vector capacity.
The final partial block can contain up to seven unused lanes. Native plan getters
`packed_blocks` and `wide_blocks` expose the actual storage mix.

The AVX2 path decodes packed values back to the original float32 values before
interpolating, with the same multiplication/addition and ties-to-even rounding.
It prefetches two source pairs from an upcoming interior block. All prefetch
pointers stay within validated image bounds. Rust's
[`_mm_prefetch`](https://doc.rust-lang.org/core/arch/x86_64/fn._mm_prefetch.html)
is a cache hint whose effect depends on the processor; it does not change pixels.

The temporary coordinate and validity arrays are released after construction.
Accessing `prepared.xy` or `prepared.valid` regenerates and caches both read-only
arrays; the NumPy fallback does this automatically. It adds 9 bytes per output
pixel and a one-time geometry computation. Contiguous uint8 sampling needs neither
array. `nbytes` counts all retained plan storage plus any cached diagnostic maps.
It excludes small object/shape headers, allocator overhead, transient construction
buffers, input/output images, and process memory. Setup still temporarily creates
full coordinate maps; this change primarily reduces memory retained between frames.

For a comparison that keeps compact plans free of diagnostic maps, run:

```sh
uv run --no-sync python native/benchmark_memory.py --image .test-data/panorama/examplepano.jpg --iterations 600 --workflow-iterations 120 --output .test-results/rust-memory.json
```

The optional `--baseline-package PATH` loads a saved previous `ray360_native`
package directory, including its `__init__.py` and compatible native binary.
Save that directory before rebuilding to compare both implementations in one
process. `--extra-baseline-package PATH` can compare a second saved variant.
The report includes retained bytes, packing counts, scalar and auto timings, exact pixel
checks, setup, and 24-view sampling/geometry workflows. OpenCV remains an optional
benchmark dependency. Do not run tests or builds concurrently with timing runs.
Backends rotate through every timing position in both directions. Coordinate
inspection uses a separate plan, so neither the current nor previous runtime plan
silently accumulates diagnostic maps during the memory comparison.

With OpenCV installed, run the matched fused benchmark:

```sh
uv run --no-sync python native/benchmark_fused.py --image .test-data/panorama/examplepano.jpg --iterations 30 --workflow-iterations 6 --output .test-results/rust-fused.json
```

It compares prepared scalar Rust, AVX2, the direct Rust sampler, NumPy, and
OpenCV float/fixed maps using the same coordinates. OpenCV remaps only the valid
rectangle into a zeroed output, so it also avoids processing padding. The report
separates warm timing, setup costs, array memory, coordinate accuracy, pixel
accuracy, and the deliberate pixel differences from eager preprocessing.
OpenCV's setup timings start with shared fused geometry; Rust's setup includes
geometry generation and the interpolation plan, so those setup columns measure
different work. Warm timings are matched. Output allocation is included.
This broader benchmark deliberately accesses diagnostic maps for its NumPy and
direct-sampler comparisons, so its Rust memory figures include those cached maps.

## Validation and measurement

The native tests compare random inputs against NumPy and independently check
seams, clamping, and rounding. They cover both coordinate precisions, grayscale,
multiple channel counts, invalid coordinates, masks, strides, large label IDs,
fractional preprocessing, poles, and cubemap round trips.

The tests also run public APIs with the Rust backend installed and compare their
results with NumPy. The core suite checks that processing works without the backend.

The benchmark checks pixel equality and records warm p50/p95 timings for
320- and 640-pixel views and the existing 24-view geometry workflow. Both paths
perform identical preprocessing; neither uses padding/crop copy shortcuts.
Backends rotate through each timing position to reduce the effect of changing
system load and CPU clocks. Pass `--baseline-package PATH` to compare a saved
copy of `ray360_native` using the same inputs.
Compilation, map creation, inference, decoding, and image I/O are excluded.
Pass `--image PATH` for a real panorama and `--opencv` to compare an installed
OpenCV sampler with one native thread. That OpenCV comparison measures only
view sampling, not the full workflow.

### Standard API with Rust enabled

An 18-round run used a 4096 × 2048 photo on Windows x86-64 with CPython 3.12 and
Rust 1.100.0. The benchmark reused coordinate maps and checked that each backend
produced identical pixels for the individual views.

| Operation, median | NumPy | Previous Rust prototype | Updated Rust |
| --- | ---: | ---: | ---: |
| 320 × 320 bilinear view | 40.84 ms | 7.19 ms | 5.89 ms |
| 640 × 640 bilinear view | 196.71 ms | 30.15 ms | 25.21 ms |
| 24-view workflow with letterboxing and annotation mapping | 2,093.66 ms | 630.28 ms | 601.58 ms |

The full workflow still spends time generating preprocessing coordinates and
mapping annotations in Python. For repeated frames with fixed geometry,
`PreparedView` is available when a single resampling step suits the application.

### Lossless packed plans and tail latency

The next optimization packs weights and block-relative indices only when their
values round-trip exactly. It keeps wide records for every exception. The 24-view
benchmark packs 208,628 of 230,400 active blocks; 21,772 use the wide format.

| Plans | Previous compact Rust | Packed Rust | OpenCV float ROI |
| --- | ---: | ---: | ---: |
| One 320-square output | 1,075,200 bytes | 668,112 bytes | 614,400 bytes |
| One 640-square output | 4,300,800 bytes | 2,652,720 bytes | 2,457,600 bytes |
| 24 320-square views | 24.609 MiB | 15.059 MiB | 14.062 MiB |

The 24-view plan storage is 38.8% smaller than the previous compact format and
7.1% larger than OpenCV float maps. The savings depend on coordinate precision
and index locality. Arbitrary maps can use the wide format throughout. Diagnostic
map access still adds nine bytes per output pixel, and construction still creates
temporary full coordinate maps. These numbers describe retained plan storage,
not peak RSS or total application memory.

The paired optimization comparison used 240 single-view rounds and 40 workflow
rounds, rotating and reversing all five backends through every measurement
position. It loaded the previous compact binary, packed plans with and without
prefetching, and OpenCV into one process. The prefetched version reduced p95
against the previous compact binary from 2.397 to 1.984 ms at 320 pixels and from
7.341 to 5.980 ms at 640 pixels. The packed version without prefetching measured
2.127 and 6.962 ms. Across 24 views, sampling p95 fell from 58.978 to 51.588 ms;
the complete geometry workflow fell from 184.721 to 176.429 ms. This comparison
is saved in `.test-results/cpu-research/rust-packed-prefetch-comparison.json`.

An independent confirmation run used 600 rounds per individual view and 120 per
workflow on the same real 4096 by 2048 RGB image. Both Rust and OpenCV 5.0.0 used
one thread; all backends included output allocation and shared fused geometry.
Each comparison below is against the saved previous compact binary in that same
run, rather than an earlier run's absolute timings.

| Operation, p95 | Previous compact Rust | Packed Rust | OpenCV float ROI |
| --- | ---: | ---: | ---: |
| Fused 320-square output | 2.132 ms | 1.429 ms | 2.885 ms |
| Fused 640-square output | 8.928 ms | 6.524 ms | 11.076 ms |
| 24 views, sampling only | 62.604 ms | 50.901 ms | 86.004 ms |
| Complete 24-view workflow | 185.727 ms | 179.846 ms | 210.392 ms |

Observed p95 decreased by 33.0%, 26.9%, 18.7%, and 3.2% respectively. The smaller
complete-workflow gain reflects shared Python geometry work and timing variability;
it is not a guarantee for every frame. Medians also improved: packed Rust measured
0.708/3.753 ms for the two individual views, 44.080 ms for 24-view sampling, and
147.781 ms for the complete workflow. Previous compact Rust measured 0.810/5.091,
52.399, and 156.606 ms respectively. Scalar p95 did not improve: it measured
8.165/21.782 ms versus 7.787/21.732 ms previously. The tail-latency improvement is
established here for the AVX2 path, not the portable scalar fallback.

Rust matched the single-pass float32 NumPy reference exactly in both sizes and
all 24 workflow views. OpenCV float maps had seven one-level channel differences
across the workflow; fixed maps differed by up to three levels. Fused geometry,
rounding, metadata, and the existing eager-versus-fused semantics are unchanged.
Preparation remained a one-time cost: packed medians were 11.125/46.556 ms versus
11.074/45.526 ms previously. Accessing diagnostic maps still regenerates geometry.

The confirmation report is `.test-results/cpu-research/rust-packed-final-photo.json`.
It contains all timing samples and SHA-256 hashes of both native binaries. To
repeat the comparison, add
`--baseline-package .test-results/cpu-research/compact_baseline_native` to the
memory benchmark command. The optional no-prefetch variant was saved locally at
`.test-results/cpu-research/packed_no_prefetch_native` for the optimization comparison.

Validation passed 176 native tests, including a sweep of all 65,536 representable
fractions on each axis through scalar and AVX2 sampling. Tests also force wide
fallbacks for finer fractions, weights of exactly 1.0, and index-offset overflow,
plus mixed validity, out-of-order packed/wide blocks, tails, seams and image edges.
The existing suite with the direct Rust hook passed 113 tests with 7 optional skips.
Clippy, Ruff, formatting, and core mypy checks passed. The tested platform remains
Windows CPython 3.12 on a Ryzen 9 7940HS with AVX2; prefetch performance on other
processors has not been established.

### Earlier compact plan memory

The compact format keeps float32 weights and the same arithmetic/rounding order.
It removes redundant neighbor indices and completely invalid blocks, then releases
the preparation coordinates. It also writes fully valid SIMD blocks after checking
their output span once. No interpolation precision was reduced.

For the benchmark's 25% letterbox padding, retained storage changed as follows:

| Plans | Previous Rust | Compact Rust | OpenCV float ROI |
| --- | ---: | ---: | ---: |
| One 320-square output | 3.369 MiB | 1.025 MiB | 0.586 MiB |
| One 640-square output | 13.477 MiB | 4.102 MiB | 2.344 MiB |
| 24 320-square views | 80.859 MiB | 24.609 MiB | 14.062 MiB |

This is a 69.6% reduction in retained Rust plan storage. OpenCV float maps still
use less memory: compact Rust stores 1.75 times their bytes for these cases.
Accessing the lazy diagnostic maps raises a compact 320 plan to 1,996,800 bytes
and a 640 plan to 7,987,200 bytes, still 43.5% below the previous implementation.
A dense, unpadded plan uses about 14 bytes per output pixel instead of 34.5,
so the reduction depends on the amount and distribution of padding.

The final comparison loaded the saved previous native binary and the compact
release build in the same process. It used the real 4096 by 2048 panorama,
OpenCV 5.0.0 with one thread, 100 alternating single-view rounds, and 40 rounds
per 24-view workflow. The complete workflow also maps a point and box and renders
polygons for each view. Warm medians, including output allocation, were:

| Operation | Previous Rust | Compact Rust | OpenCV float ROI |
| --- | ---: | ---: | ---: |
| Fused 320-square output | 0.658 ms | 0.627 ms | 0.924 ms |
| Fused 640-square output | 4.571 ms | 4.432 ms | 5.755 ms |
| 24 views, sampling only | 46.983 ms | 43.952 ms | 53.223 ms |
| Complete 24-view workflow | 134.449 ms | 128.910 ms | 149.774 ms |

Compact Rust reduced median times by about 3-6% relative to the previous plan.
The two single-view p95 times were slightly higher: 2.256/8.180 ms versus
2.090/7.930 ms. The complete-workflow p95 improved from 156.871 to 148.555 ms.
System timing varied across runs, so these medians do not establish that every
frame becomes faster. The report retains each timing sample for inspection.
Compact scalar medians were 4.454/13.510 ms versus 4.619/13.755 ms previously.
Full Rust preparation took 9.539/36.655 ms versus 10.805/39.936 ms previously.
OpenCV setup in the report starts with shared coordinates and measures less work.

All Rust outputs matched the float32 single-pass NumPy reference exactly in both
sizes and all 24 workflow views. OpenCV float maps matched both single-view cases;
over the 24 views, seven channel values differed by one intensity level. Fixed
OpenCV maps used less memory but differed by up to three levels over the workflow.
Neither fusion semantics nor geometry changed in this memory optimization.

The complete report, with timings, accuracy and storage accounting, is
`.test-results/cpu-research/rust-memory-final-photo.json`. To repeat the local
comparison, add `--baseline-package .test-results/cpu-research/baseline_native`
to the memory benchmark command above. The saved baseline is a local artifact,
not a dependency or published package.

The release build passed 166 native tests. Added cases cover lazy map accounting,
omitted padding blocks, sparse output positions, partial SIMD blocks, borders,
and unquantized weights. Rust scalar and auto outputs matched NumPy exactly.
The existing suite with the direct Rust hook passed 113 tests with 7 optional
skips. Clippy, Ruff, formatting, and core mypy checks passed. Validation remains
limited to Windows CPython 3.12 on the Ryzen 9 7940HS.

### Earlier prepared SIMD benchmark

The follow-up run on the same Ryzen 9 7940HS used a real 4096 by 2048 RGB
panorama, 30 alternating sampling rounds, and 6 complete-workflow rounds. Both
Rust and OpenCV 5.0.0 used one thread. These are warm medians with output
allocation included and prepared maps retained between frames.

| Operation | Rust auto, AVX2 selected | OpenCV float maps, valid ROI only |
| --- | ---: | ---: |
| Fused 320-square output | 0.766 ms | 1.304 ms |
| Fused 640-square output | 4.490 ms | 5.459 ms |
| Complete 24-view workflow | 139.743 ms | 147.178 ms |

The prepared scalar Rust medians were 3.586 ms and 14.614 ms. AVX2 reduced
those times by about 4.7x and 3.3x. At 640 pixels the auto-path p95 was
8.219 ms versus OpenCV's 8.196 ms, so the median improvement does not establish
a consistent tail-latency improvement. The full-workflow advantage was only
about 5%; more hardware and repeated runs are needed before generalizing it.
Polygon rendering and mapping remain common costs in both workflows.

Rust scalar, auto, and forced AVX2 outputs matched the single-pass NumPy
reference exactly in both benchmark sizes. Independently generated coordinates
from `ViewTransform.to_panorama()` also produced identical reference pixels.
Float32 coordinate error was at most 0.000123 source pixels horizontally and
0.000062 vertically. OpenCV float maps matched those pixels too. Fixed-point
OpenCV maps differed by up to 2 intensity levels and were not consistently faster.

The fused output differed from eager extraction/crop/resize/letterbox by mean
absolute errors of 0.835 and 0.368 intensity levels at 320 and 640 respectively,
with maxima of 50 and 33. These differences reflect changed filtering, not
equivalence to the eager image pipeline. The transform and metadata are retained.
This workflow adds a fractional crop and resize, so its eager timings should
not be compared directly with the initial benchmark below.

Rust geometry plus plan preparation took about 12.6 ms at 320 pixels and
41.5 ms at 640. One 320 plan owned 3,532,800 array bytes versus 614,400 for
OpenCV's valid-ROI float map. Across 24 views, this was 80.86 MiB versus
14.06 MiB. This optimization exchanges memory and preparation work for lower
repeated-frame sampling cost. Reuse prepared plans rather than rebuilding per frame.

Validation passed 155 native tests, including scalar/AVX2 parity, odd tails,
tiny images, final-pixel gather bounds, nonfinite coordinates, seams, poles,
fractional transforms, empty padding, and source metadata. The existing suite
with the direct Rust sampler hook passed 113 tests and skipped 7 optional tests.
Clippy, Ruff, formatting, and core mypy checks passed. The tested build is still
Windows CPython 3.12; other platform wheels have not been validated.

The local report is `.test-results/cpu-research/rust-fused-final-photo.json`.

### Initial scalar prototype run

On November 15, 2026, the release wheel was built with Rust 1.100.0, PyO3 0.27.2,
and rust-numpy 0.27.1. Tests ran on Windows with Python 3.12.13 and NumPy 2.5.4.
The CPU was an AMD Ryzen 9 7940HS. Ten alternating measurement rounds used the
local 4096 by 2048 `examplepano.jpg` fixture. OpenCV 5.0.0 used one thread.

| Operation | NumPy p50 | Rust p50 | OpenCV p50 |
| --- | ---: | ---: | ---: |
| 320-square bilinear view | 35.41 ms | 5.39 ms | 1.61 ms |
| 640-square bilinear view | 170.80 ms | 24.52 ms | 8.14 ms |
| 320-square nearest view | 11.61 ms | 3.18 ms | Not measured |
| 640-square nearest view | 52.09 ms | 12.74 ms | Not measured |
| 24-view geometry workflow | 2,472.32 ms | 723.45 ms | Not measured |

The workflow p95 was 2,881.21 ms for NumPy and 790.90 ms for Rust. These are
small-sample observations on one machine, not throughput guarantees. Rust and
NumPy pixels matched exactly in all four view benchmark cases. OpenCV remains
faster for sampling; the prototype does not establish that Rust beats OpenCV.

The native suite passed 49 tests. The existing suite with the sampler hook passed
113 tests and skipped 7 optional tests. Rust Clippy, Python Ruff, and the existing
core mypy checks passed. Only the Windows CPython 3.12 wheel has been exercised.
The local JSON report is `.test-results/cpu-research/rust-final-photo.json`.

These initial measurements describe `sample()`, before the prepared SIMD path.
The sampler alone does not remove Python preprocessing coordinate work or polygon
rendering, which limits its full-workflow gain.

Before incorporating this into the main package, validate platform wheels,
minimum dependency versions, cold costs, peak memory, and a wider range of
images and hardware. Further ports should follow measurements: cubemap lookup
preparation and region membership are
candidates. Small Python metadata objects and serialization are not current
performance priorities.

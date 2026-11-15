# Validation

```bash
uv sync --locked --group dev --extra image
uv run --no-sync python -m pytest
uv run --no-sync ruff check src tests scripts examples benchmarks
uv run --no-sync ruff format --check src tests scripts examples benchmarks
uv run --no-sync mypy
uv build --no-sources
uv run --no-sync python scripts/check_wheel.py
uv run --no-sync python scripts/check_wheel.py --image
```

The wheel checker creates a fresh environment, installs the built distribution,
and runs outside the checkout. It verifies imports, the small dependency set,
removed modules, typing marker, preprocessing, prediction mapping, and persistence.
It needs package-index access for dependencies unless uv already has cached wheels.

The core suite uses analytic coordinate fields and known spherical regions. It
checks seam interpolation, poles, rotations, full-sphere coverage, inverse
preprocessing, clipped boxes, retained padding, label-ID preservation, map cache
budgets/concurrency, and experimental overlap. Rotated whole-sphere and wide-region
tests specifically guard the previous boundary/interior regression. Invalid inputs
and persisted-state validation run without optional packages.
Cube tests use independent XYZ direction fields, exact face-edge samples, rotated
targets, and large integer label IDs to catch axis flips and cross-face filtering errors.

Image tests skip when Pillow is absent. Two stitched photos are optional:

```bash
uv run --no-sync python scripts/fetch_test_data.py
uv run --no-sync python scripts/fetch_test_data.py --verify-only
uv run --no-sync python -m pytest --public-data
```

[The manifest](../tests/data-sources.json) records source URLs, provenance, byte
counts, and SHA-256 hashes. The downloader verifies hashes and bounds downloads;
pytest never downloads anything. The default download includes only the two photos.
`--include-e2e` adds six photographic cube faces, two video encodings, and one
detector checkpoint. Source media and weights remain outside built distributions.

The optional photo check exercises extraction, crop/letterbox inverses, prediction
mapping, overlays, and output files. It supplies synthetic boxes, not detector
ground truth. It does not certify model accuracy or physical camera calibration.

CI is configured for Linux, Windows, and macOS, a Python version matrix, a NumPy-only
installed wheel, and the minimum declared NumPy/Pillow versions on Python 3.11.

## Real media and model validation

The external dependencies stay in the consuming application's environment. A
separate environment keeps them out of the NumPy-only core installation. These
commands use the PyAV and Ultralytics versions tested locally:

```bash
uv venv --python 3.12 .venv-e2e
# On Windows, use .venv-e2e/Scripts/python.exe below instead.
uv pip install --python .venv-e2e/bin/python -e ".[image]" pytest av==19.0.2 ultralytics==8.4.213
uv run --no-project --python .venv-e2e/bin/python python scripts/fetch_test_data.py --include-e2e
uv run --no-project --python .venv-e2e/bin/python python -m pytest tests/test_end_to_end.py --e2e --e2e-output .test-results/e2e-run -v -s
```

Use a new output directory for each run. `--public-data /path/to/cache` selects a
different asset cache. The suite verifies all eleven asset hashes before model
loading. Dependencies and fixtures must already be installed/downloaded; tests
make no download requests. The model runs on CPU, with two Torch worker threads.

The six integration tests cover:

- Two stitched photos converted to 512-pixel cube faces and back, followed by
  actual YOLO11n inference across overlapping 320 by 240 perspective views.
  Letterboxing produces 320-square model inputs. Prediction records retain the
  input boxes, inverse transforms, source IDs, and mapped spherical regions.
- Six independently supplied photographic cube faces converted to a panorama and
  compared against the upstream equirectangular photo of the same scene. This
  checks imported face orientation independently of a self round-trip.
- Complete MP4 and WebM decode, cube conversion, and six 320-square model views
  on **every frame**. Checks cover original presentation timestamps, mapped
  prediction inverses, metadata, cache budget, and successful end-of-stream.
- The actual command-line entry points in `examples/detect.py` and
  `examples/video_frames.py`, including saved overlays and JSON output.

The [Three.js video example](https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/webgl_video_panorama_equirectangular.html)
identifies the video fixtures as equirectangular. Both encode the same scene;
they are not two independent scenes. Their stored dimensions are 1920 by 1080,
which ray360 maps over the full sphere. The MP4 has 188 frames with timestamps
0.00 through 7.48 seconds; WebM has 189 frames from 0.04 through 7.56 seconds.
The suite preserves that offset instead of resetting timestamps to zero.

The model example converts RGB views to the BGR arrays expected by
[Ultralytics](https://docs.ultralytics.com/modes/predict/). The decoder uses
[PyAV](https://pyav.basswood.io/docs/stable/api/video.html) to produce RGB frames
and retains presentation time. Decoding, model selection, and timestamp policy
remain application responsibilities.

Each run saves a `report.json`, per-frame counts and timestamps, representative
overlays, cube faces, reconstructed panoramas, and prediction JSON. The report
records package versions and fixture hashes. The manual `real media and model
validation` GitHub workflow runs this suite separately from ordinary CI and
uploads the JSON report. Local results do not establish that hosted CI passed.

These are execution and geometry checks. The fixtures have no detector ground
truth, so detection counts do not establish precision, recall, or successful
deduplication. Video inference uses six non-overlapping cube views; an object
split across their boundaries may be missed. No tracking or merging runs in the
examples. The tests do not certify GPU, live streaming, variable-frame-rate
decoding, or raw-camera stitching.

## Observed local run

[The saved report](validation-e2e.json) records the Windows CPU run with Python
3.12.13, NumPy 2.5.4, Pillow 13.0.0, PyAV 19.0.2, Ultralytics 8.4.213, Torch
2.15.0+cpu, and torchvision 0.29.2+cpu. The complete suite passed 120 tests in 287.15 seconds, including all six
real-media integration tests and the independent cube-to-panorama reference check.

| Input | Model views processed | Mapped predictions | Conversion check |
|---|---:|---:|---|
| Pannellum photo, 4096 by 2048 | 34 | 1 | 29.79 dB round-trip PSNR |
| Ricoh THETA photo, 2048 by 1024 | 34 | 42 | 42.83 dB round-trip PSNR |
| MP4, all 188 frames | 1,128 | 668 | Cube conversion on every frame |
| WebM, all 189 frames | 1,134 | 781 | Cube conversion on every frame |

Photo PSNR compares the reconstruction from 512-pixel faces against a bilinearly
resized 1024 by 512 reference. The separately supplied 1584-pixel cube faces
reconstruct a 2048 by 1024 panorama at 28.78 dB against the independent upstream
photo. These are resampling checks, not detector metrics. Visual inspection of
the model overlays shows duplicate predictions and questionable detections.

The complete video workflows took about 115 and 115 seconds respectively on this
machine. Each kept about 5.5 MiB of coordinate maps. This is an offline CPU run;
it does not establish real-time throughput. The model CLI and both decoder CLI
runs also passed. Hosted CI has not been run from this workspace.

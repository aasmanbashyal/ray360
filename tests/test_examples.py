"""Argument failures must happen before importing or running an optional model."""

import subprocess
import sys
from pathlib import Path


def test_detector_rejects_colliding_output_paths_before_loading_model(tmp_path):
    checkpoint = tmp_path / "unused.pt"
    checkpoint.touch()
    output = tmp_path / "result.png"
    example = Path(__file__).resolve().parents[1] / "examples/detect.py"
    process = subprocess.run(
        [
            sys.executable,
            str(example),
            "unused.jpg",
            str(checkpoint),
            str(output),
            "--json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "different files" in process.stderr
    assert not output.exists()

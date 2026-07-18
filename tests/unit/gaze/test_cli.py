"""Gaze operational CLI tests that do not require a camera or model asset."""

from __future__ import annotations

import json
from pathlib import Path

from jarvis.gaze.cli import main


def test_evaluate_writes_reproducible_result(tmp_path: Path, capsys: object) -> None:
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "frame_id,timestamp_ms,predicted_target,ground_truth_target\n"
        "0,1000,laptop,laptop\n"
        "1,1033,UNKNOWN,room.bulb\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    exit_code = main(
        [
            "evaluate",
            "--input",
            str(trace),
            "--dataset-id",
            "bright-no-glasses-01",
            "--conditions",
            "bright, no glasses, 60cm",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["accuracy"] == 0.5
    assert payload["total_frames"] == 2
    assert payload["meets_target"] is False


def test_evaluate_rejects_missing_columns(tmp_path: Path) -> None:
    trace = tmp_path / "bad.csv"
    trace.write_text("frame_id,predicted_target\n0,laptop\n", encoding="utf-8")

    exit_code = main(
        [
            "evaluate",
            "--input",
            str(trace),
            "--dataset-id",
            "bad",
            "--conditions",
            "n/a",
        ]
    )

    assert exit_code == 2

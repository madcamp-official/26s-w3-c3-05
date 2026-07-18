"""README 13장 "필수 제약": Target Selection Accuracy ≥ 90%."""

from __future__ import annotations

import pytest

from jarvis.gaze.evaluation import LabeledFrame, compute_target_selection_accuracy


def test_accuracy_counts_matches_over_total() -> None:
    frames = [
        LabeledFrame(0, 1_000, "laptop", "laptop"),
        LabeledFrame(1, 1_050, "laptop", "laptop"),
        LabeledFrame(2, 1_100, "room.bulb", "laptop"),
        LabeledFrame(3, 1_150, "UNKNOWN", "UNKNOWN"),
    ]
    result = compute_target_selection_accuracy(frames, dataset_id="ds-1", conditions="bright, no glasses")
    assert result.total_frames == 4
    assert result.correct_frames == 3
    assert result.accuracy == pytest.approx(0.75)
    assert result.dataset_id == "ds-1"
    assert result.conditions == "bright, no glasses"


def test_raises_on_empty_trace() -> None:
    with pytest.raises(ValueError):
        compute_target_selection_accuracy([], dataset_id="ds-1", conditions="n/a")

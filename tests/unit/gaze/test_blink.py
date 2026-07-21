from __future__ import annotations

import pytest

from jarvis.gaze.blink import AdaptiveBlinkDetector
from jarvis.gaze.config import GazeConfig


def test_partial_blink_is_rejected_relative_to_personal_open_baseline() -> None:
    detector = AdaptiveBlinkDetector()

    assert detector.update(0.25, 0.24)
    assert not detector.update(0.15, 0.14)


def test_reopen_hysteresis_waits_for_eyelids_to_recover() -> None:
    detector = AdaptiveBlinkDetector()

    assert detector.update(0.25, 0.25)
    assert not detector.update(0.08, 0.08)
    assert not detector.update(0.18, 0.18)
    assert detector.update(0.22, 0.22)


def test_open_baseline_decays_slowly_instead_of_learning_a_partial_blink() -> None:
    config = GazeConfig(eye_openness_baseline_decay=0.01)
    detector = AdaptiveBlinkDetector(config)

    assert detector.update(0.25, 0.25)
    assert detector.update(0.24, 0.24)

    assert detector.open_baseline == pytest.approx(0.2475)

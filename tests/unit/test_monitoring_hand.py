"""Tests for the hand probe's honest degradation and status (no camera/model).

The probe does hand *tracking* only; it must never imply gesture *recognition*
is happening, and it must degrade honestly when the model/mediapipe is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gesture_fusion.config import GestureConfig
from jarvis.gesture_fusion.smoothing import OneEuroFilter
from jarvis.monitoring.gesture_source import UntrainedGestureSource
from jarvis.monitoring.hand_probe import (
    DEFAULT_IMAGE_SMOOTHING,
    HandProbe,
    ImageSmoothingConfig,
)


def test_probe_without_model_is_unavailable() -> None:
    probe = HandProbe(model_path=None)
    assert probe.start() is False
    assert probe.available is False
    assert "모델" in probe.status_text
    assert probe.process_bgr(object(), 0, 0) is None  # type: ignore[arg-type]


def test_gesture_recognition_status_is_honest_about_untrained_model() -> None:
    probe = HandProbe(model_path=None)
    status = probe.gesture_recognition_status
    assert "미학습" in status
    assert "비활성" in status
    # never claims recognition is active
    assert "인식됨" not in status


def test_untrained_gesture_source_yields_nothing() -> None:
    source = UntrainedGestureSource()
    assert source.available is False
    assert source.poll() == []
    assert "미학습" in source.status_text


def test_display_smoothing_defaults_on_and_toggles() -> None:
    probe = HandProbe(model_path=None)
    assert probe.smoothing is True  # smoothed vertices shown by default
    probe.set_smoothing(False)
    assert probe.smoothing is False
    probe.set_smoothing(True)
    assert probe.smoothing is True


# --- 오버레이 평활화: 이미지 좌표계 전용 파라미터 (지연 회귀 방지) ---
#
# 웹캠 스켈레톤은 이미지 정규화 좌표([0,1] 프레임 비율)를 평활화하는데, 모델 입력은
# 손바닥 정규화 좌표(palm-width)를 쓴다. One-Euro의 컷오프는 `min_cutoff + beta×|속도|`라
# 속도가 신호와 같은 단위로 들어가므로, palm 공간용 beta를 이미지 공간에 재사용하면
# 컷오프가 거의 열리지 않아 손을 늦게 따라온다. 아래 테스트가 그 회귀를 막는다.

_DT_MS = 33  # 30fps
_RAMP_SPEED = 0.5  # 프레임 폭/초 — 보통 속도의 손 이동


def _tracking_error_after_ramp(filt: OneEuroFilter, *, frames: int = 20) -> float:
    """일정 속도로 움직이는 신호를 따라갈 때 마지막 프레임의 추종 오차(절대값)."""
    error = 0.0
    for i in range(frames):
        timestamp_ms = i * _DT_MS
        truth = 0.2 + _RAMP_SPEED * (timestamp_ms / 1000.0)
        smoothed = float(np.asarray(filt.filter(np.array([truth]), timestamp_ms))[0])
        error = abs(truth - smoothed)
    return error


def test_image_space_smoothing_tracks_motion_far_better_than_model_params() -> None:
    """이미지 공간 전용 파라미터가 palm 공간 파라미터보다 확연히 덜 지연되어야 한다."""
    model_config = GestureConfig()
    palm_space = OneEuroFilter(
        min_cutoff=model_config.smoothing_min_cutoff,
        beta=model_config.smoothing_beta,
        d_cutoff=model_config.smoothing_d_cutoff,
    )
    image_space = OneEuroFilter(
        min_cutoff=DEFAULT_IMAGE_SMOOTHING.min_cutoff,
        beta=DEFAULT_IMAGE_SMOOTHING.beta,
        d_cutoff=DEFAULT_IMAGE_SMOOTHING.d_cutoff,
    )
    palm_error = _tracking_error_after_ramp(palm_space)
    image_error = _tracking_error_after_ramp(image_space)

    # 재사용하던 palm 파라미터는 프레임 폭의 2%가량 뒤처진다(640px에서 약 15px).
    assert palm_error > 0.015
    # 전용 파라미터는 1% 미만(약 3px)으로, 절반 이하 수준이어야 한다.
    assert image_error < 0.010
    assert image_error < palm_error * 0.40


def test_image_space_smoothing_lag_stays_flat_as_hand_speeds_up() -> None:
    """적응 컷오프가 이미지 단위에서 실제로 열리는지 — 지연이 속도에 따라 커지면 안 된다.

    palm 공간 파라미터를 재사용하면 beta가 거의 기여하지 못해 손이 빠를수록 지연이
    커진다(이 문제의 체감 증상). 전용 파라미터는 속도가 올라가도 지연이 거의 일정해야 한다.
    """

    def error_at(speed: float) -> float:
        filt = OneEuroFilter(
            min_cutoff=DEFAULT_IMAGE_SMOOTHING.min_cutoff,
            beta=DEFAULT_IMAGE_SMOOTHING.beta,
            d_cutoff=DEFAULT_IMAGE_SMOOTHING.d_cutoff,
        )
        error = 0.0
        for i in range(20):
            timestamp_ms = i * _DT_MS
            truth = 0.2 + speed * (timestamp_ms / 1000.0)
            smoothed = float(np.asarray(filt.filter(np.array([truth]), timestamp_ms))[0])
            error = abs(truth - smoothed)
        return error

    slow, fast = error_at(0.2), error_at(1.0)
    # 손이 5배 빨라져도 지연은 2배 미만으로만 늘어야 한다(컷오프가 함께 열리므로).
    assert fast < slow * 2.0


def test_image_space_smoothing_still_suppresses_jitter_on_a_still_hand() -> None:
    """지연을 줄이려고 평활화를 사실상 꺼버린 것이 아님을 확인한다."""
    rng = np.random.default_rng(0)
    filt = OneEuroFilter(
        min_cutoff=DEFAULT_IMAGE_SMOOTHING.min_cutoff,
        beta=DEFAULT_IMAGE_SMOOTHING.beta,
        d_cutoff=DEFAULT_IMAGE_SMOOTHING.d_cutoff,
    )
    truth = 0.5
    raw_deviations: list[float] = []
    smoothed_deviations: list[float] = []
    for i in range(60):
        noisy = truth + float(rng.normal(0.0, 0.002))  # mediapipe 수준의 지터
        out = float(np.asarray(filt.filter(np.array([noisy]), i * _DT_MS))[0])
        raw_deviations.append(noisy - truth)
        smoothed_deviations.append(out - truth)

    assert float(np.std(smoothed_deviations)) < float(np.std(raw_deviations)) * 0.75


def test_image_smoothing_params_are_not_coupled_to_model_params() -> None:
    """표시용 beta는 모델용 beta와 별개여야 한다 — 다시 묶이면 지연이 재발한다."""
    assert DEFAULT_IMAGE_SMOOTHING.beta > GestureConfig().smoothing_beta * 5


def test_image_smoothing_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="min_cutoff"):
        ImageSmoothingConfig(min_cutoff=0.0)
    with pytest.raises(ValueError, match="d_cutoff"):
        ImageSmoothingConfig(d_cutoff=-1.0)
    with pytest.raises(ValueError, match="beta"):
        ImageSmoothingConfig(beta=-0.1)

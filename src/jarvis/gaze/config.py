"""Named thresholds for the Gaze Targeting Engine.

README 7장 "초기 기준" 값을 코드의 단일 기준으로 옮긴 것이다. 값을 바꿀 때는
documents/gaze.md와 documents/decisions.md도 함께 갱신한다 (development-principles.md 8절).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GazeConfig:
    """Tunable thresholds for smoothing, classification and Gaze Lock.

    모든 필드는 README 7장에 명시된 의미와 단위를 그대로 따른다.
    """

    # Gaze Lock state machine (README 7장 "초기 기준")
    dwell_time_ms: int = 500
    """CANDIDATE 상태를 TARGET_LOCKED로 승격하기 전 유지해야 하는 최소 시간."""

    minimum_probability: float = 0.80
    """대상 후보로 인정하거나 Lock을 유지하기 위한 최소 top-1 확률."""

    minimum_margin: float = 0.20
    """Lock 유지에 필요한 top-1과 top-2 확률의 최소 차이."""

    target_lock_ttl_ms: int = 1500
    """TARGET_LOCKED/GESTURE_WAIT 상태가 만료되기까지의 유예 시간."""

    # Target classifier (README 7장 "Target 추정")
    unknown_probability_threshold: float = 0.80
    """최고 유사도 기반 확률이 이 값 미만이면 UNKNOWN으로 거부한다.

    Lock 승격에 쓰이는 minimum_probability와 같은 개념(최소 확신도)이므로 기본값을
    공유한다 — 서로 다른 값이 필요해지면 그 이유를 documents/decisions.md에 남긴다.
    """

    unknown_max_angle_deg: float = 25.0
    """가장 가까운 등록 방향과도 이 각도보다 멀면 UNKNOWN으로 거부한다.

    기기 간 상대 확률만 사용하면 등록 기기가 하나일 때 모든 방향의 확률이 1.0이
    되는 문제를 막는 절대 거리 안전 기준이다.
    """

    # Gaze vector composition (README 7장 "시선 방향 벡터 합성")
    max_eye_offset_deg: float = 30.0
    """홍채 상대 위치(-1..1)를 각도로 환산할 때 사용하는 눈 최대 회전각."""

    # Smoothing (README 7장에 정의되지 않은 구현 세부값 — gaze.md에 기록)
    smoothing_window_frames: int = 8
    """Temporal smoothing에 사용하는 최근 프레임 수."""

    minimum_tracking_confidence: float = 0.5
    """이 값 미만의 eye/face tracking confidence는 추적 손실로 취급한다."""

    ema_min_alpha: float = 0.15
    """낮은 confidence 프레임에 적용할 EMA 반영률."""

    ema_max_alpha: float = 0.65
    """높은 confidence 프레임에 적용할 EMA 반영률."""

    blink_hold_ms: int = 300
    """Short eye-closed intervals keep the last stable gaze instead of jumping."""

    small_motion_deadzone_deg: float = 5.0
    """Ignore tiny smoothed-gaze changes below this angle to reduce jitter."""

    UNKNOWN_TARGET: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if self.dwell_time_ms < 0 or self.target_lock_ttl_ms <= 0:
            raise ValueError("Gaze timing thresholds must be non-negative and TTL must be positive")
        if self.blink_hold_ms < 0:
            raise ValueError("blink_hold_ms must be non-negative")
        if self.smoothing_window_frames <= 0:
            raise ValueError("smoothing_window_frames must be positive")
        probability_fields = {
            "minimum_probability": self.minimum_probability,
            "minimum_margin": self.minimum_margin,
            "unknown_probability_threshold": self.unknown_probability_threshold,
            "minimum_tracking_confidence": self.minimum_tracking_confidence,
            "ema_min_alpha": self.ema_min_alpha,
            "ema_max_alpha": self.ema_max_alpha,
        }
        for name, value in probability_fields.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {value}")
        if self.ema_min_alpha > self.ema_max_alpha:
            raise ValueError("ema_min_alpha must not exceed ema_max_alpha")
        if not math.isfinite(self.max_eye_offset_deg) or self.max_eye_offset_deg <= 0.0:
            raise ValueError("max_eye_offset_deg must be finite and positive")
        if not math.isfinite(self.unknown_max_angle_deg) or not 0.0 < self.unknown_max_angle_deg <= 180.0:
            raise ValueError("unknown_max_angle_deg must be finite and within (0, 180]")
        if (
            not math.isfinite(self.small_motion_deadzone_deg)
            or self.small_motion_deadzone_deg < 0.0
        ):
            raise ValueError("small_motion_deadzone_deg must be finite and non-negative")
        if not self.UNKNOWN_TARGET:
            raise ValueError("UNKNOWN_TARGET must not be empty")


DEFAULT_GAZE_CONFIG = GazeConfig()

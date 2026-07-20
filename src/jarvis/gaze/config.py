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
    max_eye_offset_deg: float = 45.0

    head_yaw_weight: float = 0.45
    """How strongly head yaw contributes to gaze yaw before iris correction."""

    head_pitch_weight: float = 0.45
    """How strongly head pitch contributes to gaze pitch before iris correction."""

    head_only_confidence_scale: float = 0.45
    """Confidence multiplier when iris/eyes are unavailable and only head pose is used."""

    horizontal_axis_sign: float = -1.0
    """Sign correction from camera/MediaPipe horizontal motion to yaw(+right)."""
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

    tracking_loss_hold_ms: int = 800
    """Briefly keep the last gaze during full face-landmarker dropouts."""

    small_motion_deadzone_deg: float = 5.0
    """Ignore tiny smoothed-gaze changes below this angle to reduce jitter."""

    UNKNOWN_TARGET: str = "UNKNOWN"

    enable_3d_target_matching: bool = False
    """Use triangulated 3D geometry for live matching.

    Disabled by default for webcam demos: head translation from a monocular
    face model is noisy, so registration may store 3D diagnostics while
    classification stays on the more stable angle profile.
    """

    require_3d_target_registration: bool = False
    """Reject look-to-register targets unless multi-ray 3D triangulation succeeds."""

    # 3D triangulation (calibration/triangulation.py) — 10초 등록 동안 머리를
    # 움직여 얻은 여러 시선 광선으로 물체의 실제 위치·크기를 추정할 때의 품질
    # 기준. 기준을 만족하지 못하면 각도 기반(mean_direction + variance) 등록으로
    # 자동 대체한다(documents/decisions.md 참고).
    minimum_triangulation_baseline_mm: float = 60.0
    """광선 원점(머리 위치)들의 강건한 퍼짐(중앙값 기준 90퍼센타일*2)의 최소값.

    이 값보다 작으면 등록 중 머리가 충분히 움직이지 않은 것으로 보고 3D를
    포기한다 — 원점이 거의 고정된 채 눈만 움직인 경우, 삼각측량이 카메라
    바로 앞의 한 점으로 수렴해 버리는 것을 막는다.
    """

    minimum_triangulation_eigenvalue: float = 0.004
    """광선 방향들의 각도 다양성 하한(A 행렬의 최소 고유값 / 프레임 수).

    baseline_mm만으로는 "원점은 퍼졌지만 물체가 멀어 광선이 여전히 거의
    평행한" 경우를 잡지 못한다 — 이 값이 함께 낮으면 조건이 나쁜 것으로 본다.
    합성 광선(다양한 baseline·거리 조합)으로 실측 없이 보정한 값이다: baseline
    반경 150mm·거리 2000mm(스마트 전구 정도 거리) 조합은 고유값≈0.0057·위치
    오차 26mm로 통과시키되, baseline 60~100mm·거리 2000~3000mm(고유값
    0.0004~0.0025, 오차 35~300mm)는 거부한다 — 실제 카메라로 첫 통합 테스트를
    할 때(README 16장 Day 1) 재보정이 필요할 수 있다.
    """

    maximum_triangulation_residual_mm: float = 35.0
    """추정된 위치와 각 광선 사이 수직 거리의 RMS 상한 — 이보다 크면 광선들이
    한 점에서 잘 수렴하지 않은 것으로 보고 3D를 포기한다. 깊이 방향 오조건은
    residual만으로 잘 드러나지 않으므로(위 min_eigenvalue가 그 역할을 한다),
    이 값은 주로 서로 다른 물체를 보는 등 명백히 어긋난 광선을 걸러내는
    안전망이다."""

    minimum_triangulation_frames: int = 20
    """3D 삼각측량을 시도하기 위한 최소 유효 프레임 수."""

    target_radius_floor_mm: float = 20.0
    """등록된 물체의 유효 반경(radius_mm)이 이보다 작아지지 않도록 하는 하한.

    이 반경은 실제로 측정한 물체 크기가 아니라 삼각측량 잔차에서 유도한
    "판정 허용 오차"다 — 운 좋게 낮은 잔차가 나와도 비현실적으로 좁은 반경이
    되지 않도록 막는다.
    """

    target_minimum_angular_variance_deg: float = 4.0
    """3D 모드에서 계산한 각도 분산(atan(radius/depth)^2)의 하한(도).

    각도 기반 등록의 기존 최소 퍼짐(4도, target_registration.py)과 맞춰,
    3D 모드가 각도 모드보다 더 엄격하게(더 쉽게 UNKNOWN으로) 판정하지 않도록
    한다.
    """

    def __post_init__(self) -> None:
        if self.dwell_time_ms < 0 or self.target_lock_ttl_ms <= 0:
            raise ValueError("Gaze timing thresholds must be non-negative and TTL must be positive")
        if self.blink_hold_ms < 0:
            raise ValueError("blink_hold_ms must be non-negative")
        if self.tracking_loss_hold_ms < 0:
            raise ValueError("tracking_loss_hold_ms must be non-negative")
        if self.smoothing_window_frames <= 0:
            raise ValueError("smoothing_window_frames must be positive")
        probability_fields = {
            "minimum_probability": self.minimum_probability,
            "minimum_margin": self.minimum_margin,
            "unknown_probability_threshold": self.unknown_probability_threshold,
            "minimum_tracking_confidence": self.minimum_tracking_confidence,
            "head_only_confidence_scale": self.head_only_confidence_scale,
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
        for name, value in {
            "head_yaw_weight": self.head_yaw_weight,
            "head_pitch_weight": self.head_pitch_weight,
        }.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {value}")
        if self.horizontal_axis_sign not in (-1.0, 1.0):
            raise ValueError("horizontal_axis_sign must be either -1.0 or 1.0")
        if not math.isfinite(self.unknown_max_angle_deg) or not 0.0 < self.unknown_max_angle_deg <= 180.0:
            raise ValueError("unknown_max_angle_deg must be finite and within (0, 180]")
        if (
            not math.isfinite(self.small_motion_deadzone_deg)
            or self.small_motion_deadzone_deg < 0.0
        ):
            raise ValueError("small_motion_deadzone_deg must be finite and non-negative")
        if not self.UNKNOWN_TARGET:
            raise ValueError("UNKNOWN_TARGET must not be empty")
        positive_mm_fields = {
            "minimum_triangulation_baseline_mm": self.minimum_triangulation_baseline_mm,
            "maximum_triangulation_residual_mm": self.maximum_triangulation_residual_mm,
            "target_radius_floor_mm": self.target_radius_floor_mm,
        }
        for name, value in positive_mm_fields.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if (
            not math.isfinite(self.minimum_triangulation_eigenvalue)
            or self.minimum_triangulation_eigenvalue <= 0.0
        ):
            raise ValueError("minimum_triangulation_eigenvalue must be finite and positive")
        if self.minimum_triangulation_frames <= 0:
            raise ValueError("minimum_triangulation_frames must be positive")
        if (
            not math.isfinite(self.target_minimum_angular_variance_deg)
            or not 0.0 < self.target_minimum_angular_variance_deg <= 90.0
        ):
            raise ValueError("target_minimum_angular_variance_deg must be finite and within (0, 90]")


DEFAULT_GAZE_CONFIG = GazeConfig()

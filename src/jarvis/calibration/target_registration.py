"""Two-phase boundary collection for deterministic target profiles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from jarvis.calibration.registry import (
    TargetDirection,
    TargetRecord,
    TargetSpread,
)
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.gaze.feature_profile import (
    TargetFeatureSample,
    build_area_profile,
    build_feature_profile,
    build_pose_correction,
)
from jarvis.gaze.smoothing import SmoothedGaze


class RegistrationPhase(StrEnum):
    CENTER = "CENTER"
    BOUNDARY = "BOUNDARY"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True, slots=True)
class CoverageCondition:
    """1단계 자세·거리 coverage 한 구간의 진행 상황."""

    key: str
    label: str
    count: int
    required: int

    @property
    def met(self) -> bool:
        return self.count >= self.required


_YAW_SIDE_GROUP: tuple[str, ...] = ("left", "right")
_PITCH_SIDE_GROUP: tuple[str, ...] = ("up", "down")
_SCALE_SIDE_GROUP: tuple[str, ...] = ("near", "far")
_COVERAGE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("front",),
    _YAW_SIDE_GROUP,
    _PITCH_SIDE_GROUP,
    _SCALE_SIDE_GROUP,
)
"""완료 판정 단위. 좌/우, 상/하, 근/원은 짝 중 **한쪽만** 채우면 된다(아래
클래스 docstring 참고) — 정면만 개별로 필요하다."""


class PoseCoverageTracker:
    """1단계(중앙 응시 + 자세 스윕)의 조건 충족식 coverage 추적.

    시간제(20초) 등록은 스윕이 한쪽에 치우쳐도 완료돼, 실제로 정면·왼쪽
    보정점 없이 오른쪽 2개 bin만 저장된 등록이 발생했다(2026-07-22 실측).
    이 추적기는 정면/좌/우/상/하/근/원 각 구간의 유효 프레임 수를 센다.

    좌/우, 상/하, 근/원은 **한쪽만** 채우면 완료로 본다(2026-07-22 2차 실측: 물체가
    카메라 기준 오른쪽에 있으면, "고개 왼쪽"을 채우려면 눈이 물체 반대편에서
    다시 그만큼 꺾여야 해 보상각이 40도 안팎까지 벌어진다 — 반대쪽으로
    한없이 요구하면 물체 위치에 따라 구조적으로 못 끝나는 등록이 생긴다).
    반대로 물체가 중앙 근처면 스윕하다 보면 양쪽 다 자연스럽게 채워지므로
    막을 이유가 없다.

    근/원도 같은 문제가 있다(2026-07-22 3차 실측): 물체가 카메라에서 멀리
    떨어져 있으면 "가까이"(카메라 기준 face scale을 정면 대비 1.15배 이상)를
    채우려고 카메라 쪽으로 몸을 기울이는 동안 물체를 향한 시선 각이 급격히
    커져 등록이 무너지거나, 애초에 그 자세 자체가 부자연스러워 근/원 중 한쪽이
    구조적으로 채워지지 않는다. 근/원 각각을 개별 필수로 두면 물체 거리에
    따라 등록이 영영 끝나지 않으므로, 좌우/상하와 같은 방식으로 한쪽만
    요구한다.

    near/far는 절대 face scale이 아니라 '정면 구간에서 관측한 기준 scale'
    대비 배율로 판정한다 — 기준이 쌓이기 전(정면 10프레임 미만)에는 세지
    않는다(지어내지 않는다). 정면을 먼저 수집하도록 안내가 유도한다.
    """

    _MINIMUM_REFERENCE_FRAMES = 10

    def __init__(self, config: GazeConfig, minimum_frames: int) -> None:
        if minimum_frames <= 0:
            raise ValueError("minimum_frames must be positive")
        self._config = config
        self._minimum_frames = minimum_frames
        self._counts: dict[str, int] = {
            key: 0 for key in ("front", "left", "right", "up", "down", "near", "far")
        }
        self._front_scales: list[float] = []

    @property
    def reference_face_scale(self) -> float | None:
        if len(self._front_scales) < self._MINIMUM_REFERENCE_FRAMES:
            return None
        return float(np.median(np.asarray(self._front_scales, dtype=np.float64)))

    def add(self, sample: TargetFeatureSample) -> None:
        config = self._config
        if abs(sample.head_yaw) < config.coverage_yaw_front_threshold_deg:
            self._counts["front"] += 1
            self._front_scales.append(sample.face_scale)
        if sample.head_yaw <= -config.coverage_yaw_side_threshold_deg:
            self._counts["left"] += 1
        if sample.head_yaw >= config.coverage_yaw_side_threshold_deg:
            self._counts["right"] += 1
        if sample.head_pitch >= config.coverage_pitch_threshold_deg:
            self._counts["up"] += 1
        if sample.head_pitch <= -config.coverage_pitch_threshold_deg:
            self._counts["down"] += 1
        reference = self.reference_face_scale
        if reference is not None and reference > 0.0:
            ratio = sample.face_scale / reference
            if ratio >= config.coverage_scale_near_ratio:
                self._counts["near"] += 1
            if ratio <= config.coverage_scale_far_ratio:
                self._counts["far"] += 1

    def report(self) -> tuple[CoverageCondition, ...]:
        labels = {
            "front": "정면",
            "left": "고개 왼쪽",
            "right": "고개 오른쪽",
            "up": "고개 위",
            "down": "고개 아래",
            "near": "가까이",
            "far": "멀리",
        }
        return tuple(
            CoverageCondition(
                key=key,
                label=labels[key],
                count=count,
                required=self._minimum_frames,
            )
            for key, count in self._counts.items()
        )

    def complete(self) -> bool:
        conditions = {condition.key: condition for condition in self.report()}
        return all(
            any(conditions[key].met for key in group) for group in _COVERAGE_GROUPS
        )

    def missing_labels(self) -> list[str]:
        conditions = {condition.key: condition for condition in self.report()}
        missing = []
        for group in _COVERAGE_GROUPS:
            if any(conditions[key].met for key in group):
                continue
            missing.append("/".join(conditions[key].label for key in group))
        return missing

    def progress(self) -> float:
        """그룹별 충족률(짝은 더 앞선 쪽 기준) 평균(0..1) — UI 진행 막대에 그대로 쓴다."""
        conditions = {condition.key: condition for condition in self.report()}
        group_progress = [
            max(min(1.0, conditions[key].count / conditions[key].required) for key in group)
            for group in _COVERAGE_GROUPS
        ]
        return float(sum(group_progress) / len(group_progress))


class TargetRegistrationSession:
    def __init__(
        self,
        target_id: str,
        name: str,
        device_type: str,
        device_id: str,
        *,
        center_duration_ms: int = 20_000,
        boundary_duration_ms: int = 16_000,
        minimum_valid_frames: int = 30,
        minimum_boundary_frames: int | None = None,
        minimum_confidence: float = 0.35,
        maximum_jump_deg: float = 18.0,
        config: GazeConfig = GazeConfig(),
        coverage_min_frames: int = 0,
        raw_sample_dir: str | Path | None = None,
        requires_nod_gate: bool = False,
    ) -> None:
        if center_duration_ms <= 0 or boundary_duration_ms <= 0 or minimum_valid_frames <= 0:
            raise ValueError("duration and frame count must be positive")
        if minimum_boundary_frames is not None and minimum_boundary_frames <= 0:
            raise ValueError("minimum boundary frame count must be positive")
        if coverage_min_frames < 0:
            raise ValueError("coverage_min_frames must be non-negative")
        self.target_id, self.name = target_id, name
        self.device_type, self.device_id = device_type, device_id
        self.center_duration_ms = center_duration_ms
        self.boundary_duration_ms = boundary_duration_ms
        self.duration_ms = center_duration_ms + boundary_duration_ms
        self.minimum_valid_frames = minimum_valid_frames
        self.minimum_boundary_frames = minimum_boundary_frames or minimum_valid_frames
        self.minimum_confidence, self.maximum_jump_deg = minimum_confidence, maximum_jump_deg
        self.config = config
        self.phase = RegistrationPhase.CENTER
        self.started_at_ms: int | None = None
        self.phase_started_at_ms: int | None = None
        self._center_samples: list[tuple[float, float]] = []
        self._boundary_samples: list[tuple[float, float]] = []
        self._center_face_scales: list[float] = []
        self._center_feature_samples: list[TargetFeatureSample] = []
        self._boundary_feature_samples: list[TargetFeatureSample] = []
        self.total_frames_seen = 0
        self.rejected_tracking_lost = 0
        self.rejected_closed_eyes = 0
        self.rejected_low_confidence = 0
        self.rejected_jump = 0
        self.last_rejection_reason: str | None = None
        # coverage_min_frames=0이면 기존 시간제 등록 그대로다(하위 호환).
        self.coverage: PoseCoverageTracker | None = (
            PoseCoverageTracker(config, coverage_min_frames) if coverage_min_frames > 0 else None
        )
        self._raw_sample_dir = Path(raw_sample_dir) if raw_sample_dir is not None else None
        self.requires_nod_gate = requires_nod_gate

    @property
    def valid_frame_count(self) -> int:
        return self.center_valid_frame_count + self.boundary_valid_frame_count

    @property
    def center_valid_frame_count(self) -> int:
        return len(self._center_samples)

    @property
    def boundary_valid_frame_count(self) -> int:
        return len(self._boundary_samples)

    @property
    def center_yaw_pitch(self) -> tuple[float, float] | None:
        # The precision boundary is the authoritative object extent. Its robust
        # midpoint is therefore the target center. Phase-1 samples exist to
        # learn pose/distance/face-location context, not a regression label.
        samples = (
            self._boundary_samples
            if self.boundary_valid_frame_count >= self.minimum_boundary_frames
            else self._center_samples
        )
        if len(samples) < self.minimum_valid_frames:
            return None
        center = np.median(np.asarray(samples, dtype=np.float64), axis=0)
        return float(center[0]), float(center[1])

    @property
    def feature_samples(self) -> tuple[TargetFeatureSample, ...]:
        """Center and boundary evidence used by the target classifier."""
        return tuple((*self._center_feature_samples, *self._boundary_feature_samples))

    def phase_elapsed_ms(self, timestamp_ms: int) -> int:
        if self.phase_started_at_ms is None:
            return 0
        return max(0, timestamp_ms - self.phase_started_at_ms)

    def phase_duration_ms(self) -> int:
        if self.phase == RegistrationPhase.CENTER:
            return self.center_duration_ms
        if self.phase == RegistrationPhase.BOUNDARY:
            return self.boundary_duration_ms
        return 0

    def phase_progress(self, timestamp_ms: int) -> float:
        if self.phase == RegistrationPhase.COMPLETE:
            return 1.0
        duration = self.phase_duration_ms()
        time_progress = min(1.0, self.phase_elapsed_ms(timestamp_ms) / duration)
        if self.phase == RegistrationPhase.CENTER:
            if self.coverage is not None:
                # 조건 충족식 등록: 진행률은 시간이 아니라 coverage 충족률이다.
                return self.coverage.progress()
            frame_progress = min(1.0, self.center_valid_frame_count / self.minimum_valid_frames)
        else:
            frame_progress = min(
                1.0, self.boundary_valid_frame_count / self.minimum_boundary_frames
            )
        return min(time_progress, frame_progress)

    def add(
        self,
        gaze: SmoothedGaze | None,
        confidence: float,
        *,
        eyes_open: bool = True,
        face_scale: float | None = None,
        feature_sample: TargetFeatureSample | None = None,
    ) -> bool:
        self.total_frames_seen += 1
        if gaze is not None:
            self._advance_phase_if_ready(gaze.timestamp_ms)
        if self.phase == RegistrationPhase.COMPLETE:
            return False
        if gaze is None:
            self.rejected_tracking_lost += 1
            self.last_rejection_reason = "face/gaze tracking unavailable"
            return False
        if not eyes_open:
            self.rejected_closed_eyes += 1
            self.last_rejection_reason = "eyes classified closed"
            return False
        if confidence < self.minimum_confidence:
            self.rejected_low_confidence += 1
            self.last_rejection_reason = (
                f"tracking confidence {confidence:.2f} < {self.minimum_confidence:.2f}"
            )
            return False
        if self.started_at_ms is None:
            self.started_at_ms = gaze.timestamp_ms
            self.phase_started_at_ms = gaze.timestamp_ms
        accepted_phase = self.phase
        yaw, pitch = direction_to_yaw_pitch(gaze.direction)
        samples = (
            self._center_samples
            if accepted_phase == RegistrationPhase.CENTER
            else self._boundary_samples
        )
        if samples:
            previous_yaw, previous_pitch = samples[-1]
            if math.hypot(yaw - previous_yaw, pitch - previous_pitch) > self.maximum_jump_deg:
                self.rejected_jump += 1
                self.last_rejection_reason = "gaze jumped too far between frames"
                return False
        samples.append((yaw, pitch))
        self.last_rejection_reason = None
        if accepted_phase == RegistrationPhase.CENTER:
            if face_scale is not None and math.isfinite(face_scale) and face_scale > 0.0:
                self._center_face_scales.append(face_scale)
            if feature_sample is not None:
                self._center_feature_samples.append(feature_sample)
                if self.coverage is not None:
                    self.coverage.add(feature_sample)
        elif feature_sample is not None:
            self._boundary_feature_samples.append(feature_sample)
        self._advance_phase_if_ready(gaze.timestamp_ms)
        return True

    def is_elapsed(self, timestamp_ms: int) -> bool:
        self._advance_phase_if_ready(timestamp_ms)
        return self.phase == RegistrationPhase.COMPLETE

    def start_boundary(self, timestamp_ms: int) -> None:
        """Advance explicitly after enough center frames; useful for UI/tests."""
        if self.center_valid_frame_count < self.minimum_valid_frames:
            raise ValueError(
                "not enough valid center frames: "
                f"{self.center_valid_frame_count}/{self.minimum_valid_frames}"
            )
        if self.coverage is not None and not self.coverage.complete():
            raise ValueError(
                "pose coverage incomplete: " + ", ".join(self.coverage.missing_labels())
            )
        self.phase = RegistrationPhase.BOUNDARY
        self.phase_started_at_ms = timestamp_ms

    def _advance_phase_if_ready(self, timestamp_ms: int) -> None:
        if self.phase_started_at_ms is None:
            return
        elapsed_ms = self.phase_elapsed_ms(timestamp_ms)
        if self.phase == RegistrationPhase.CENTER and self.coverage is not None:
            # 조건 충족식: 시간과 무관하게 coverage가 다 차면 2단계로 넘어간다.
            if (
                self.coverage.complete()
                and self.center_valid_frame_count >= self.minimum_valid_frames
            ):
                self.start_boundary(timestamp_ms)
            return
        if (
            self.phase == RegistrationPhase.CENTER
            and elapsed_ms >= self.center_duration_ms
            and self.center_valid_frame_count >= self.minimum_valid_frames
        ):
            self.start_boundary(timestamp_ms)
            return
        if (
            self.phase == RegistrationPhase.BOUNDARY
            and elapsed_ms >= self.boundary_duration_ms
            and self.boundary_valid_frame_count >= self.minimum_boundary_frames
        ):
            self.phase = RegistrationPhase.COMPLETE
            self.phase_started_at_ms = timestamp_ms

    def finalize(self) -> TargetRecord:
        if self.center_valid_frame_count < self.minimum_valid_frames:
            raise ValueError(
                "not enough valid center frames: "
                f"{self.center_valid_frame_count}/{self.minimum_valid_frames}"
            )
        if self.coverage is not None and not self.coverage.complete():
            # 불완전한 coverage는 저장하지 않는다 — 오른쪽 bin 2개만 저장된
            # 등록(2026-07-22)처럼 편향된 보정표를 만들지 않기 위함이다.
            raise ValueError(
                "pose coverage incomplete: " + ", ".join(self.coverage.missing_labels())
            )
        if self.boundary_valid_frame_count < self.minimum_boundary_frames:
            raise ValueError(
                "not enough valid boundary frames: "
                f"{self.boundary_valid_frame_count}/{self.minimum_boundary_frames}"
            )
        boundary_samples = np.asarray(self._boundary_samples, dtype=np.float64)
        center_yaw_pitch = self.center_yaw_pitch
        assert center_yaw_pitch is not None
        center = np.asarray(center_yaw_pitch, dtype=np.float64)
        deviations = np.abs(boundary_samples - center)
        spread = np.percentile(deviations, 90, axis=0)
        spread_yaw = min(
            self.config.registration_max_spread_deg,
            max(self.config.registration_min_spread_deg, float(spread[0])),
        )
        spread_pitch = min(
            self.config.registration_max_spread_deg,
            max(self.config.registration_min_spread_deg, float(spread[1])),
        )
        # Boundary rays touch different surface points, so triangulating them as
        # if they met at one 3D point would fabricate a target position. The
        # deterministic profile instead uses face scale and image location.
        position_3d = None
        if self.config.require_3d_target_registration and position_3d is None:
            raise ValueError("3D registration is incompatible with boundary tracing")
        feature_profile = (
            build_feature_profile(list(self.feature_samples)).profile
            if len(self.feature_samples) >= self.minimum_valid_frames
            else None
        )
        area_profile = build_area_profile(
            self._boundary_samples,
            center_yaw_pitch=(float(center[0]), float(center[1])),
            minimum_radius_deg=self.config.registration_min_spread_deg,
            maximum_radius_deg=self.config.registration_max_area_radius_deg,
        )
        # 2단계는 고개를 고정하고 hull을 그리므로 그 head yaw가 보정의 기준
        # 자세다 — 이 자세에서 pose 보정이 0이 되도록 정규화된다.
        reference_head_yaw = (
            float(
                np.median(
                    np.asarray(
                        [sample.head_yaw for sample in self._boundary_feature_samples],
                        dtype=np.float64,
                    )
                )
            )
            if self._boundary_feature_samples
            else None
        )
        pose_correction = build_pose_correction(
            list(self._center_feature_samples),
            center_yaw_pitch=(float(center[0]), float(center[1])),
            reference_head_yaw_deg=reference_head_yaw,
            bin_edges_deg=self.config.pose_correction_bin_edges_deg,
            minimum_bin_samples=self.config.pose_correction_min_bin_samples,
            maximum_offset_deg=self.config.pose_correction_max_offset_deg,
        )
        self._export_raw_samples((float(center[0]), float(center[1])), reference_head_yaw)
        return TargetRecord(
            target_id=self.target_id,
            name=self.name,
            device_type=self.device_type,
            direction=TargetDirection(float(center[0]), float(center[1])),
            spread=TargetSpread(spread_yaw, spread_pitch),
            device_id=self.device_id,
            position_3d=position_3d,
            reference_face_scale=(
                float(np.median(np.asarray(self._center_face_scales, dtype=np.float64)))
                if self._center_face_scales
                else None
            ),
            feature_profile=feature_profile,
            area_profile=area_profile,
            pose_correction=pose_correction,
            requires_nod_gate=self.requires_nod_gate,
        )

    def _export_raw_samples(
        self, center_yaw_pitch: tuple[float, float], reference_head_yaw: float | None
    ) -> None:
        """1단계 중앙 응시 원시 샘플을 JSON으로 남긴다(오프라인 A/B 학습용).

        `raw_sample_dir`가 없으면 아무것도 하지 않는다. 파일명은 세션 시작
        타임스탬프로 구분해 같은 target의 재등록 이력이 서로 덮어쓰지 않는다.
        """
        if self._raw_sample_dir is None:
            return
        self._raw_sample_dir.mkdir(parents=True, exist_ok=True)
        path = self._raw_sample_dir / (
            f"{self.target_id}_phase1_{self.started_at_ms or 0}.json"
        )
        payload = {
            "target_id": self.target_id,
            "name": self.name,
            "started_at_ms": self.started_at_ms,
            "center_yaw_pitch": list(center_yaw_pitch),
            "reference_head_yaw_deg": reference_head_yaw,
            "feature_names": [
                "gaze_yaw",
                "gaze_pitch",
                "head_yaw",
                "head_pitch",
                "head_roll",
                "face_scale",
                "face_center_x",
                "face_center_y",
            ],
            "samples": [
                [float(value) for value in sample.as_array()]
                for sample in self._center_feature_samples
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def diagnostic_summary(self) -> str:
        """Human-readable counts explaining why registration frames were rejected."""
        return (
            f"phase={self.phase}, seen={self.total_frames_seen}, "
            f"center={self.center_valid_frame_count}, boundary={self.boundary_valid_frame_count}, "
            f"center_scale={len(self._center_face_scales)}, "
            f"center_features={len(self._center_feature_samples)}, "
            f"boundary_features={len(self._boundary_feature_samples)}, "
            f"tracking_lost={self.rejected_tracking_lost}, "
            f"closed_eyes={self.rejected_closed_eyes}, "
            f"low_conf={self.rejected_low_confidence}, jump={self.rejected_jump}"
            + (
                f", coverage_missing=[{', '.join(self.coverage.missing_labels())}]"
                if self.coverage is not None and not self.coverage.complete()
                else ""
            )
        )

"""라벨된 gaze 디버깅 세션을 프레임 단위 JSONL로 기록한다.

지금까지의 디버깅은 raw 스윕 CSV만 남아 매번 합성·보정·판정을 밖에서 재현해야
했다. 이 레코더는 실제 파이프라인이 프레임마다 산출한 모든 단계(GazeSnapshot)와
사용자가 표시한 정답 라벨("지금 무엇을 보고 있는지")을 한 파일에 남겨,
`jarvis-gaze report`가 재현 계산 없이 정확도·편향을 집계할 수 있게 한다.

파일 구조 (한 줄에 JSON 객체 하나):

- 첫 줄 ``{"type": "header", ...}`` — 기록 시점의 GazeConfig 전체와 등록된
  target 스냅샷(area/보정 테이블 포함). 세션 파일 하나로 분석이 완결되도록
  한다(나중에 profiles.json이 바뀌어도 세션은 그대로 해석된다).
- 이후 ``{"type": "frame", ...}`` — 프레임별 관측/합성/판정/lock 값과 라벨.
- 마지막 ``{"type": "footer", ...}`` — 프레임 수와 라벨별 카운트.

라벨 규약: ``None``은 "라벨 없음(집계 제외)", ``"none"``은 "등록된 어떤
target도 보고 있지 않음"이라는 명시적 정답이다.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import IO, TYPE_CHECKING, Sequence

import numpy as np

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import direction_to_yaw_pitch

if TYPE_CHECKING:
    from jarvis.calibration.registry import TargetRecord
    from jarvis.monitoring.gaze_probe import GazeSnapshot

SESSION_FORMAT_VERSION = 2

#: 정답 라벨로 "아무 target도 보지 않음"을 뜻하는 예약어.
NO_TARGET_LABEL = "none"


def _yaw_pitch(direction: tuple[float, float, float] | None) -> list[float] | None:
    if direction is None:
        return None
    yaw, pitch = direction_to_yaw_pitch(np.asarray(direction, dtype=np.float64))
    return [round(float(yaw), 3), round(float(pitch), 3)]


def _round_pair(pair: tuple[float, float] | None, digits: int = 4) -> list[float] | None:
    if pair is None:
        return None
    return [round(float(pair[0]), digits), round(float(pair[1]), digits)]


def _round_triplet(
    values: tuple[float, float, float] | None,
    digits: int = 3,
) -> list[float] | None:
    if values is None:
        return None
    return [round(float(value), digits) for value in values]


def _round_optional(value: float | None, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def _face_center(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
) -> list[float] | None:
    centers = [center for center in (left, right) if center is not None]
    if not centers:
        return None
    return [
        round(sum(center[0] for center in centers) / len(centers), 4),
        round(sum(center[1] for center in centers) / len(centers), 4),
    ]


class GazeSessionRecorder:
    """GazeSnapshot 스트림을 라벨과 함께 JSONL 세션 파일로 남긴다."""

    def __init__(self) -> None:
        self._file: IO[str] | None = None
        self._path: Path | None = None
        self._frame_count = 0
        self._label_counts: Counter[str] = Counter()

    @property
    def recording(self) -> bool:
        return self._file is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(
        self,
        path: Path,
        *,
        config: GazeConfig,
        targets: Sequence["TargetRecord"],
        started_timestamp_ms: int | None = None,
    ) -> Path:
        if self.recording:
            raise ValueError("session recording already active")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._path = path
        self._frame_count = 0
        self._label_counts = Counter()
        header = {
            "type": "header",
            "version": SESSION_FORMAT_VERSION,
            "started_timestamp_ms": started_timestamp_ms,
            "config": asdict(config),
            "targets": [asdict(record) for record in targets],
        }
        self._write(header)
        return path

    def record(self, snapshot: "GazeSnapshot", label: str | None) -> None:
        """한 프레임의 파이프라인 산출 전체와 현재 정답 라벨을 기록한다."""
        if self._file is None:
            raise ValueError("session recording is not active")
        per_target: dict[str, dict[str, object]] = {}
        for area in snapshot.area_details:
            per_target.setdefault(area.device_id, {}).update(
                {
                    "area_nd": round(float(area.normalized_distance), 4),
                    "used_gaze": [
                        round(float(area.used_gaze_yaw), 3),
                        round(float(area.used_gaze_pitch), 3),
                    ],
                    "correction_applied": bool(area.correction_applied),
                }
            )
        for feature in snapshot.feature_details:
            per_target.setdefault(feature.device_id, {})[
                "feature_nd"
            ] = round(float(feature.normalized_distance), 4)

        sample = snapshot.feature_sample
        frame = {
            "type": "frame",
            "t": snapshot.timestamp_ms,
            "frame": snapshot.frame_id,
            "label": label,
            "obs": {
                "face_detected": snapshot.face_detected,
                "head": [
                    round(snapshot.head_yaw_deg, 3),
                    round(snapshot.head_pitch_deg, 3),
                    round(snapshot.head_roll_deg, 3),
                ],
                "iris_l": _round_pair(snapshot.left_iris_relative),
                "iris_r": _round_pair(snapshot.right_iris_relative),
                "eye_l": _round_pair(snapshot.left_eye_center_normalized),
                "eye_r": _round_pair(snapshot.right_eye_center_normalized),
                "face_center": _face_center(
                    snapshot.left_eye_center_normalized,
                    snapshot.right_eye_center_normalized,
                ),
                "face_scale": (
                    round(snapshot.face_scale, 5) if snapshot.face_scale is not None else None
                ),
                "eyes_open": snapshot.eyes_open,
                "eye_open_ratio": [
                    _round_optional(snapshot.left_eye_open_ratio),
                    _round_optional(snapshot.right_eye_open_ratio),
                ],
                "eye_open_baseline": [
                    _round_optional(snapshot.left_eye_open_baseline),
                    _round_optional(snapshot.right_eye_open_baseline),
                ],
                "confidence": round(snapshot.tracking_confidence, 3),
            },
            "gaze": {
                "raw": _yaw_pitch(snapshot.raw_gaze_direction),
                "raw_confidence": _round_optional(snapshot.raw_gaze_confidence, 3),
                "smoothed": _yaw_pitch(snapshot.smoothed_gaze_direction),
                "origin_mm": _round_triplet(snapshot.smoothed_gaze_origin),
                "confidence": _round_optional(snapshot.gaze_confidence, 3),
                "feature": (
                    [round(sample.gaze_yaw, 3), round(sample.gaze_pitch, 3)]
                    if sample is not None
                    else None
                ),
                "stability": (
                    round(snapshot.smoothed_stability, 3)
                    if snapshot.smoothed_stability is not None
                    else None
                ),
                "source": snapshot.gaze_source,
                "source_reason": snapshot.gaze_source_reason,
                "buffer": [snapshot.buffer_fill, snapshot.buffer_capacity],
                "motion_delta": _round_pair(snapshot.gaze_motion_delta_deg, 3),
                "velocity_deg_s": _round_pair(
                    snapshot.gaze_motion_velocity_deg_s,
                    3,
                ),
                "acceleration_deg_s2": _round_pair(
                    snapshot.gaze_motion_acceleration_deg_s2,
                    3,
                ),
                "settle_velocity_deg_s": _round_pair(
                    snapshot.gaze_settle_velocity_deg_s,
                    3,
                ),
                "settle_age_ms": snapshot.gaze_settle_age_ms,
                "motion_history_valid": snapshot.gaze_motion_history_valid,
            },
            "cls": {
                "target": snapshot.target,
                "p": round(snapshot.probability, 4),
                "p2": round(snapshot.second_best_probability, 4),
                "reject": snapshot.reject_reason,
                "confident": snapshot.is_confident,
            },
            "lock": {
                "state": snapshot.lock_state.name,
                "candidate": snapshot.candidate_device,
                "dwell_ms": snapshot.dwell_elapsed_ms,
                "dwell_required_ms": snapshot.dwell_required_ms,
                "locked": snapshot.locked_device,
                "unknown_ms": snapshot.unknown_elapsed_ms,
            },
            "targets": per_target,
        }
        self._write(frame)
        self._frame_count += 1
        self._label_counts[label if label is not None else "(unlabeled)"] += 1

    def stop(self) -> dict[str, object]:
        """기록을 닫고 요약을 반환한다."""
        if self._file is None:
            raise ValueError("session recording is not active")
        footer = {
            "type": "footer",
            "frames": self._frame_count,
            "labels": dict(self._label_counts),
        }
        self._write(footer)
        self._file.close()
        self._file = None
        return footer

    def _write(self, payload: dict[str, object]) -> None:
        assert self._file is not None
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

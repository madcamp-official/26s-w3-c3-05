"""Persist up to ten user-triggered gaze diagnostic samples as JSON."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from jarvis.gaze.direction import direction_to_yaw_pitch
from jarvis.monitoring.gaze_probe import GazeSnapshot


class GazeSampleStore:
    def __init__(self, path: Path, capacity: int = 10) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._path = path
        self._capacity = capacity
        self._samples = self._load()

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def full(self) -> bool:
        return self.count >= self.capacity

    @property
    def samples(self) -> list[dict[str, object]]:
        return [dict(sample) for sample in self._samples]

    def add(self, snapshot: GazeSnapshot) -> dict[str, object]:
        return self.add_window([snapshot], minimum_frames=1)

    def clear(self) -> None:
        self._samples.clear()
        self._save()

    def add_window(
        self, snapshots: Sequence[GazeSnapshot], *, minimum_frames: int = 5
    ) -> dict[str, object]:
        """직전 시간 구간의 유효 snapshot을 평균해 진단 샘플 하나로 저장한다."""
        if self.full:
            raise ValueError(f"gaze sample capacity reached ({self.capacity})")
        valid = [
            snapshot
            for snapshot in snapshots
            if snapshot.face_detected and snapshot.smoothed_gaze_direction is not None
        ]
        if len(valid) < minimum_frames:
            face_detected = sum(1 for snapshot in snapshots if snapshot.face_detected)
            smoothed = sum(
                1 for snapshot in snapshots if snapshot.smoothed_gaze_direction is not None
            )
            eyes_open = sum(1 for snapshot in snapshots if snapshot.eyes_open)
            raise ValueError(
                f"not enough valid gaze frames: {len(valid)}/{minimum_frames} "
                f"(history={len(snapshots)}, face={face_detected}, "
                f"smoothed={smoothed}, eyes_open={eyes_open})"
            )

        latest = valid[-1]
        directions = np.stack(
            [
                snapshot.smoothed_gaze_direction
                for snapshot in valid
                if snapshot.smoothed_gaze_direction is not None
            ]
        )
        mean_direction = directions.mean(axis=0)
        norm = float(np.linalg.norm(mean_direction))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("gaze directions cancel out to an invalid mean")
        mean_direction = mean_direction / norm
        gaze_yaw_deg, gaze_pitch_deg = direction_to_yaw_pitch(mean_direction)
        raw_directions = [
            snapshot.raw_gaze_direction for snapshot in valid if snapshot.raw_gaze_direction is not None
        ]
        raw_gaze_yaw_pitch: dict[str, float] | None = None
        if raw_directions:
            raw_mean_direction = np.asarray(raw_directions, dtype=np.float64).mean(axis=0)
            raw_norm = float(np.linalg.norm(raw_mean_direction))
            if math.isfinite(raw_norm) and raw_norm > 0.0:
                raw_mean_direction = raw_mean_direction / raw_norm
                raw_yaw_deg, raw_pitch_deg = direction_to_yaw_pitch(raw_mean_direction)
                raw_gaze_yaw_pitch = {"yaw": raw_yaw_deg, "pitch": raw_pitch_deg}
        nearest = latest.device_details[0] if latest.device_details else None
        raw_nearest = latest.raw_device_details[0] if latest.raw_device_details else None
        nearest_feature = latest.feature_details[0] if latest.feature_details else None

        def mean(values: Sequence[float]) -> float:
            return float(np.mean(np.asarray(values, dtype=np.float64)))

        def mean_pair(values: Sequence[tuple[float, float]]) -> list[float]:
            array = np.asarray(values, dtype=np.float64)
            return [float(array[:, 0].mean()), float(array[:, 1].mean())]

        eye_confidences = [snapshot.tracking_confidence for snapshot in valid]
        left_eye_centers = [
            item.left_eye_center_normalized
            for item in valid
            if item.left_eye_center_normalized is not None
        ]
        right_eye_centers = [
            item.right_eye_center_normalized
            for item in valid
            if item.right_eye_center_normalized is not None
        ]
        face_center: list[float] | None = None
        face_scale: float | None = None
        if left_eye_centers and right_eye_centers:
            left_mean = mean_pair(left_eye_centers)
            right_mean = mean_pair(right_eye_centers)
            face_center = [
                (left_mean[0] + right_mean[0]) * 0.5,
                (left_mean[1] + right_mean[1]) * 0.5,
            ]
            face_scale = math.hypot(right_mean[0] - left_mean[0], right_mean[1] - left_mean[1])
        origins = [
            item.smoothed_gaze_origin for item in valid if item.smoothed_gaze_origin is not None
        ]
        sample: dict[str, object] = {
            "sample_index": self.count + 1,
            "timestamp_ms": latest.timestamp_ms,
            "frame_id": latest.frame_id,
            "window_frame_count": len(valid),
            "window_duration_ms": (
                valid[-1].timestamp_ms - valid[0].timestamp_ms
            ),
            "gaze_direction": mean_direction.tolist(),
            "gaze_yaw_pitch_deg": {
                "yaw": gaze_yaw_deg,
                "pitch": gaze_pitch_deg,
            },
            "raw_gaze_yaw_pitch_deg": raw_gaze_yaw_pitch,
            "calibration_applied": any(item.calibration_applied for item in valid),
            "calibration_model_kind": latest.calibration_model_kind,
            "gaze_velocity_deg_s": latest.gaze_motion_velocity_deg_s,
            "gaze_acceleration_deg_s2": latest.gaze_motion_acceleration_deg_s2,
            "gaze_motion_history_valid": latest.gaze_motion_history_valid,
            "personal_feature_weights": latest.personal_feature_weights,
            "gaze_confidence": mean(eye_confidences),
            "head_pose_deg": {
                "yaw": mean([item.head_yaw_deg for item in valid]),
                "pitch": mean([item.head_pitch_deg for item in valid]),
                "roll": mean([item.head_roll_deg for item in valid]),
            },
            "left_iris_relative": mean_pair(
                [item.left_iris_relative for item in valid]
            ),
            "right_iris_relative": mean_pair(
                [item.right_iris_relative for item in valid]
            ),
            "left_eye_center_normalized": (
                mean_pair(left_eye_centers) if left_eye_centers else None
            ),
            "right_eye_center_normalized": (
                mean_pair(right_eye_centers) if right_eye_centers else None
            ),
            "face_metrics": {
                "center": face_center,
                "scale": face_scale,
            },
            "head_origin": (
                [
                    mean([origin[0] for origin in origins]),
                    mean([origin[1] for origin in origins]),
                    mean([origin[2] for origin in origins]),
                ]
                if origins
                else None
            ),
            "target": latest.target,
            "target_label": latest.target_label,
            "probability": latest.probability,
            "second_best_probability": latest.second_best_probability,
            "candidate_target": latest.candidate_device,
            "candidate_target_label": latest.candidate_label,
            "dwell_elapsed_ms": latest.dwell_elapsed_ms,
            "dwell_required_ms": latest.dwell_required_ms,
            "dwell_progress": latest.dwell_progress,
            "confirmed_target": latest.locked_device,
            "confirmed_target_label": latest.locked_target_label,
            "unknown_elapsed_ms": latest.unknown_elapsed_ms,
            "unknown_required_ms": latest.unknown_required_ms,
            "reject_reason": latest.reject_reason,
            "nearest_target_range": (
                {
                    "device_id": nearest.device_id,
                    "angular_distance_deg": nearest.angular_distance_deg,
                    "allowed_radius_deg": nearest.allowed_radius_deg,
                    "normalized_distance": nearest.normalized_distance,
                    "status": nearest.range_status,
                }
                if nearest is not None
                else None
            ),
            "raw_nearest_target_range": (
                {
                    "device_id": raw_nearest.device_id,
                    "angular_distance_deg": raw_nearest.angular_distance_deg,
                    "allowed_radius_deg": raw_nearest.allowed_radius_deg,
                    "normalized_distance": raw_nearest.normalized_distance,
                    "status": raw_nearest.range_status,
                }
                if raw_nearest is not None
                else None
            ),
            "nearest_feature_profile": (
                {
                    "device_id": nearest_feature.device_id,
                    "distance": nearest_feature.distance,
                    "threshold": nearest_feature.threshold,
                    "normalized_distance": nearest_feature.normalized_distance,
                    "status": nearest_feature.range_status,
                }
                if nearest_feature is not None
                else None
            ),
            "stability": mean(
                [item.smoothed_stability or 0.0 for item in valid]
            ),
            "lock_state": str(latest.lock_state),
        }
        self._samples.append(sample)
        self._save()
        return dict(sample)

    def _load(self) -> list[dict[str, object]]:
        if not self._path.is_file():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid gaze sample file: {self._path}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"gaze sample file must contain a JSON list: {self._path}")
        return [item for item in payload if isinstance(item, dict)][: self._capacity]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._samples, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._path)


def format_gaze_sample(sample: dict[str, object]) -> str:
    """Render one persisted sample as a compact, human-readable UI row."""
    direction = sample.get("gaze_direction")
    head_pose = sample.get("head_pose_deg")
    gaze_angles = sample.get("gaze_yaw_pitch_deg")
    raw_gaze_angles = sample.get("raw_gaze_yaw_pitch_deg")
    nearest_range = sample.get("nearest_target_range")
    raw_nearest_range = sample.get("raw_nearest_target_range")
    nearest_feature = sample.get("nearest_feature_profile")
    vector = direction if isinstance(direction, list) else []
    head = head_pose if isinstance(head_pose, dict) else {}
    gaze_yaw_pitch = gaze_angles if isinstance(gaze_angles, dict) else {}
    raw_gaze_yaw_pitch = raw_gaze_angles if isinstance(raw_gaze_angles, dict) else {}
    range_detail = nearest_range if isinstance(nearest_range, dict) else None
    raw_range_detail = raw_nearest_range if isinstance(raw_nearest_range, dict) else None
    feature_detail = nearest_feature if isinstance(nearest_feature, dict) else None

    def number(value: object) -> float:
        return float(value) if isinstance(value, (int, float)) else 0.0

    x = number(vector[0]) if len(vector) > 0 else 0.0
    y = number(vector[1]) if len(vector) > 1 else 0.0
    z = number(vector[2]) if len(vector) > 2 else 0.0
    yaw = number(head.get("yaw"))
    pitch = number(head.get("pitch"))
    roll = number(head.get("roll"))
    gaze_yaw = number(gaze_yaw_pitch.get("yaw"))
    gaze_pitch = number(gaze_yaw_pitch.get("pitch"))
    raw_gaze_yaw = number(raw_gaze_yaw_pitch.get("yaw"))
    raw_gaze_pitch = number(raw_gaze_yaw_pitch.get("pitch"))
    index = sample.get("sample_index", "?")
    target = sample.get("target", "UNKNOWN")
    target_label = sample.get("target_label", target)
    confirmed_target = sample.get("confirmed_target")
    confirmed_target_label = sample.get("confirmed_target_label", confirmed_target)
    probability = number(sample.get("probability"))
    frame_count = sample.get("window_frame_count", 1)
    if target == "UNKNOWN":
        judged = "응시대상 없음"
    elif isinstance(target_label, str) and target_label != target:
        judged = f"{target_label}[{target}]"
    else:
        judged = str(target)
    if confirmed_target is None:
        confirmed = "없음"
    elif isinstance(confirmed_target_label, str) and confirmed_target_label != confirmed_target:
        confirmed = f"{confirmed_target_label}[{confirmed_target}]"
    else:
        confirmed = str(confirmed_target)
    dwell_elapsed_ms = number(sample.get("dwell_elapsed_ms"))
    dwell_required_ms = number(sample.get("dwell_required_ms"))
    row = (
        f"#{index} [{frame_count}f] gaze=({x:+.3f}, {y:+.3f}, {z:+.3f})  "
        f"raw_y/p=({raw_gaze_yaw:+.1f}, {raw_gaze_pitch:+.1f})  "
        f"final_y/p=({gaze_yaw:+.1f}, {gaze_pitch:+.1f})  "
        f"head=({yaw:+.1f}, {pitch:+.1f}, {roll:+.1f})  "
        f"실시간={judged} P={probability:.2f}  확정={confirmed} "
        f"dwell={dwell_elapsed_ms / 1000.0:.1f}/{dwell_required_ms / 1000.0:.1f}s"
    )
    if sample.get("calibration_applied"):
        row += " CAL"
    if range_detail is not None:
        device_id = range_detail.get("device_id", "--")
        distance = number(range_detail.get("angular_distance_deg"))
        radius = number(range_detail.get("allowed_radius_deg"))
        ratio = number(range_detail.get("normalized_distance"))
        status = range_detail.get("status", "--")
        row += f"  nearest={device_id} {distance:.1f}/{radius:.1f}deg x{ratio:.2f} {status}"
    if raw_range_detail is not None:
        device_id = raw_range_detail.get("device_id", "--")
        distance = number(raw_range_detail.get("angular_distance_deg"))
        radius = number(raw_range_detail.get("allowed_radius_deg"))
        ratio = number(raw_range_detail.get("normalized_distance"))
        status = raw_range_detail.get("status", "--")
        row += f"  raw_nearest={device_id} {distance:.1f}/{radius:.1f}deg x{ratio:.2f} {status}"
    if feature_detail is not None:
        device_id = feature_detail.get("device_id", "--")
        distance = number(feature_detail.get("distance"))
        threshold = number(feature_detail.get("threshold"))
        ratio = number(feature_detail.get("normalized_distance"))
        status = feature_detail.get("status", "--")
        row += f"  feature={device_id} {distance:.2f}/{threshold:.2f} x{ratio:.2f} {status}"
    reject = sample.get("reject_reason")
    if isinstance(reject, str) and reject:
        row += f"  reject={reject}"
    return row

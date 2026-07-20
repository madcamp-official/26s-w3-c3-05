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
            raise ValueError(
                f"not enough valid gaze frames: {len(valid)}/{minimum_frames}"
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
        nearest = latest.device_details[0] if latest.device_details else None

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
            "target": latest.target,
            "target_label": latest.target_label,
            "probability": latest.probability,
            "second_best_probability": latest.second_best_probability,
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
    nearest_range = sample.get("nearest_target_range")
    vector = direction if isinstance(direction, list) else []
    head = head_pose if isinstance(head_pose, dict) else {}
    gaze_yaw_pitch = gaze_angles if isinstance(gaze_angles, dict) else {}
    range_detail = nearest_range if isinstance(nearest_range, dict) else None

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
    index = sample.get("sample_index", "?")
    target = sample.get("target", "UNKNOWN")
    target_label = sample.get("target_label", target)
    probability = number(sample.get("probability"))
    frame_count = sample.get("window_frame_count", 1)
    if target == "UNKNOWN":
        judged = "응시대상 없음"
    elif isinstance(target_label, str) and target_label != target:
        judged = f"{target_label}[{target}]"
    else:
        judged = str(target)
    row = (
        f"#{index} [{frame_count}f] gaze=({x:+.3f}, {y:+.3f}, {z:+.3f})  "
        f"gaze_y/p=({gaze_yaw:+.1f}, {gaze_pitch:+.1f})  "
        f"head=({yaw:+.1f}, {pitch:+.1f}, {roll:+.1f})  "
        f"판단={judged} P={probability:.2f}"
    )
    if range_detail is not None:
        device_id = range_detail.get("device_id", "--")
        distance = number(range_detail.get("angular_distance_deg"))
        radius = number(range_detail.get("allowed_radius_deg"))
        ratio = number(range_detail.get("normalized_distance"))
        status = range_detail.get("status", "--")
        row += f"  nearest={device_id} {distance:.1f}/{radius:.1f}deg x{ratio:.2f} {status}"
    return row

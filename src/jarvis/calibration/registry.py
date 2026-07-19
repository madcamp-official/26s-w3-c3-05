"""Persistent user-managed target registry for look-to-register calibration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from jarvis.calibration.profiles import profile_from_dict
from jarvis.gaze.classifier import DeviceGazeProfile
from jarvis.gaze.direction import direction_to_yaw_pitch, yaw_pitch_to_direction


@dataclass(frozen=True, slots=True)
class TargetDirection:
    yaw: float
    pitch: float


@dataclass(frozen=True, slots=True)
class TargetSpread:
    yaw: float
    pitch: float


@dataclass(frozen=True, slots=True)
class TargetRecord:
    target_id: str
    name: str
    device_type: str
    direction: TargetDirection
    spread: TargetSpread
    device_id: str

    def __post_init__(self) -> None:
        if not self.target_id or not self.name or not self.device_type or not self.device_id:
            raise ValueError("target metadata must not be empty")
        values = (self.direction.yaw, self.direction.pitch, self.spread.yaw, self.spread.pitch)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("target direction and spread must be finite")
        if self.spread.yaw <= 0.0 or self.spread.pitch <= 0.0:
            raise ValueError("target spread must be positive")

    def to_profile(self) -> DeviceGazeProfile:
        radius_deg = max(self.spread.yaw, self.spread.pitch)
        return DeviceGazeProfile(
            device_id=self.target_id,
            mean_direction=yaw_pitch_to_direction(self.direction.yaw, self.direction.pitch),
            variance=math.radians(radius_deg) ** 2,
        )


class TargetRegistry:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._records: dict[str, TargetRecord] = {}
        self._load()

    @property
    def records(self) -> list[TargetRecord]:
        return list(self._records.values())

    def get(self, target_id: str) -> TargetRecord | None:
        return self._records.get(target_id)

    def upsert(self, record: TargetRecord) -> None:
        self._records[record.target_id] = record
        self._save()

    def rename(self, target_id: str, name: str) -> TargetRecord:
        current = self._records[target_id]
        updated = TargetRecord(
            target_id=current.target_id,
            name=name,
            device_type=current.device_type,
            direction=current.direction,
            spread=current.spread,
            device_id=current.device_id,
        )
        self.upsert(updated)
        return updated

    def remove(self, target_id: str) -> None:
        self._records.pop(target_id, None)
        self._save()

    def nearby(
        self, yaw: float, pitch: float, minimum_distance_deg: float = 5.0
    ) -> list[TargetRecord]:
        return [
            record
            for record in self.records
            if math.hypot(record.direction.yaw - yaw, record.direction.pitch - pitch)
            < minimum_distance_deg
        ]

    def _load(self) -> None:
        if not self._path.is_file():
            return
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("target registry must contain a JSON list")
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "direction" not in item:
                migrated = self._migrate_legacy_profile(item)
                if migrated is not None:
                    self._records[migrated.target_id] = migrated
                continue
            direction = item["direction"]
            spread = item.get("spread", {"yaw": 4.0, "pitch": 4.0})
            if not isinstance(direction, dict) or not isinstance(spread, dict):
                continue
            record = TargetRecord(
                target_id=str(item["target_id"]),
                name=str(item["name"]),
                device_type=str(item["device_type"]),
                direction=TargetDirection(float(direction["yaw"]), float(direction["pitch"])),
                spread=TargetSpread(float(spread["yaw"]), float(spread["pitch"])),
                device_id=str(item["device_id"]),
            )
            self._records[record.target_id] = record

    def _migrate_legacy_profile(self, item: dict[str, object]) -> TargetRecord | None:
        profile_payload = item.get("gaze_profile")
        target_id = item.get("device_id")
        if not isinstance(target_id, str) or not isinstance(profile_payload, dict):
            return None
        try:
            profile = profile_from_dict(item)
            yaw, pitch = direction_to_yaw_pitch(profile.mean_direction)
            radius_deg = max(4.0, math.degrees(math.sqrt(profile.variance)))
        except (TypeError, ValueError):
            return None
        return TargetRecord(
            target_id=target_id,
            name=target_id,
            device_type="UNKNOWN",
            direction=TargetDirection(yaw, pitch),
            spread=TargetSpread(radius_deg, radius_deg),
            device_id=target_id,
        )

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for record in self.records:
            item = asdict(record)
            profile = record.to_profile()
            item["gaze_profile"] = {
                "mean_direction": profile.mean_direction.tolist(),
                "variance": profile.variance,
            }
            payload.append(item)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(self._path)

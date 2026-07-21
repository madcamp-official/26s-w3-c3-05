"""Persistent user-managed target registry for look-to-register calibration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jarvis.calibration.profiles import profile_from_dict
from jarvis.gaze.classifier import DeviceGazeProfile, TargetGeometry3D
from jarvis.gaze.direction import direction_to_yaw_pitch, yaw_pitch_to_direction
from jarvis.gaze.feature_profile import (
    FEATURE_DIMENSION,
    PoseCorrectionPoint,
    TargetAreaProfile,
    TargetFeatureProfile,
    TargetPoseCorrection,
)


@dataclass(frozen=True, slots=True)
class TargetDirection:
    yaw: float
    pitch: float


@dataclass(frozen=True, slots=True)
class TargetSpread:
    yaw: float
    pitch: float


@dataclass(frozen=True, slots=True)
class TargetGeometry3DRecord:
    """`TargetGeometry3D`의 JSON 영속화 형태 — 평범한 tuple/float만 쓴다.

    `TargetRegistry._save()`가 `json.dumps(asdict(record))`를 직접 호출하므로
    numpy 배열이 아닌 plain tuple이어야 한다(`TargetDirection`/`TargetSpread`와
    같은 관례).
    """

    center_mm: tuple[float, float, float]
    radius_mm: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.center_mm):
            raise ValueError("center_mm must contain three finite values")
        if not math.isfinite(self.radius_mm) or self.radius_mm <= 0.0:
            raise ValueError(f"radius_mm must be finite and positive, got {self.radius_mm}")

    def to_geometry_3d(self) -> TargetGeometry3D:
        return TargetGeometry3D(
            center_mm=np.array(self.center_mm, dtype=np.float64),
            radius_mm=self.radius_mm,
        )


@dataclass(frozen=True, slots=True)
class TargetRecord:
    target_id: str
    name: str
    device_type: str
    direction: TargetDirection
    spread: TargetSpread
    device_id: str
    position_3d: TargetGeometry3DRecord | None = None
    reference_face_scale: float | None = None
    feature_profile: TargetFeatureProfile | None = None
    area_profile: TargetAreaProfile | None = None
    pose_correction: TargetPoseCorrection | None = None
    """10초 등록 동안 모은 시선 광선의 삼각측량 결과(calibration/triangulation.py).

    품질 기준(baseline·각도 다양성·잔차)을 만족했을 때만 채워지며, 그렇지 않으면
    None으로 남아 각도 기반(direction+spread) 매칭만 쓰인다 — 지어낸 3D 위치를
    담지 않는다(documents/decisions.md).
    """

    def __post_init__(self) -> None:
        if not self.target_id or not self.name or not self.device_type or not self.device_id:
            raise ValueError("target metadata must not be empty")
        values = (self.direction.yaw, self.direction.pitch, self.spread.yaw, self.spread.pitch)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("target direction and spread must be finite")
        if self.spread.yaw <= 0.0 or self.spread.pitch <= 0.0:
            raise ValueError("target spread must be positive")
        if self.reference_face_scale is not None and (
            not math.isfinite(self.reference_face_scale) or self.reference_face_scale <= 0.0
        ):
            raise ValueError("reference_face_scale must be finite and positive")

    def to_profile(self) -> DeviceGazeProfile:
        radius_deg = max(self.spread.yaw, self.spread.pitch)
        return DeviceGazeProfile(
            device_id=self.target_id,
            mean_direction=yaw_pitch_to_direction(self.direction.yaw, self.direction.pitch),
            variance=math.radians(radius_deg) ** 2,
            reference_face_scale=self.reference_face_scale,
        )

    def to_geometry_3d(self) -> TargetGeometry3D | None:
        """3D 삼각측량이 성공했을 때만 값을 반환한다(그 외에는 None)."""
        return self.position_3d.to_geometry_3d() if self.position_3d is not None else None


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
            position_3d=current.position_3d,
            reference_face_scale=current.reference_face_scale,
            feature_profile=current.feature_profile,
            area_profile=current.area_profile,
            pose_correction=current.pose_correction,
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
                position_3d=self._parse_position_3d(item.get("position_3d")),
                reference_face_scale=self._parse_positive_float(item.get("reference_face_scale")),
                feature_profile=self._parse_feature_profile(item.get("feature_profile")),
                area_profile=self._parse_area_profile(item.get("area_profile")),
                pose_correction=self._parse_pose_correction(item.get("pose_correction")),
            )
            self._records[record.target_id] = record

    @staticmethod
    def _parse_position_3d(payload: object) -> TargetGeometry3DRecord | None:
        if not isinstance(payload, dict):
            return None
        center_mm = payload.get("center_mm")
        radius_mm = payload.get("radius_mm")
        if not isinstance(center_mm, (list, tuple)) or len(center_mm) != 3:
            return None
        if not isinstance(radius_mm, (int, float)):
            return None
        try:
            return TargetGeometry3DRecord(
                center_mm=(float(center_mm[0]), float(center_mm[1]), float(center_mm[2])),
                radius_mm=float(radius_mm),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_positive_float(payload: object) -> float | None:
        if not isinstance(payload, (int, float)):
            return None
        value = float(payload)
        return value if math.isfinite(value) and value > 0.0 else None

    @staticmethod
    def _parse_feature_profile(payload: object) -> TargetFeatureProfile | None:
        if not isinstance(payload, dict):
            return None
        mean = payload.get("mean")
        covariance = payload.get("covariance")
        sample_count = payload.get("sample_count")
        threshold = payload.get("threshold")
        if (
            not isinstance(mean, list)
            or len(mean) not in (6, FEATURE_DIMENSION)
            or not isinstance(covariance, list)
            or len(covariance) != len(mean)
            or not isinstance(sample_count, int)
            or not isinstance(threshold, (int, float))
        ):
            return None
        try:
            dimension = len(mean)
            rows = []
            for row in covariance:
                if not isinstance(row, list) or len(row) != dimension:
                    return None
                rows.append(tuple(float(value) for value in row))
            parsed_mean = [float(value) for value in mean]
            if dimension == 6:
                # Old registrations did not include face location. Preserve
                # them with a neutral center and deliberately broad variance;
                # re-registering upgrades them to a real 8D profile.
                parsed_mean.extend((0.5, 0.5))
                expanded = np.eye(FEATURE_DIMENSION, dtype=np.float64)
                expanded[:6, :6] = np.asarray(rows, dtype=np.float64)
                expanded[6, 6] = 1.0
                expanded[7, 7] = 1.0
                rows = [tuple(float(value) for value in row) for row in expanded]
            return TargetFeatureProfile(
                mean=tuple(parsed_mean),
                covariance=tuple(rows),
                sample_count=sample_count,
                threshold=float(threshold),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_pose_correction(payload: object) -> TargetPoseCorrection | None:
        """예전 JSON에는 필드가 없으므로 None으로 로드된다(하위 호환)."""
        if not isinstance(payload, dict):
            return None
        points_payload = payload.get("points")
        if not isinstance(points_payload, list) or not points_payload:
            return None
        try:
            points = tuple(
                PoseCorrectionPoint(
                    head_yaw_deg=float(point["head_yaw_deg"]),
                    offset_yaw_deg=float(point["offset_yaw_deg"]),
                    offset_pitch_deg=float(point["offset_pitch_deg"]),
                    sample_count=int(point["sample_count"]),
                )
                for point in points_payload
                if isinstance(point, dict)
            )
            reference = payload.get("reference_head_yaw_deg")
            return TargetPoseCorrection(
                points=points,
                reference_head_yaw_deg=(
                    float(reference) if isinstance(reference, (int, float)) else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _parse_area_profile(payload: object) -> TargetAreaProfile | None:
        if not isinstance(payload, dict):
            return None
        try:
            polygon_payload = payload.get("boundary_polygon", [])
            if not isinstance(polygon_payload, list):
                return None
            polygon: list[tuple[float, float]] = []
            for point in polygon_payload:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    return None
                polygon.append((float(point[0]), float(point[1])))
            return TargetAreaProfile(
                center_yaw=float(payload["center_yaw"]),
                center_pitch=float(payload["center_pitch"]),
                radius_yaw=float(payload["radius_yaw"]),
                radius_pitch=float(payload["radius_pitch"]),
                sample_count=int(payload["sample_count"]),
                boundary_polygon=tuple(polygon),
            )
        except (KeyError, TypeError, ValueError):
            return None

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

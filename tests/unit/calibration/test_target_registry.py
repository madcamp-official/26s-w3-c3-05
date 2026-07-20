from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis.calibration.registry import (
    TargetDirection,
    TargetGeometry3DRecord,
    TargetRecord,
    TargetRegistry,
    TargetSpread,
)
from jarvis.calibration.target_registration import TargetRegistrationSession
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.direction import yaw_pitch_to_direction
from jarvis.gaze.smoothing import SmoothedGaze


def _gaze(frame: int, yaw: float, pitch: float = 0.0) -> SmoothedGaze:
    return SmoothedGaze(yaw_pitch_to_direction(yaw, pitch), 1.0, frame * 50, frame)


def _ray_gaze(frame: int, direction: np.ndarray, origin: np.ndarray) -> SmoothedGaze:
    """3D 삼각측량 테스트용 — origin이 있는 SmoothedGaze."""
    return SmoothedGaze(
        direction=direction, stability=1.0, timestamp_ms=frame * 30, frame_id=frame, origin=origin
    )


def _rays_at(
    target: np.ndarray, baseline_radius_mm: float, n: int = 24
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`target`을 향하는 n개의 정확한(잡음 없는) (direction, origin) 쌍."""
    rays = []
    for i in range(n):
        angle = i / n * 2 * np.pi
        origin = np.array([baseline_radius_mm * np.cos(angle), baseline_radius_mm * np.sin(angle), 0.0])
        direction = target - origin
        direction = direction / np.linalg.norm(direction)
        rays.append((direction, origin))
    return rays


def test_registration_uses_robust_center_and_minimum_spread() -> None:
    session = TargetRegistrationSession("lamp", "조명", "LIGHT", "device-1", minimum_valid_frames=3)
    for frame, yaw in enumerate((9.8, 10.0, 10.2)):
        assert session.add(_gaze(frame, yaw), 1.0)
    record = session.finalize()
    assert record.direction.yaw == pytest.approx(10.0)
    assert record.spread.yaw == 4.0
    assert record.spread.pitch == 4.0


def test_registration_defaults_are_demo_tolerant() -> None:
    session = TargetRegistrationSession("lamp", "조명", "LIGHT", "device-1")
    assert session.duration_ms == 15_000
    assert session.minimum_valid_frames == 15
    assert session.minimum_confidence == pytest.approx(0.35)
    assert session.maximum_jump_deg == pytest.approx(18.0)


def test_registration_diagnostics_count_rejected_frames() -> None:
    session = TargetRegistrationSession(
        "lamp",
        "조명",
        "LIGHT",
        "device-1",
        minimum_valid_frames=2,
        maximum_jump_deg=12.0,
    )
    assert not session.add(None, 1.0)
    assert not session.add(_gaze(1, 0.0), 1.0, eyes_open=False)
    assert not session.add(_gaze(2, 0.0), 0.1)
    assert session.add(_gaze(3, 0.0), 1.0)
    assert not session.add(_gaze(4, 30.0), 1.0)

    assert session.diagnostic_summary() == (
        "seen=5, valid=1, tracking_lost=1, closed_eyes=1, low_conf=1, jump=1"
    )


def test_registration_rejects_jump_and_insufficient_frames() -> None:
    session = TargetRegistrationSession(
        "lamp",
        "조명",
        "LIGHT",
        "device-1",
        minimum_valid_frames=2,
        maximum_jump_deg=12.0,
    )
    assert session.add(_gaze(0, 0.0), 1.0)
    assert not session.add(_gaze(1, 30.0), 1.0)
    with pytest.raises(ValueError, match="not enough"):
        session.finalize()


def test_registration_rejects_closed_eyes() -> None:
    session = TargetRegistrationSession(
        "lamp", "desk lamp", "LIGHT", "device-1", minimum_valid_frames=1
    )
    assert not session.add(_gaze(0, 0.0), 1.0, eyes_open=False)
    with pytest.raises(ValueError, match="not enough"):
        session.finalize()


def test_registry_round_trip_and_nearby_warning_data(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    registry = TargetRegistry(path)
    record = TargetRecord(
        "lamp",
        "책상 조명",
        "LIGHT",
        TargetDirection(10.0, 4.0),
        TargetSpread(5.0, 4.0),
        "smartthings-1",
    )
    registry.upsert(record)
    loaded = TargetRegistry(path)
    assert loaded.get("lamp") == record
    assert loaded.nearby(12.0, 4.0) == [record]
    profile = record.to_profile()
    assert profile.variance == pytest.approx(np_radians_squared(5.0))


def test_registry_migrates_legacy_gaze_profile(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        """
[
  {
    "device_id": "lamp",
    "gaze_profile": {
      "mean_direction": [0.0, 0.0, 1.0],
      "variance": 0.004873878
    }
  }
]
""".strip(),
        encoding="utf-8",
    )

    registry = TargetRegistry(path)
    record = registry.get("lamp")

    assert record is not None
    assert record.direction.yaw == pytest.approx(0.0)
    assert record.direction.pitch == pytest.approx(0.0)
    assert record.spread.yaw == pytest.approx(4.0, abs=0.1)


def np_radians_squared(degrees: float) -> float:
    import math

    return math.radians(degrees) ** 2


# --- 3D triangulation during registration (documents/decisions.md 2026-07-20) ---


def test_good_parallax_registration_populates_position_3d() -> None:
    target = np.array([50.0, -20.0, 1500.0])
    session = TargetRegistrationSession(
        "lamp", "조명", "LIGHT", "device-1", minimum_valid_frames=20
    )
    for frame, (direction, origin) in enumerate(_rays_at(target, baseline_radius_mm=150.0)):
        assert session.add(_ray_gaze(frame, direction, origin), 1.0)

    record = session.finalize()

    assert record.position_3d is not None
    np.testing.assert_allclose(record.position_3d.center_mm, target, atol=1e-3)
    assert session.triangulation_result is not None
    assert session.triangulation_result.passes_quality_gates(session.config)
    # 각도 기반 direction/spread는 3D 성공 여부와 무관하게 항상 함께 계산된다.
    assert record.direction is not None
    assert record.spread.yaw > 0.0


def test_insufficient_head_movement_falls_back_to_angular_only() -> None:
    target = np.array([50.0, -20.0, 1500.0])
    session = TargetRegistrationSession(
        "lamp", "조명", "LIGHT", "device-1", minimum_valid_frames=20
    )
    # 머리가 거의 고정된 채(반경 1mm) 등록 — 실제로는 눈만 움직인 것과 같다.
    for frame, (direction, origin) in enumerate(_rays_at(target, baseline_radius_mm=1.0)):
        assert session.add(_ray_gaze(frame, direction, origin), 1.0)

    record = session.finalize()

    assert record.position_3d is None
    assert session.triangulation_result is not None
    assert not session.triangulation_result.passes_quality_gates(session.config)
    # 각도 기반 등록은 3D 실패와 무관하게 정상적으로 완료된다.
    assert record.direction.yaw != 0.0 or record.direction.pitch != 0.0


def test_too_few_rays_skips_triangulation_attempt_entirely() -> None:
    """origin이 있는 프레임이 minimum_triangulation_frames 미만이면 삼각측량
    자체를 시도하지 않는다(진단 결과도 남기지 않는다)."""
    config = GazeConfig(minimum_triangulation_frames=50)
    session = TargetRegistrationSession(
        "lamp", "조명", "LIGHT", "device-1", minimum_valid_frames=5, config=config
    )
    target = np.array([50.0, -20.0, 1500.0])
    for frame, (direction, origin) in enumerate(_rays_at(target, baseline_radius_mm=150.0, n=10)):
        assert session.add(_ray_gaze(frame, direction, origin), 1.0)

    record = session.finalize()

    assert record.position_3d is None
    assert session.triangulation_result is None


def test_registry_persists_and_reloads_position_3d(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    registry = TargetRegistry(path)
    record = TargetRecord(
        "bulb",
        "전구",
        "LIGHT",
        TargetDirection(10.0, -5.0),
        TargetSpread(6.0, 6.0),
        "smartthings-1",
        position_3d=TargetGeometry3DRecord((300.0, -20.0, 1800.0), radius_mm=25.0),
    )
    registry.upsert(record)

    reloaded = TargetRegistry(path)
    loaded = reloaded.get("bulb")

    assert loaded is not None
    assert loaded.position_3d == record.position_3d
    geometry = loaded.to_geometry_3d()
    assert geometry is not None
    np.testing.assert_allclose(geometry.center_mm, [300.0, -20.0, 1800.0])
    assert geometry.radius_mm == pytest.approx(25.0)


def test_rename_preserves_position_3d(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    registry = TargetRegistry(path)
    record = TargetRecord(
        "bulb",
        "전구",
        "LIGHT",
        TargetDirection(10.0, -5.0),
        TargetSpread(6.0, 6.0),
        "smartthings-1",
        position_3d=TargetGeometry3DRecord((300.0, -20.0, 1800.0), radius_mm=25.0),
    )
    registry.upsert(record)

    renamed = registry.rename("bulb", "거실 전구")

    assert renamed.name == "거실 전구"
    assert renamed.position_3d == record.position_3d


def test_legacy_records_without_position_3d_still_load(tmp_path: Path) -> None:
    """position_3d 필드가 없는 예전 JSON도 그대로 불러와진다(None으로 채워짐)."""
    path = tmp_path / "targets.json"
    path.write_text(
        """
[
  {
    "target_id": "lamp",
    "name": "조명",
    "device_type": "LIGHT",
    "direction": {"yaw": 10.0, "pitch": 4.0},
    "spread": {"yaw": 5.0, "pitch": 4.0},
    "device_id": "smartthings-1"
  }
]
""".strip(),
        encoding="utf-8",
    )

    registry = TargetRegistry(path)
    record = registry.get("lamp")

    assert record is not None
    assert record.position_3d is None
    assert record.to_geometry_3d() is None

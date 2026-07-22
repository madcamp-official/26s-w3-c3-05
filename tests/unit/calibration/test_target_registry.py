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
from jarvis.calibration.target_registration import RegistrationPhase, TargetRegistrationSession
from jarvis.gaze.direction import yaw_pitch_to_direction
from jarvis.gaze.feature_profile import TargetAreaProfile, TargetFeatureProfile, TargetFeatureSample
from jarvis.gaze.smoothing import SmoothedGaze


def _gaze(frame: int, yaw: float, pitch: float = 0.0) -> SmoothedGaze:
    return SmoothedGaze(yaw_pitch_to_direction(yaw, pitch), 1.0, frame * 50, frame)


def _add_boundary_samples(
    session: TargetRegistrationSession,
    count: int,
    *,
    center_yaw: float = 0.0,
    center_pitch: float = 0.0,
    start_frame: int = 1000,
) -> None:
    session.start_boundary(start_frame * 50)
    for index in range(count):
        angle = index / max(1, count) * 2.0 * np.pi
        assert session.add(
            _gaze(
                start_frame + index,
                center_yaw + 2.0 * float(np.cos(angle)),
                center_pitch + 2.0 * float(np.sin(angle)),
            ),
            1.0,
        )


def test_registration_uses_robust_center_and_minimum_spread() -> None:
    session = TargetRegistrationSession("lamp", "조명", "LIGHT", "device-1", minimum_valid_frames=3)
    for frame, yaw in enumerate((9.8, 10.0, 10.2)):
        assert session.add(_gaze(frame, yaw), 1.0)
    _add_boundary_samples(session, 12, center_yaw=10.0)
    record = session.finalize()
    assert record.direction.yaw == pytest.approx(10.0)
    assert record.spread.yaw == 4.0
    assert record.spread.pitch == 4.0


def test_registration_builds_feature_profile() -> None:
    session = TargetRegistrationSession("lamp", "desk lamp", "LIGHT", "device-1", minimum_valid_frames=3)
    for frame in range(3):
        assert session.add(
            _gaze(frame, 10.0 + frame),
            1.0,
            face_scale=0.10 + frame * 0.01,
            feature_sample=TargetFeatureSample(
                gaze_yaw=10.0 + frame,
                gaze_pitch=5.0,
                head_yaw=frame * 2.0,
                head_pitch=10.0,
                head_roll=0.0,
                face_scale=0.10 + frame * 0.01,
            ),
        )
    session.start_boundary(1000)
    for frame in range(3):
        assert session.add(
            _gaze(20 + frame, 10.0 + frame),
            1.0,
            feature_sample=TargetFeatureSample(
                gaze_yaw=10.0 + frame,
                gaze_pitch=5.0,
                head_yaw=0.0,
                head_pitch=10.0,
                head_roll=0.0,
                face_scale=0.11,
            ),
        )

    record = session.finalize()

    assert record.feature_profile is not None
    assert record.feature_profile.sample_count == 6
    assert record.area_profile is not None
    assert record.area_profile.contains(10.0, 0.0)
    assert len(record.area_profile.boundary_polygon) >= 3
    assert record.reference_face_scale == pytest.approx(0.11)


def test_registration_defaults_are_demo_tolerant() -> None:
    session = TargetRegistrationSession("lamp", "조명", "LIGHT", "device-1")
    assert session.center_duration_ms == 20_000
    assert session.boundary_duration_ms == 16_000
    assert session.duration_ms == 36_000
    assert session.minimum_valid_frames == 30
    assert session.minimum_boundary_frames == 30
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
    assert session.last_rejection_reason == "gaze jumped too far between frames"

    assert session.diagnostic_summary() == (
        "phase=CENTER, seen=5, center=1, boundary=0, "
        "center_scale=0, center_features=0, boundary_features=0, "
        "tracking_lost=1, closed_eyes=1, low_conf=1, jump=1"
    )


def test_two_phase_registration_advances_only_after_time_and_frames() -> None:
    session = TargetRegistrationSession(
        "lamp",
        "desk lamp",
        "LIGHT",
        "device-1",
        center_duration_ms=100,
        boundary_duration_ms=100,
        minimum_valid_frames=2,
    )
    assert session.add(_gaze(0, 0.0), 1.0)
    assert session.phase == RegistrationPhase.CENTER
    # Time has elapsed but the minimum center count has not: remain in phase 1.
    assert session.add(SmoothedGaze(yaw_pitch_to_direction(0.1, 0.0), 1.0, 100, 1), 1.0)
    assert session.phase == RegistrationPhase.BOUNDARY
    assert not session.is_elapsed(250)
    assert session.add(SmoothedGaze(yaw_pitch_to_direction(-4.0, 0.0), 1.0, 250, 2), 1.0)
    assert session.add(SmoothedGaze(yaw_pitch_to_direction(4.0, 0.0), 1.0, 300, 3), 1.0)
    assert session.is_elapsed(350)
    assert session.phase == RegistrationPhase.COMPLETE


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
    assert session.last_rejection_reason == "eyes classified closed"
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
        reference_face_scale=0.12,
        feature_profile=TargetFeatureProfile(
            mean=(1.0, 2.0, 3.0, 4.0, 5.0, 0.12, 0.45, 0.55),
            covariance=tuple(
                tuple(float(value) for value in row)
                for row in np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01])
            ),
            sample_count=12,
            threshold=2.5,
        ),
        area_profile=TargetAreaProfile(
            center_yaw=10.0,
            center_pitch=4.0,
            radius_yaw=6.0,
            radius_pitch=5.0,
            sample_count=24,
            boundary_polygon=((4.0, -1.0), (16.0, -1.0), (16.0, 9.0), (4.0, 9.0)),
        ),
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
    assert renamed.registration_signature == record.registration_signature


def test_registration_signature_changes_when_same_id_moves() -> None:
    original = TargetRecord(
        "target_001",
        "모니터",
        "UNKNOWN",
        TargetDirection(2.0, 5.0),
        TargetSpread(6.0, 4.0),
        "target_001",
        reference_face_scale=0.09,
    )
    moved = TargetRecord(
        "target_001",
        "모니터",
        "UNKNOWN",
        TargetDirection(-18.0, 5.0),
        TargetSpread(6.0, 4.0),
        "target_001",
        reference_face_scale=0.09,
    )

    assert original.registration_signature != moved.registration_signature


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

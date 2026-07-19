from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.calibration.registry import (
    TargetDirection,
    TargetRecord,
    TargetRegistry,
    TargetSpread,
)
from jarvis.calibration.target_registration import TargetRegistrationSession
from jarvis.gaze.direction import yaw_pitch_to_direction
from jarvis.gaze.smoothing import SmoothedGaze


def _gaze(frame: int, yaw: float, pitch: float = 0.0) -> SmoothedGaze:
    return SmoothedGaze(yaw_pitch_to_direction(yaw, pitch), 1.0, frame * 50, frame)


def test_registration_uses_robust_center_and_minimum_spread() -> None:
    session = TargetRegistrationSession("lamp", "조명", "LIGHT", "device-1", minimum_valid_frames=3)
    for frame, yaw in enumerate((9.8, 10.0, 10.2)):
        assert session.add(_gaze(frame, yaw), 1.0)
    record = session.finalize()
    assert record.direction.yaw == pytest.approx(10.0)
    assert record.spread.yaw == 4.0
    assert record.spread.pitch == 4.0


def test_registration_rejects_jump_and_insufficient_frames() -> None:
    session = TargetRegistrationSession("lamp", "조명", "LIGHT", "device-1", minimum_valid_frames=2)
    assert session.add(_gaze(0, 0.0), 1.0)
    assert not session.add(_gaze(1, 30.0), 1.0)
    with pytest.raises(ValueError, match="not enough"):
        session.finalize()


def test_registry_round_trip_and_nearby_warning_data(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    registry = TargetRegistry(path)
    record = TargetRecord(
        "lamp", "책상 조명", "LIGHT", TargetDirection(10.0, 4.0),
        TargetSpread(5.0, 4.0), "smartthings-1",
    )
    registry.upsert(record)
    loaded = TargetRegistry(path)
    assert loaded.get("lamp") == record
    assert loaded.nearby(12.0, 4.0) == [record]
    profile = record.to_profile()
    assert profile.spread_yaw_deg == 5.0
    assert profile.spread_pitch_deg == 4.0

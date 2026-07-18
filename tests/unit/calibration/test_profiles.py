"""README 7장 등록 JSON 포맷 기준 DeviceGazeProfile 직렬화/역직렬화."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis.calibration.profiles import load_profiles, profile_from_dict, profile_to_dict, save_profiles
from jarvis.gaze.classifier import DeviceGazeProfile


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def test_profile_to_dict_matches_readme_format() -> None:
    profile = DeviceGazeProfile("room.bulb", _unit([0.12, -0.04, 0.99]), variance=0.015)
    data = profile_to_dict(profile)
    assert data["device_id"] == "room.bulb"
    assert data["gaze_profile"]["variance"] == pytest.approx(0.015)
    assert len(data["gaze_profile"]["mean_direction"]) == 3


def test_round_trip_dict() -> None:
    profile = DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.02)
    restored = profile_from_dict(profile_to_dict(profile))
    assert restored.device_id == profile.device_id
    assert restored.variance == pytest.approx(profile.variance)
    np.testing.assert_allclose(restored.mean_direction, profile.mean_direction)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    profiles = [
        DeviceGazeProfile("laptop", _unit([0.0, 0.0, 1.0]), variance=0.02),
        DeviceGazeProfile("room.bulb", _unit([0.3, -0.1, 0.95]), variance=0.03),
    ]
    target = tmp_path / "calibration" / "profiles.json"
    save_profiles(profiles, target)

    loaded = load_profiles(target)
    assert {p.device_id for p in loaded} == {"laptop", "room.bulb"}
    by_id = {p.device_id: p for p in loaded}
    np.testing.assert_allclose(
        by_id["laptop"].mean_direction, profiles[0].mean_direction, atol=1e-12
    )
    assert by_id["room.bulb"].variance == pytest.approx(profiles[1].variance)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profiles(tmp_path / "does-not-exist.json")

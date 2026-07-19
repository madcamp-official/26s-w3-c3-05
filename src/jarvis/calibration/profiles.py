"""Load/save `DeviceGazeProfile`s as JSON (README 7장 "기기 등록 방식" 포맷).

프로필은 사용자·카메라 위치별로 달라지는 로컬 상태이므로 저장 위치는 호출자가
명시한다(하드코딩된 경로에 의존하지 않는다, development-principles.md 1절 2).
로컬에 저장할 때는 `data/calibration/`처럼 `.gitignore`에 등록된 경로를 쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jarvis.gaze.classifier import DeviceGazeProfile


def profile_to_dict(profile: DeviceGazeProfile) -> dict[str, object]:
    """README 7장의 등록 JSON 포맷으로 직렬화한다."""
    return {
        "device_id": profile.device_id,
        "gaze_profile": {
            "mean_direction": profile.mean_direction.tolist(),
            "variance": profile.variance,
            "spread_yaw_deg": profile.spread_yaw_deg,
            "spread_pitch_deg": profile.spread_pitch_deg,
        },
    }


def profile_from_dict(data: dict[str, object]) -> DeviceGazeProfile:
    """README 7장의 등록 JSON 포맷에서 역직렬화한다."""
    gaze_profile = data["gaze_profile"]
    assert isinstance(gaze_profile, dict)
    mean_direction = np.array(gaze_profile["mean_direction"], dtype=np.float64)
    device_id = data.get("device_id", data.get("target_id"))
    assert isinstance(device_id, str)
    variance = gaze_profile["variance"]
    assert isinstance(variance, (int, float))
    return DeviceGazeProfile(
        device_id=device_id,
        mean_direction=mean_direction,
        variance=float(variance),
        spread_yaw_deg=(
            float(gaze_profile["spread_yaw_deg"])
            if gaze_profile.get("spread_yaw_deg") is not None else None
        ),
        spread_pitch_deg=(
            float(gaze_profile["spread_pitch_deg"])
            if gaze_profile.get("spread_pitch_deg") is not None else None
        ),
    )


def save_profiles(profiles: list[DeviceGazeProfile], path: str | Path) -> None:
    """등록된 기기 프로필 목록을 하나의 JSON 파일에 저장한다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [profile_to_dict(profile) for profile in profiles]
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_profiles(path: str | Path) -> list[DeviceGazeProfile]:
    """JSON 파일에서 기기 프로필 목록을 불러온다."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Calibration profile file not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return [profile_from_dict(entry) for entry in payload if "gaze_profile" in entry]

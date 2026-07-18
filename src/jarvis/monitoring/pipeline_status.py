"""Detect which pipeline stages are actually runnable right now.

The Pipeline tab shows each stage's real availability so nothing is faked: a
stage is ``LIVE`` only if its code and dependencies are present, ``DEGRADED`` if
present but missing config/model, ``UNAVAILABLE`` if the module isn't implemented
yet (Gesture/Fusion — dev-2), and ``ERROR`` on an unexpected import failure.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class StageState(StrEnum):
    LIVE = "LIVE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class StageStatus:
    name: str
    state: StageState
    detail: str


def _module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _capture_status() -> StageStatus:
    if _module_installed("cv2"):
        return StageStatus("Capture", StageState.LIVE, "OpenCV camera available")
    return StageStatus(
        "Capture", StageState.DEGRADED, "opencv 미설치 — pip install -e \".[ui]\""
    )


def _gaze_status(model_path: Path | None) -> StageStatus:
    if not _module_installed("mediapipe"):
        return StageStatus(
            "Gaze Targeting", StageState.DEGRADED, "mediapipe 미설치 (vision extra 필요)"
        )
    if model_path is None or not model_path.is_file():
        return StageStatus(
            "Gaze Targeting", StageState.DEGRADED, "face_landmarker.task 모델 파일 없음"
        )
    return StageStatus("Gaze Targeting", StageState.LIVE, f"model: {model_path.name}")


def _gesture_status() -> StageStatus:
    # dev-2 module: implemented only when gesture_fusion has real code.
    if _module_installed("jarvis.gesture_fusion.spotter"):
        return StageStatus("Gesture Spotter", StageState.LIVE, "gesture spotter loaded")
    return StageStatus(
        "Gesture Spotter", StageState.UNAVAILABLE, "2인 파트 미구현"
    )


def _fusion_status() -> StageStatus:
    if _module_installed("jarvis.gesture_fusion.fusion"):
        return StageStatus("Intent Fusion", StageState.LIVE, "fusion engine loaded")
    return StageStatus("Intent Fusion", StageState.UNAVAILABLE, "2인 파트 미구현")


def _protocol_status() -> StageStatus:
    if _module_installed("jarvis.runtime_protocol.protocol.engine"):
        return StageStatus(
            "Protocol / Command", StageState.LIVE, "capability·TTL·dedup 준비됨"
        )
    return StageStatus("Protocol / Command", StageState.ERROR, "protocol 모듈 로드 실패")


def _adapters_status(env: Mapping[str, str]) -> StageStatus:
    windows_ok = os.name == "nt"
    smartthings_ok = bool(env.get("SMARTTHINGS_TOKEN", "").strip())
    parts = [
        f"Windows: {'준비됨' if windows_ok else 'Windows 아님'}",
        f"SmartThings: {'토큰 있음' if smartthings_ok else 'UNCONFIGURED'}",
    ]
    state = StageState.LIVE if (windows_ok or smartthings_ok) else StageState.DEGRADED
    return StageStatus("Adapters", state, " · ".join(parts))


def detect_pipeline_status(
    env: Mapping[str, str], model_path: Path | None = None
) -> list[StageStatus]:
    """Snapshot of every stage's runnability. Pure w.r.t. the given ``env``."""
    return [
        _capture_status(),
        _gaze_status(model_path),
        _gesture_status(),
        _fusion_status(),
        _protocol_status(),
        _adapters_status(env),
    ]

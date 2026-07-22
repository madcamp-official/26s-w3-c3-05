"""녹화된 세션을 설정만 바꿔 다시 판정하는 오프라인 what-if 리플레이.

세션 파일의 헤더(config + target 프로필)와 프레임별 원시 관측값만으로 실제
파이프라인(`GazeProbe.process_observation` — 라이브와 같은 상태 스레딩)을 다시
돌린다. 재녹화 없이 "tolerance를 1.2로 올리면", "pose 보정을 끄면" 같은 질문에
라벨 기준 정확도 변화를 수치로 답하기 위한 도구다.

정직성: 리플레이는 녹화 당시 스무더/lock의 내부 상태를 이어받지 않고 세션
시작부터 다시 누적한다. 첫 스무딩 윈도(약 8프레임)의 결과는 라이브 녹화와 미세하게
다를 수 있으며, 그래서 비교는 항상 "리플레이 vs 리플레이(동일 조건)"가 아니라
집계 수준(라벨×bin 정확도)에서 해석해야 한다.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jarvis.calibration.registry import TargetRegistry
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import FaceObservation
from jarvis.gaze.session_report import SessionData
from jarvis.monitoring.gaze_probe import GazeProbe
from jarvis.monitoring.session_recorder import snapshot_frame_dict


def _tupleize(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def config_from_header(
    header: dict[str, Any], overrides: dict[str, Any] | None = None
) -> GazeConfig:
    """헤더에 저장된 config로 GazeConfig를 복원하고 overrides를 덮어쓴다.

    알 수 없는 키(예: 구버전 세션의 삭제된 필드)는 조용히 버리지 않고 에러를
    낸다 — override 오타가 no-op이 되는 것을 막기 위해서다.
    """
    stored = {key: _tupleize(value) for key, value in header["config"].items()}
    valid_fields = set(GazeConfig.__dataclass_fields__)
    unknown_stored = set(stored) - valid_fields
    stored = {key: value for key, value in stored.items() if key in valid_fields}
    if overrides:
        unknown = set(overrides) - valid_fields
        if unknown:
            raise ValueError(f"unknown GazeConfig field(s): {', '.join(sorted(unknown))}")
        stored.update(overrides)
    if unknown_stored:
        # 세션이 더 새로운 config로 녹화된 경우만 알리고 진행한다.
        print(f"note: ignoring config fields absent from this build: {sorted(unknown_stored)}")
    return GazeConfig(**stored)


def _records_from_header(header: dict[str, Any]) -> list[Any]:
    """헤더의 target 스냅샷을 TargetRecord 목록으로 복원한다.

    profiles.json과 같은 포맷이므로 TargetRegistry의 로더(레거시 마이그레이션
    포함)를 그대로 재사용한다 — 파싱 로직을 복제하지 않는다.
    """
    targets = header.get("targets", [])
    if not targets:
        return []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "session_targets.json"
        path.write_text(json.dumps(targets, ensure_ascii=False), encoding="utf-8")
        return TargetRegistry(path).records


def _observation_from_frame(frame: dict[str, Any]) -> FaceObservation:
    obs = frame["obs"]
    ratios = obs.get("eye_open_ratio") or [None, None]
    baselines = obs.get("eye_open_baseline") or [None, None]

    def pair(value: Any) -> tuple[float, float] | None:
        return (float(value[0]), float(value[1])) if value is not None else None

    confidence = float(obs["confidence"])
    iris_l = pair(obs["iris_l"]) or (0.0, 0.0)
    iris_r = pair(obs["iris_r"]) or (0.0, 0.0)
    return FaceObservation(
        timestamp_ms=int(frame["t"]),
        frame_id=int(frame["frame"]),
        left_iris_relative=iris_l,
        right_iris_relative=iris_r,
        head_yaw_deg=float(obs["head"][0]),
        head_pitch_deg=float(obs["head"][1]),
        head_roll_deg=float(obs["head"][2]),
        eye_tracking_confidence=confidence,
        face_tracking_confidence=confidence,
        face_detected=bool(obs["face_detected"]),
        left_eye_center_normalized=pair(obs.get("eye_l")),
        right_eye_center_normalized=pair(obs.get("eye_r")),
        eyes_open=bool(obs["eyes_open"]),
        left_eye_open_ratio=ratios[0],
        right_eye_open_ratio=ratios[1],
        left_eye_open_baseline=baselines[0],
        right_eye_open_baseline=baselines[1],
    )


def replay_session(
    session: SessionData,
    *,
    overrides: dict[str, Any] | None = None,
    disable_pose_correction: bool = False,
) -> SessionData:
    """세션의 원시 관측값을 새 설정으로 다시 판정한 SessionData를 반환한다."""
    config = config_from_header(session.header, overrides)
    probe = GazeProbe(model_path=None, config=config)
    for record in _records_from_header(session.header):
        probe.register_profile(
            record.to_profile(),
            geometry_3d=record.to_geometry_3d(),
            feature_profile=record.feature_profile,
            area_profile=record.area_profile,
            pose_correction=None if disable_pose_correction else record.pose_correction,
            label=record.name,
        )

    replayed_frames: list[dict[str, Any]] = []
    for frame in session.frames:
        try:
            observation = _observation_from_frame(frame)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"frame {frame.get('frame')}: cannot reconstruct observation ({error})"
            ) from None
        snapshot = probe.process_observation(observation)
        replayed_frames.append(snapshot_frame_dict(snapshot, frame.get("label")))

    header = dict(session.header)
    header["config"] = asdict(config)
    header["replay"] = {
        "overrides": overrides or {},
        "disable_pose_correction": disable_pose_correction,
    }
    return SessionData(header=header, frames=replayed_frames)


def parse_override(raw: str) -> tuple[str, Any]:
    """CLI `--set key=value`를 (key, 형변환된 값)으로 파싱한다."""
    if "=" not in raw:
        raise ValueError(f"--set expects key=value, got {raw!r}")
    key, text = raw.split("=", 1)
    key = key.strip()
    text = text.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return key, lowered == "true"
    try:
        number = float(text)
    except ValueError:
        return key, text
    if number.is_integer() and "." not in text and "e" not in lowered:
        return key, int(number)
    if not math.isfinite(number):
        raise ValueError(f"--set {key}: value must be finite")
    return key, number

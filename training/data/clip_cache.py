"""클립 단위 landmark 캐시 — Jester 오프라인 추출과 웹캠 파인튜닝 녹화가 공유하는 포맷.

`normalize_hand`가 만든 `HandObservation` 시퀀스(이미 손목 원점화·손바닥 크기
정규화·손목 평행이동 신호까지 포함, **평활화 전**)를 클립당 파일 하나로 저장한다.
`smooth_landmarks`·`include_*` 그룹 on/off·augmentation·phase 라벨링은 전부 이
캐시를 읽을 때(`training/dataset.py`)마다 다시 계산하므로, 이 값들을 조정할 때
비싼 MediaPipe 추출을 다시 돌릴 필요가 없다(단, `origin_index`·`palm_scale_*_index`·
`LANDMARK_DIMS`처럼 `normalize_hand` 자체의 좌표계를 바꾸는 변경은 재추출이
필요하다 — `documents/gesture-fusion.md` 2026-07-19 항목 참조).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS
from jarvis.gesture_fusion.landmarks import HandObservation

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CachedClip:
    """클립 하나의 프레임별 `HandObservation` 필드를 시간축으로 쌓은 배열들."""

    clip_id: str
    gesture_label: str
    landmarks: FloatArray  # (T, HAND_LANDMARK_COUNT, LANDMARK_DIMS)
    wrist_position: FloatArray  # (T, LANDMARK_DIMS)
    palm_scale: FloatArray  # (T,)
    detection_confidence: FloatArray  # (T,)
    handedness_score: FloatArray  # (T,)
    hand_detected: BoolArray  # (T,)
    timestamp_ms: IntArray  # (T,)

    def __post_init__(self) -> None:
        length = self.landmarks.shape[0]
        if self.landmarks.shape != (length, HAND_LANDMARK_COUNT, LANDMARK_DIMS):
            raise ValueError(f"landmarks must have shape (T, {HAND_LANDMARK_COUNT}, {LANDMARK_DIMS})")
        for name, arr in (
            ("wrist_position", self.wrist_position),
            ("palm_scale", self.palm_scale),
            ("detection_confidence", self.detection_confidence),
            ("handedness_score", self.handedness_score),
            ("hand_detected", self.hand_detected),
            ("timestamp_ms", self.timestamp_ms),
        ):
            if arr.shape[0] != length:
                raise ValueError(f"{name} must have {length} rows to match landmarks, got {arr.shape[0]}")
        if length == 0:
            raise ValueError("a cached clip must contain at least one frame")

    def __len__(self) -> int:
        return int(self.landmarks.shape[0])


def observations_to_cached_clip(
    observations: list[HandObservation], gesture_label: str, clip_id: str
) -> CachedClip:
    """프레임별 `HandObservation` 리스트를 시간축으로 쌓은 `CachedClip`으로 변환한다."""
    if not observations:
        raise ValueError("cannot cache an empty observation sequence")
    return CachedClip(
        clip_id=clip_id,
        gesture_label=gesture_label,
        landmarks=np.stack([o.landmarks for o in observations]).astype(np.float64),
        wrist_position=np.stack([o.wrist_position for o in observations]).astype(np.float64),
        palm_scale=np.array([o.palm_scale for o in observations], dtype=np.float64),
        detection_confidence=np.array([o.detection_confidence for o in observations], dtype=np.float64),
        handedness_score=np.array([o.handedness_score for o in observations], dtype=np.float64),
        hand_detected=np.array([o.hand_detected for o in observations], dtype=np.bool_),
        timestamp_ms=np.array([o.timestamp_ms for o in observations], dtype=np.int64),
    )


def save_clip(path: Path, clip: CachedClip) -> None:
    """클립을 `.npz`로 저장한다. 중단된 실행 재개 시 원자성을 위해 임시파일에 쓰고 교체한다.

    (gaze_samples.py와 같은 atomic-write 패턴: 임시 파일에 다 쓴 뒤에만 `replace`로
    최종 경로에 놓는다 — 추출 도중 프로세스가 죽어도 반쪽짜리 캐시 파일이 남지 않는다.)
    파일 객체를 직접 넘겨 numpy가 경로 문자열에 `.npz`를 임의로 덧붙이는 것을 피한다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as fh:
        np.savez_compressed(
            fh,
            clip_id=clip.clip_id,
            gesture_label=clip.gesture_label,
            landmarks=clip.landmarks,
            wrist_position=clip.wrist_position,
            palm_scale=clip.palm_scale,
            detection_confidence=clip.detection_confidence,
            handedness_score=clip.handedness_score,
            hand_detected=clip.hand_detected,
            timestamp_ms=clip.timestamp_ms,
        )
    tmp_path.replace(path)


def load_clip(path: Path) -> CachedClip:
    """`save_clip`이 저장한 `.npz`를 읽는다."""
    with np.load(path, allow_pickle=False) as data:
        return CachedClip(
            clip_id=str(data["clip_id"]),
            gesture_label=str(data["gesture_label"]),
            landmarks=data["landmarks"].astype(np.float64),
            wrist_position=data["wrist_position"].astype(np.float64),
            palm_scale=data["palm_scale"].astype(np.float64),
            detection_confidence=data["detection_confidence"].astype(np.float64),
            handedness_score=data["handedness_score"].astype(np.float64),
            hand_detected=data["hand_detected"].astype(np.bool_),
            timestamp_ms=data["timestamp_ms"].astype(np.int64),
        )

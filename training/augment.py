"""학습 데이터 augmentation — 원시(정규화된) landmark 시퀀스에 적용하는 순수 변환.

랜드마크 시퀀스라 이미지 augmentation(색상·조명 등)은 의미가 없다. 여기 두 변환만
쓴다(학습 파이프라인 인터뷰 결정):

1. **좌우반전 + 라벨 스왑**: x좌표를 뒤집으면 오른손 동작이 왼손 동작처럼 보이는데,
   이때 카메라 기준 시계방향이 거울상에서는 반시계방향이 되므로 라벨도 같이
   바꿔야 한다(`training.data.jester_labels.swap_label_for_flip`). `normalize_hand`가
   평행이동(원점화)·균등 스케일만 적용하고 회전은 안 하므로, 이미 정규화된 좌표에
   x반전을 적용하는 것은 원시 좌표에 반전 후 정규화한 것과 수학적으로 동일하다
   (아핀 변환이 반전과 교환 가능).
2. **시간축 속도 변형**: 프레임을 선형보간으로 리샘플해 같은 제스처를 다른 속도로
   수행한 것처럼 만든다. timestamp를 클립 자체의 평균 프레임 간격으로 다시 채워
   (하드코딩된 fps 없음 — Jester 12fps든 웹캠 다른 fps든 동일 코드로 동작),
   뒤따르는 causal 미분(velocity/acceleration)이 실제로 다른 속도의 동작처럼 스케일되게 한다.

둘 다 순수 함수다 — 확률·난수는 호출자(`training/dataset.py`)가 결정해 넘긴다.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt

from training.data.clip_cache import CachedClip
from training.data.jester_labels import swap_label_for_flip

FloatArray = npt.NDArray[np.float64]


def flip_landmarks(clip: CachedClip) -> CachedClip:
    """x좌표를 반전하고 라벨을 대응 라벨로 스왑한 새 `CachedClip`을 반환한다."""
    flipped_landmarks = clip.landmarks.copy()
    flipped_landmarks[..., 0] *= -1.0
    flipped_wrist = clip.wrist_position.copy()
    flipped_wrist[..., 0] *= -1.0
    return replace(
        clip,
        landmarks=flipped_landmarks,
        wrist_position=flipped_wrist,
        gesture_label=swap_label_for_flip(clip.gesture_label),
    )


def _resample(values: FloatArray, old_index: FloatArray, new_index: FloatArray) -> FloatArray:
    """(T, ...) 배열을 시간축 기준으로 선형보간 리샘플한다(맨 앞 축만 바뀜)."""
    original_length = values.shape[0]
    trailing_shape = values.shape[1:]
    flat = values.reshape(original_length, -1)
    resampled = np.empty((new_index.shape[0], flat.shape[1]), dtype=np.float64)
    for col in range(flat.shape[1]):
        resampled[:, col] = np.interp(new_index, old_index, flat[:, col])
    return resampled.reshape((new_index.shape[0],) + trailing_shape)


def time_warp(clip: CachedClip, rate: float) -> CachedClip:
    """시간축을 `rate`배로 리샘플한다. `rate`>1=빠르게(프레임 감소), <1=느리게(프레임 증가).

    프레임이 2개 미만으로 줄어들지 않도록 클램프한다(속도 계산이 성립하려면
    최소 2프레임 필요).
    """
    if rate <= 0.0:
        raise ValueError("rate must be positive")
    original_length = len(clip)
    new_length = max(2, round(original_length / rate))
    if new_length == original_length:
        return clip

    old_index = np.linspace(0.0, original_length - 1, num=original_length)
    new_index = np.linspace(0.0, original_length - 1, num=new_length)

    if original_length > 1:
        avg_interval_ms = float(np.diff(clip.timestamp_ms).mean())
    else:
        avg_interval_ms = 1.0
    new_timestamps = clip.timestamp_ms[0] + np.arange(new_length, dtype=np.float64) * avg_interval_ms

    return replace(
        clip,
        landmarks=_resample(clip.landmarks, old_index, new_index),
        wrist_position=_resample(clip.wrist_position, old_index, new_index),
        palm_scale=_resample(clip.palm_scale.reshape(-1, 1), old_index, new_index).reshape(-1),
        detection_confidence=_resample(
            clip.detection_confidence.reshape(-1, 1), old_index, new_index
        ).reshape(-1),
        handedness_score=_resample(
            clip.handedness_score.reshape(-1, 1), old_index, new_index
        ).reshape(-1),
        # 캐시된 클립은 손 미검출 프레임이 없는 것만 남아 있다(extract_jester.py가
        # 검출 실패 클립 전체를 제외) — 리샘플 후에도 전부 검출된 것으로 채운다.
        hand_detected=np.ones(new_length, dtype=np.bool_),
        timestamp_ms=new_timestamps.astype(np.int64),
    )

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
        # 캐시된 클립은 이제 간헐적 미검출(hand_detected=False) 프레임을 포함할 수
        # 있다(2026-07-20, extract_jester.py가 클립 전체를 버리는 대신 일정 비율까지
        # 허용하도록 바뀜) — 다른 스칼라 필드와 같은 방식(선형보간)으로 리샘플한 뒤
        # 0.5 임계로 되돌려, 미검출 구간이 리샘플 후에도 대략 같은 위치에 남게 한다.
        # 여기서 없앨 경우 dataset.py의 IGNORE_INDEX 마스킹이 깨져 "미검출=0벡터"
        # 프레임에 실제 제스처 라벨을 붙여 학습하게 된다(development-principles.md
        # 2절: 불확실한 신호를 실행 가능한 값처럼 지어내지 않는다).
        hand_detected=(
            _resample(
                clip.hand_detected.astype(np.float64).reshape(-1, 1), old_index, new_index
            ).reshape(-1)
            >= 0.5
        ),
        timestamp_ms=new_timestamps.astype(np.int64),
    )


def resample_clip_to_fps(clip: CachedClip, target_fps: float) -> CachedClip:
    """클립을 target fps로 리샘플한다 — augmentation이 아니라 **fps 정규화**다.

    `time_warp`와의 결정적 차이: `time_warp`는 새 timestamp를 **원본 평균 간격**으로
    재생성해 클립 duration 자체를 바꾼다(같은 동작을 다른 속도로 = 속도 증강). 여기서는
    **duration을 보존**하고 프레임 간격만 target fps(`1000/target_fps` ms)로 바꾼다.
    그래야 causal 미분(velocity/acceleration)이 fps에 불변으로 유지되고, TCN의 고정
    receptive field(프레임 수)가 pretrain(Jester 12fps)과 **같은 실시간 길이**를 덮는다.

    30fps 웹캠 클립을 12fps로 내리는 것이 대표 용도다(29프레임이 0.97초 대신 2.42초를
    덮게 해 느린 회전이 잘리지 않음). 실시간 인식도 같은 target fps로 다운샘플해야
    정합이 완성된다(`gesture_probe`의 frame decimation).

    프레임 수는 하드코딩한 원본 fps가 아니라 **실제 클립 duration**에서 뽑는다 — 웹캠
    fps가 흔들리기 때문이다. 2프레임 미만이거나 duration이 0이면 원본을 그대로 반환한다.
    """
    if target_fps <= 0.0:
        raise ValueError("target_fps must be positive")
    original_length = len(clip)
    if original_length < 2:
        return clip
    duration_ms = float(clip.timestamp_ms[-1] - clip.timestamp_ms[0])
    if duration_ms <= 0.0:
        return clip

    target_interval_ms = 1000.0 / target_fps
    # N 프레임은 (N-1) 간격을 이룬다: N = duration/간격 + 1.
    new_length = max(2, int(round(duration_ms / target_interval_ms)) + 1)
    if new_length == original_length:
        return clip

    old_index = np.linspace(0.0, original_length - 1, num=original_length)
    new_index = np.linspace(0.0, original_length - 1, num=new_length)
    # timestamp를 **target 간격**으로 균일 재생성(Jester 12fps 격자와 동일) — 이 한 줄이
    # time_warp(원본 간격 재사용)와 갈리는 지점이다.
    new_timestamps = (
        clip.timestamp_ms[0] + np.arange(new_length, dtype=np.float64) * target_interval_ms
    )

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
        hand_detected=(
            _resample(
                clip.hand_detected.astype(np.float64).reshape(-1, 1), old_index, new_index
            ).reshape(-1)
            >= 0.5
        ),
        timestamp_ms=new_timestamps.astype(np.int64),
    )

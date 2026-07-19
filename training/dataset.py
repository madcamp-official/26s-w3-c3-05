"""캐시된 클립 → PyTorch Dataset. Feature 조립·augmentation을 `__getitem__`마다 수행한다.

`jarvis.gesture_fusion`의 실제 `normalize_hand`(이미 캐시에 반영됨)·`HandFeatureExtractor`
(여기서 재생)를 그대로 써서, 학습 입력이 추론 경로와 항상 같은 전처리를 거치게 한다
(development-principles.md 7.3). 이 모듈은 `jarvis.gesture_fusion` 패키지에서 torch를
직접 import하는 유일한 학습 모듈이다(model.py와 같은 격리 원칙).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - only hit without the `ml`/`training` extra
    raise ImportError(
        "torch is required for training.dataset; install with `pip install -e '.[training]'`"
    ) from exc

from jarvis.contracts.messages import GesturePhase
from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, GestureConfig
from jarvis.gesture_fusion.features import HandFeatureExtractor
from jarvis.gesture_fusion.landmarks import HandObservation
from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS, PHASE_LABELS
from training.augment import flip_landmarks, time_warp
from training.config import DEFAULT_TRAINING_CONFIG, TrainingConfig
from training.data.clip_cache import CachedClip, load_clip

from training.phase_labels import label_phases

FloatArray = npt.NDArray[np.float64]

# torch.nn.CrossEntropyLoss의 기본 ignore_index와 맞춘다 — 배치 패딩 프레임을 별도
# 마스크 텐서 없이 loss 계산에서 자동으로 빼기 위함(training/losses.py 참조).
IGNORE_INDEX = -100

GESTURE_LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(DEFAULT_GESTURE_LABELS)}
PHASE_TO_INDEX: dict[GesturePhase, int] = {phase: i for i, phase in enumerate(PHASE_LABELS)}


def _cached_clip_to_observations(clip: CachedClip) -> list[HandObservation]:
    """캐시된 배열들을 프레임별 `HandObservation`으로 되돌린다(feature 조립 입력용).

    캐시된 클립은 손 미검출 프레임이 없는 것만 남아 있으므로(추출 시 필터링,
    `training/extract/extract_jester.py`) `hand_detected`는 항상 True다.
    """
    return [
        HandObservation(
            timestamp_ms=int(clip.timestamp_ms[i]),
            frame_id=i,
            landmarks=clip.landmarks[i],
            handedness="",  # feature 조립에 쓰이지 않아 캐시에 담지 않음
            palm_scale=float(clip.palm_scale[i]),
            detection_confidence=float(clip.detection_confidence[i]),
            handedness_score=float(clip.handedness_score[i]),
            hand_detected=bool(clip.hand_detected[i]),
            wrist_position=clip.wrist_position[i],
        )
        for i in range(len(clip))
    ]


def assemble_features(
    clip: CachedClip, config: GestureConfig = DEFAULT_GESTURE_CONFIG
) -> FloatArray:
    """캐시된 클립을 실제 `HandFeatureExtractor`로 재생해 (T, feature_dim) feature를 만든다."""
    extractor = HandFeatureExtractor(config)
    vectors = [extractor.push(obs).vector for obs in _cached_clip_to_observations(clip)]
    return np.stack(vectors).astype(np.float64)


class ClipDataset(Dataset):  # type: ignore[type-arg]
    """캐시된 클립 디렉토리(`*.npz`, 하위 폴더 포함) → (feature, gesture_target, phase_target).

    Jester(`cache/jester/{train,validation}/`)와 웹캠 파인튜닝(`cache/webcam/<person_id>/`)이
    같은 캐시 포맷(`clip_cache.CachedClip`)을 쓰므로 이 클래스 하나로 두 단계 모두 처리한다.
    `roots`에 여러 경로를 주면(예: 파인튜닝의 사람 단위 split — 특정 person_id 폴더들만
    골라 train/val을 나눔) 모두 합쳐 하나의 데이터셋으로 다룬다.
    """

    def __init__(
        self,
        roots: Path | list[Path],
        *,
        gesture_config: GestureConfig = DEFAULT_GESTURE_CONFIG,
        training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        root_list = [roots] if isinstance(roots, Path) else list(roots)
        self._paths = sorted(p for root in root_list for p in root.glob("**/*.npz"))
        if not self._paths:
            raise ValueError(f"no cached clips found under {root_list}")
        self._gesture_config = gesture_config
        self._training_config = training_config
        self._augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._paths)

    def gesture_labels(self) -> list[str]:
        """전체 클립의 라벨만 모은다(feature 조립·augmentation 없이) — 클래스 가중치 계산용."""
        return [load_clip(path).gesture_label for path in self._paths]

    def __getitem__(self, index: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        clip = load_clip(self._paths[index])
        if self._augment:
            cfg = self._training_config
            if self._rng.random() < cfg.flip_probability:
                clip = flip_landmarks(clip)
            if self._rng.random() < cfg.time_warp_probability:
                rate = float(self._rng.uniform(*cfg.time_warp_rate_range))
                clip = time_warp(clip, rate)

        features = assemble_features(clip, self._gesture_config)
        phases = label_phases(
            len(clip),
            clip.gesture_label,
            onset_fraction=self._training_config.onset_fraction,
            ending_fraction=self._training_config.ending_fraction,
        )
        gesture_index = GESTURE_LABEL_TO_INDEX[clip.gesture_label]
        gesture_target = np.full(len(clip), gesture_index, dtype=np.int64)
        phase_target = np.array([PHASE_TO_INDEX[p] for p in phases], dtype=np.int64)

        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(gesture_target),
            torch.from_numpy(phase_target),
        )


def collate_fn(
    batch: list[tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """가변 길이 클립을 배치 내 최대 길이로 패딩한다.

    반환 features는 `CausalTCN.forward`가 기대하는 `(batch, feature_dim, time)` 축
    순서로 이미 transpose돼 있다. gesture/phase target은 `IGNORE_INDEX`로 패딩해
    `nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)`가 패딩 프레임을 자동으로
    loss에서 제외하게 한다(별도 boolean 마스크 텐서 불필요).
    """
    max_len = max(f.shape[0] for f, _, _ in batch)
    feature_dim = batch[0][0].shape[1]

    features = torch.zeros(len(batch), feature_dim, max_len, dtype=torch.float32)
    gesture_targets = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    phase_targets = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

    for i, (feature_seq, gesture_seq, phase_seq) in enumerate(batch):
        length = feature_seq.shape[0]
        features[i, :, :length] = feature_seq.T
        gesture_targets[i, :length] = gesture_seq
        phase_targets[i, :length] = phase_seq

    return features, gesture_targets, phase_targets

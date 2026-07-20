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

    `hand_detected`가 프레임마다 False일 수 있다(2026-07-20부터: 추출 시 클립 내
    일부 프레임만 미검출이면 전체를 버리지 않고 남긴다,
    `training/extract/extract_jester.py`의 `max_missing_frame_fraction` 참조).
    `HandFeatureExtractor.push()`가 실시간 추론과 동일하게 그 프레임을 추적 손실로
    처리(reset + 0벡터)하므로, 이 함수는 그대로 전달하면 된다 — 단 그 프레임의
    loss target을 IGNORE_INDEX로 마스킹하는 것은 `ClipDataset.__getitem__`의 몫이다.
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
        self._validate_labels()

    def _validate_labels(self) -> None:
        """캐시에 현재 라벨 집합 밖의 라벨이 있으면 즉시 실패한다.

        캐시는 추출 시점의 **파생 라벨 문자열**을 담으므로, 라벨 매핑이나
        `DEFAULT_GESTURE_LABELS`를 바꾸면 옛 라벨이 남아 있을 수 있다. 이를 학습
        도중 `KeyError`로 터뜨리는 대신(원인·해결법을 알 수 없다) 데이터셋 생성
        시점에 무엇이 어긋났고 어떻게 고치는지 함께 알린다.
        """
        unknown = {
            label
            for label in (load_clip(path).gesture_label for path in self._paths)
            if label not in GESTURE_LABEL_TO_INDEX
        }
        if unknown:
            raise ValueError(
                f"캐시에 현재 라벨 집합 밖의 gesture_label이 있다: {sorted(unknown)}. "
                f"현재 라벨: {sorted(GESTURE_LABEL_TO_INDEX)}. "
                "라벨 매핑을 바꾼 뒤 캐시를 갱신하지 않은 상태다 — "
                "`python -m training.relabel_cache`로 캐시 라벨을 현재 매핑에 맞춰라"
                "(재추출 불필요: 랜드마크는 매핑과 무관하다)."
            )

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

        # 미검출(hand_detected=False) 프레임은 0벡터 feature만 나온다 — 실제 라벨을
        # 붙여 학습하면 "신호 없음"을 특정 제스처로 가르치게 된다. 패딩 프레임과
        # 같은 방식(IGNORE_INDEX)으로 gesture/phase loss에서 제외한다(2026-07-20,
        # extract_jester.py가 클립 내 일부 미검출 프레임을 남기기 시작하면서 필요해짐).
        missing = ~clip.hand_detected
        gesture_target[missing] = IGNORE_INDEX
        phase_target[missing] = IGNORE_INDEX

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

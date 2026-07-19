"""ClipDataset·collate_fn·feature 조립(training/dataset.py)을 검증한다.

torch(`ml`/`training` extra) 필요.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402  (importorskip 뒤에 와야 함)

from jarvis.contracts.messages import GesturePhase  # noqa: E402
from jarvis.gesture_fusion.config import HAND_LANDMARK_COUNT, LANDMARK_DIMS  # noqa: E402
from jarvis.gesture_fusion.features import feature_dimension  # noqa: E402
from jarvis.gesture_fusion.landmarks import HandObservation  # noqa: E402
from training.data.clip_cache import observations_to_cached_clip, save_clip  # noqa: E402
from training.dataset import (  # noqa: E402
    GESTURE_LABEL_TO_INDEX,
    IGNORE_INDEX,
    PHASE_TO_INDEX,
    ClipDataset,
    assemble_features,
    collate_fn,
)


def _observation(frame_id: int, timestamp_ms: int) -> HandObservation:
    landmarks = np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64)
    landmarks[:, 0] = 0.1 + 0.01 * frame_id
    return HandObservation(
        timestamp_ms=timestamp_ms,
        frame_id=frame_id,
        landmarks=landmarks,
        handedness="Right",
        palm_scale=0.2,
        detection_confidence=0.9,
        handedness_score=0.9,
        hand_detected=True,
        wrist_position=np.array([0.05 + 0.01 * frame_id, -0.03], dtype=np.float64),
    )


def _write_clip(path: Path, gesture_label: str, length: int = 12) -> None:
    observations = [_observation(i, 1000 + i * 33) for i in range(length)]
    clip = observations_to_cached_clip(observations, gesture_label, path.stem)
    save_clip(path, clip)


def test_assemble_features_matches_feature_dimension() -> None:
    observations = [_observation(i, 1000 + i * 33) for i in range(10)]
    clip = observations_to_cached_clip(observations, "swipe_down", "clip-a")
    features = assemble_features(clip)
    assert features.shape == (10, feature_dimension())


def test_gesture_and_phase_index_tables_cover_all_labels() -> None:
    assert set(GESTURE_LABEL_TO_INDEX.values()) == set(range(len(GESTURE_LABEL_TO_INDEX)))
    assert set(PHASE_TO_INDEX.keys()) == {
        GesturePhase.IDLE,
        GesturePhase.ONSET,
        GesturePhase.ACTIVE,
        GesturePhase.ENDING,
    }


def test_clip_dataset_len_and_getitem_shapes(tmp_path: Path) -> None:
    _write_clip(tmp_path / "clip-001.npz", "swipe_down", length=15)
    _write_clip(tmp_path / "clip-002.npz", "none", length=10)

    dataset = ClipDataset(tmp_path, augment=False)
    assert len(dataset) == 2

    features, gesture_target, phase_target = dataset[0]
    assert features.shape == (15, feature_dimension())
    assert gesture_target.shape == (15,)
    assert phase_target.shape == (15,)
    assert torch.all(gesture_target == gesture_target[0])  # 클립 전체가 같은 gesture 라벨


def test_clip_dataset_gesture_labels_matches_files(tmp_path: Path) -> None:
    _write_clip(tmp_path / "clip-a.npz", "swipe_up", length=8)
    _write_clip(tmp_path / "clip-b.npz", "swipe_down", length=8)
    dataset = ClipDataset(tmp_path, augment=False)
    assert sorted(dataset.gesture_labels()) == ["swipe_down", "swipe_up"]


def test_clip_dataset_accepts_multiple_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "person_a"
    root_b = tmp_path / "person_b"
    root_a.mkdir()
    root_b.mkdir()
    _write_clip(root_a / "clip-1.npz", "swipe_up")
    _write_clip(root_b / "clip-2.npz", "swipe_down")

    dataset = ClipDataset([root_a, root_b], augment=False)
    assert len(dataset) == 2


def test_clip_dataset_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ClipDataset(tmp_path, augment=False)


def test_collate_fn_pads_variable_length_clips_with_ignore_index() -> None:
    dim = 4
    short = (torch.zeros(3, dim), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long))
    long = (torch.ones(6, dim), torch.ones(6, dtype=torch.long), torch.ones(6, dtype=torch.long))

    features, gesture_targets, phase_targets = collate_fn([short, long])

    assert features.shape == (2, dim, 6)  # (batch, feature_dim, time)
    assert gesture_targets.shape == (2, 6)
    # 짧은 클립의 패딩 구간(3프레임)은 IGNORE_INDEX여야 한다.
    assert torch.all(gesture_targets[0, 3:] == IGNORE_INDEX)
    assert torch.all(phase_targets[0, 3:] == IGNORE_INDEX)
    # 실제 값이 있는 구간은 원본과 같아야 한다.
    assert torch.all(gesture_targets[0, :3] == 0)
    assert torch.all(gesture_targets[1, :6] == 1)
    torch.testing.assert_close(features[1, :, :6], long[0].T)

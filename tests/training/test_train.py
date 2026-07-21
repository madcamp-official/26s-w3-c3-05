"""training/train.py의 stage 검증 로직을 확인한다(실제 학습 없이).

`train()`는 무거운 데이터셋/torch 학습 루프를 도는 함수라 end-to-end 테스트는
비싸다 — 여기서는 데이터셋에 닿기 전에 실패해야 하는 입력 검증만 다룬다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from jarvis.gesture_fusion.config import (  # noqa: E402
    DEFAULT_GESTURE_CONFIG,
    HAND_LANDMARK_COUNT,
    LANDMARK_DIMS,
)
from jarvis.gesture_fusion.landmarks import HandObservation  # noqa: E402
from training.config import DEFAULT_TRAINING_CONFIG  # noqa: E402
from training.data.clip_cache import observations_to_cached_clip, save_clip  # noqa: E402
from training.train import _resolve_datasets, train  # noqa: E402


def _write_webcam_clip(cache_dir: Path, person: str, label: str, idx: int) -> None:
    obs = [
        HandObservation(
            timestamp_ms=1000 + i * 33,
            frame_id=i,
            landmarks=np.zeros((HAND_LANDMARK_COUNT, LANDMARK_DIMS), dtype=np.float64),
            handedness="Right",
            palm_scale=0.2,
            detection_confidence=0.9,
            handedness_score=0.9,
            hand_detected=True,
            wrist_position=np.zeros(LANDMARK_DIMS, dtype=np.float64),
        )
        for i in range(6)
    ]
    out = cache_dir / "webcam" / person
    out.mkdir(parents=True, exist_ok=True)
    save_clip(out / f"{person}-{label}-{idx:03d}.npz", observations_to_cached_clip(obs, label, f"{person}-{label}-{idx}"))


def _finetune_config(tmp_path: Path, **overrides: object) -> object:
    return replace(DEFAULT_TRAINING_CONFIG, cache_dir=tmp_path, **overrides)  # type: ignore[arg-type]


def test_finetune_pooled_when_no_persons_given(tmp_path: Path) -> None:
    """person 인자를 안 주면 pooled — 사람 폴더를 가로질러 클립 단위로 무작위 split."""
    for p in ("alice", "bob"):
        for i in range(10):
            _write_webcam_clip(tmp_path, p, "rotate_clockwise", i)
    cfg = _finetune_config(tmp_path, webcam_val_fraction=0.2)
    train_ds, val_ds, dataset_id = _resolve_datasets("finetune", cfg, DEFAULT_GESTURE_CONFIG, None, None)
    assert "pooled" in dataset_id
    # 20개 중 20% = 4개 val, 16개 train, 겹치지 않아야 한다.
    assert len(val_ds) == 4
    assert len(train_ds) == 16
    train_names = {p.name for p in train_ds._paths}  # type: ignore[attr-defined]
    val_names = {p.name for p in val_ds._paths}  # type: ignore[attr-defined]
    assert not (train_names & val_names)


def test_finetune_pooled_split_is_deterministic(tmp_path: Path) -> None:
    for i in range(10):
        _write_webcam_clip(tmp_path, "alice", "stop_sign", i)
    cfg = _finetune_config(tmp_path)
    _, val_a, _ = _resolve_datasets("finetune", cfg, DEFAULT_GESTURE_CONFIG, None, None)
    _, val_b, _ = _resolve_datasets("finetune", cfg, DEFAULT_GESTURE_CONFIG, None, None)
    assert [p.name for p in val_a._paths] == [p.name for p in val_b._paths]  # type: ignore[attr-defined]


def test_finetune_person_split_still_works(tmp_path: Path) -> None:
    for i in range(5):
        _write_webcam_clip(tmp_path, "alice", "stop_sign", i)
        _write_webcam_clip(tmp_path, "bob", "stop_sign", i)
    cfg = _finetune_config(tmp_path)
    train_ds, val_ds, dataset_id = _resolve_datasets(
        "finetune", cfg, DEFAULT_GESTURE_CONFIG, ["alice"], ["bob"]
    )
    assert "person-split" in dataset_id
    assert len(train_ds) == 5 and len(val_ds) == 5


def test_finetune_person_split_rejects_only_one_side(tmp_path: Path) -> None:
    _write_webcam_clip(tmp_path, "alice", "stop_sign", 0)
    cfg = _finetune_config(tmp_path)
    with pytest.raises(ValueError, match="둘 다 요구"):
        _resolve_datasets("finetune", cfg, DEFAULT_GESTURE_CONFIG, ["alice"], None)


def test_finetune_pooled_empty_webcam_raises(tmp_path: Path) -> None:
    cfg = _finetune_config(tmp_path)
    with pytest.raises(ValueError, match="webcam 클립이 없다"):
        _resolve_datasets("finetune", cfg, DEFAULT_GESTURE_CONFIG, None, None)


def test_finetune_without_init_from_raises_before_touching_datasets() -> None:
    """--stage finetune에 --init-from이 없으면 무작위 초기화 가중치로 조용히
    학습되고도 '파인튜닝' 체크포인트로 저장될 위험이 있다(2026-07-20 수정) —
    데이터셋을 찾기도 전에 즉시 거부해야 한다."""
    with pytest.raises(ValueError, match="init-from"):
        train(
            stage="finetune",
            init_from=None,
            train_persons=["alice"],
            val_persons=["bob"],
        )


def test_unknown_stage_raises() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        train(stage="bogus")


# --- 배경 합산 지표가 런타임 결정 규칙과 일치하는지 ---
#
# 학습 지표(torch, 배치)와 런타임(numpy, 프레임 하나)이 각각 구현돼 있다. 둘이
# 어긋나면 배포되는 규칙과 다른 기준으로 체크포인트를 고르게 되므로, 무작위 logits
# 위에서 두 경로가 프레임 단위로 완전히 같은 결정을 내는지 고정한다.


def test_collapsed_predictions_match_the_runtime_decision_rule() -> None:
    import numpy as np
    import torch

    from jarvis.gesture_fusion.model_protocol import collapse_background_probabilities
    from training.train import (
        _BACKGROUND_INDICES,
        _FOREGROUND_INDICES,
        _NUM_GESTURE_CLASSES,
        _collapsed_predictions,
    )

    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4, _NUM_GESTURE_CLASSES, 7, generator=generator)
    collapsed = _collapsed_predictions(logits).numpy()

    probs = torch.softmax(logits, dim=1).numpy()
    # 원본 index → 접은 공간 index (배경은 전부 0, 제스처는 등장 순서대로 1..N)
    to_collapsed = {index: 0 for index in _BACKGROUND_INDICES}
    for position, index in enumerate(_FOREGROUND_INDICES):
        to_collapsed[index] = position + 1

    for batch in range(probs.shape[0]):
        for frame in range(probs.shape[2]):
            chosen, _confidence, _distribution = collapse_background_probabilities(
                probs[batch, :, frame].astype(np.float64),
                _BACKGROUND_INDICES,
                _FOREGROUND_INDICES,
            )
            assert collapsed[batch, frame] == to_collapsed[chosen]


def test_collapsed_predictions_break_ties_toward_background() -> None:
    """동점 처리도 런타임과 같아야 한다 — 배경이 앞에 와서 argmax가 배경을 고른다."""
    import torch

    from training.train import (
        _BACKGROUND_INDICES,
        _FOREGROUND_INDICES,
        _NUM_GESTURE_CLASSES,
        _collapsed_predictions,
    )

    # 배경 각 클래스가 같은 logit 0을 갖게 하고(각각 확률 p, 합 N*p), 제스처 하나가
    # 정확히 그 합과 같아지도록 log(N)을 준다. 나머지는 무시할 만큼 낮춘다.
    logits = torch.full((1, _NUM_GESTURE_CLASSES, 1), -20.0)
    for index in _BACKGROUND_INDICES:
        logits[0, index, 0] = 0.0
    logits[0, _FOREGROUND_INDICES[0], 0] = float(
        torch.log(torch.tensor(float(len(_BACKGROUND_INDICES))))
    )

    assert int(_collapsed_predictions(logits)[0, 0]) == 0

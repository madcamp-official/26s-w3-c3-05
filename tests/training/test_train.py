"""training/train.py의 stage 검증 로직을 확인한다(실제 학습 없이).

`train()`는 무거운 데이터셋/torch 학습 루프를 도는 함수라 end-to-end 테스트는
비싸다 — 여기서는 데이터셋에 닿기 전에 실패해야 하는 입력 검증만 다룬다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from training.train import train  # noqa: E402


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

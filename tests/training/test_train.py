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

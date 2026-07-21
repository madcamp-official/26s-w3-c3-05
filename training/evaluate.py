"""체크포인트를 로드해 검증 데이터에서 macro-F1·클래스별 F1·혼동행렬을 리포트한다.

README 13장의 Wrong Actuation Rate·Gesture Event Recall(전체 런타임 Fusion+Spotter
통합 후 `jarvis.gesture_fusion.hard_negative_mining`으로 별도 평가)과는 다른,
모델 자체의 프레임 단위 분류 성능 리포트다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - only hit without the `training` extra
    raise ImportError(
        "torch is required for training.evaluate; install with `pip install -e '.[training]'`"
    ) from exc

import numpy as np

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, GestureConfig
from jarvis.gesture_fusion.features import feature_dimension
from jarvis.gesture_fusion.model import CausalTCN, ModelConfig
from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS
from training.config import DEFAULT_TRAINING_CONFIG, TrainingConfig
from training.dataset import IGNORE_INDEX, ClipDataset, collate_fn
from training.metrics import (
    ClassificationReport,
    collapse_class_indices,
    compute_classification_report,
)

_NUM_GESTURE_CLASSES = len(DEFAULT_GESTURE_LABELS)

# 배경/전경 index는 라벨 집합에서 유도한다(train.py와 동일한 정의를 공유).
_LABEL_LAYOUT = ModelConfig(feature_dim=1)
_BACKGROUND_INDICES = _LABEL_LAYOUT.background_indices
_FOREGROUND_INDICES = _LABEL_LAYOUT.foreground_indices
_NUM_COLLAPSED_CLASSES = 1 + len(_FOREGROUND_INDICES)
_COLLAPSED_LABELS = ("background",) + tuple(
    DEFAULT_GESTURE_LABELS[i] for i in _FOREGROUND_INDICES
)
# tuple이 아니라 list로 인덱싱한다(train.py와 같은 이유 — 다차원 인덱스 오해 방지).
_BACKGROUND_SELECTOR = list(_BACKGROUND_INDICES)
_FOREGROUND_SELECTOR = list(_FOREGROUND_INDICES)


def _collapsed_predictions(gesture_logits: "torch.Tensor") -> "torch.Tensor":
    """런타임 결정 규칙과 동일하게 배경 확률을 합산해 접은 예측을 만든다(train.py와 동일)."""
    probs = torch.softmax(gesture_logits, dim=1)
    background = probs[:, _BACKGROUND_SELECTOR, :].sum(dim=1, keepdim=True)
    foreground = probs[:, _FOREGROUND_SELECTOR, :]
    return torch.cat([background, foreground], dim=1).argmax(dim=1)


def evaluate(
    checkpoint_path: Path,
    data_root: Path | list[Path],
    gesture_config: GestureConfig = DEFAULT_GESTURE_CONFIG,
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> tuple[ClassificationReport, ClassificationReport]:
    """`(원본 클래스 기준 리포트, 배경 합산 기준 리포트)`를 반환한다.

    두 번째가 런타임 결정 규칙과 같은 기준이며 모델 선택에 쓰인 지표다. 첫 번째는
    어떤 배경끼리 섞이는지 등을 보기 위한 진단용이다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = ModelConfig(feature_dim=feature_dimension(gesture_config))
    net = CausalTCN(model_config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    net.eval()

    dataset = ClipDataset(
        data_root, gesture_config=gesture_config, training_config=training_config, augment=False, seed=0
    )
    loader = DataLoader(
        dataset, batch_size=training_config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_collapsed_preds: list[np.ndarray] = []
    with torch.no_grad():
        for features, gesture_targets, _phase_targets in loader:
            gesture_logits, _phase_logits = net(features.to(device))
            all_preds.append(gesture_logits.argmax(dim=1).cpu().numpy())
            all_collapsed_preds.append(_collapsed_predictions(gesture_logits).cpu().numpy())
            all_targets.append(gesture_targets.numpy())

    flat_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    flat_targets = np.concatenate([t.reshape(-1) for t in all_targets])
    flat_collapsed_preds = np.concatenate([p.reshape(-1) for p in all_collapsed_preds])
    raw_report = compute_classification_report(
        flat_preds, flat_targets, num_classes=_NUM_GESTURE_CLASSES, ignore_index=IGNORE_INDEX
    )
    collapsed_report = compute_classification_report(
        flat_collapsed_preds,
        collapse_class_indices(
            flat_targets, _BACKGROUND_INDICES, _FOREGROUND_INDICES, IGNORE_INDEX
        ),
        num_classes=_NUM_COLLAPSED_CLASSES,
        ignore_index=IGNORE_INDEX,
    )
    return raw_report, collapsed_report


def _print_report(report: ClassificationReport, labels: tuple[str, ...]) -> None:
    print(f"macro-F1: {report.macro_f1:.4f}")
    print("클래스별 F1:")
    for cls, f1 in sorted(report.per_class_f1.items()):
        print(f"  {labels[cls]}: {f1:.4f}")
    print("혼동행렬 (행=정답, 열=예측):")
    header = "".join(f"{label[:8]:>9}" for label in labels)
    print(" " * 9 + header)
    for i, row in enumerate(report.confusion):
        row_str = "".join(f"{int(v):>9d}" for v in row)
        print(f"{labels[i][:8]:>9}{row_str}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="평가할 캐시 클립 디렉토리 (예: training/cache/jester/validation)",
    )
    args = parser.parse_args(argv)

    raw_report, collapsed_report = evaluate(args.checkpoint, args.data_root)
    # 접은 기준을 먼저 낸다 — 이것이 런타임 결정 규칙과 같고 모델 선택에 쓰인 지표다.
    print(f"=== 배경 합산 기준 ({_NUM_COLLAPSED_CLASSES}클래스, 런타임 결정 규칙과 동일) ===")
    _print_report(collapsed_report, _COLLAPSED_LABELS)
    print()
    print(f"=== 원본 클래스 기준 ({_NUM_GESTURE_CLASSES}클래스, 진단용) ===")
    _print_report(raw_report, DEFAULT_GESTURE_LABELS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

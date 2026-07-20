"""Gesture TCN 학습 루프 — `--stage pretrain`(Jester)과 `--stage finetune`(웹캠, 사람 단위
split)을 전환한다.

`jarvis.gesture_fusion.model.CausalTCN`(전체 시퀀스 dense 출력 클래스, streaming용
`CausalTCNGestureModel`이 아님 — model.py docstring이 "테스트·학습용"으로 명시)을
매 프레임 loss로 학습한다. 체크포인트는 검증 macro-F1이 개선될 때마다 저장하고
(early stopping), `ModelMetadata`는 `<checkpoint>.metadata.json` sidecar로 남긴다
(development-principles.md 7.3 — `CausalTCNGestureModel.load_weights`는 metadata를
파일이 아니라 호출자가 넘겨야 하므로, 이 sidecar를 읽어 구성한다).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:  # pragma: no cover - only hit without the `training` extra
    raise ImportError(
        "torch is required for training.train; install with `pip install -e '.[training]'`"
    ) from exc

import numpy as np

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, GestureConfig
from jarvis.gesture_fusion.features import feature_dimension
from jarvis.gesture_fusion.model import CausalTCN, ModelConfig
from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS, ModelMetadata
from training.config import DEFAULT_TRAINING_CONFIG, TrainingConfig
from training.dataset import GESTURE_LABEL_TO_INDEX, IGNORE_INDEX, ClipDataset, collate_fn
from training.losses import GesturePhaseLoss, compute_class_weights
from training.metrics import ClassificationReport, compute_classification_report

_NUM_GESTURE_CLASSES = len(DEFAULT_GESTURE_LABELS)


def _build_model(gesture_config: GestureConfig) -> tuple[CausalTCN, ModelConfig]:
    """`feature_dimension(gesture_config)`을 동적으로 읽어 모델 입력 차원을 맞춘다.

    하드코딩하지 않는다 — `GestureConfig`가 나중에 조정돼도(예: feature 그룹
    on/off) 이 스크립트를 고칠 필요가 없다.
    """
    model_config = ModelConfig(feature_dim=feature_dimension(gesture_config))
    return CausalTCN(model_config), model_config


def _run_epoch(
    net: CausalTCN,
    loss_fn: GesturePhaseLoss,
    loader: DataLoader,
    optimizer: "torch.optim.Optimizer | None",
    device: str,
) -> tuple[float, ClassificationReport]:
    """한 epoch(학습 또는 검증)을 돈다. `optimizer=None`이면 검증(gradient 없음)."""
    is_train = optimizer is not None
    net.train(is_train)

    total_loss = 0.0
    n_batches = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for features, gesture_targets, phase_targets in loader:
        features = features.to(device)
        gesture_targets = gesture_targets.to(device)
        phase_targets = phase_targets.to(device)

        with torch.set_grad_enabled(is_train):
            gesture_logits, phase_logits = net(features)
            loss, _gesture_loss, _phase_loss = loss_fn(
                gesture_logits, phase_logits, gesture_targets, phase_targets
            )
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach())
        n_batches += 1
        all_preds.append(gesture_logits.detach().argmax(dim=1).cpu().numpy())
        all_targets.append(gesture_targets.cpu().numpy())

    flat_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    flat_targets = np.concatenate([t.reshape(-1) for t in all_targets])
    report = compute_classification_report(
        flat_preds, flat_targets, num_classes=_NUM_GESTURE_CLASSES, ignore_index=IGNORE_INDEX
    )
    return total_loss / max(1, n_batches), report


def _save_checkpoint(
    net: CausalTCN,
    checkpoint_path: Path,
    metadata: ModelMetadata,
) -> None:
    """가중치는 `.pt`, 메타데이터는 `<name>.pt.metadata.json` sidecar로 저장한다.

    `CausalTCNGestureModel.load_weights(path, metadata)`는 metadata를 파일에서 읽지
    않고 호출자가 넘겨야 하므로(development-principles.md 7.3), 이 sidecar가 그
    metadata의 기록 매체다 — 로드하는 쪽이 이 JSON을 읽어 `ModelMetadata(**json)`으로
    구성해 넘기면 된다.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(net.state_dict(), tmp_path)
    tmp_path.replace(checkpoint_path)

    sidecar_path = checkpoint_path.with_name(checkpoint_path.name + ".metadata.json")
    sidecar_path.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_datasets(
    stage: str,
    training_config: TrainingConfig,
    gesture_config: GestureConfig,
    train_persons: list[str] | None,
    val_persons: list[str] | None,
) -> tuple[ClipDataset, ClipDataset, str]:
    if stage == "pretrain":
        train_root = training_config.cache_dir / "jester" / "train"
        val_root = training_config.cache_dir / "jester" / "validation"
        dataset_id = "jester-v1 (공식 train/validation split)"
    elif stage == "finetune":
        if not train_persons or not val_persons:
            raise ValueError(
                "--stage finetune requires --train-persons and --val-persons "
                "(사람 단위 split — record_webcam_clips.py가 person_id별 하위 폴더로 저장)"
            )
        if set(train_persons) & set(val_persons):
            raise ValueError("--train-persons and --val-persons must not overlap (사람 단위 분리 원칙)")
        webcam_root = training_config.cache_dir / "webcam"
        train_root = [webcam_root / p for p in train_persons]
        val_root = [webcam_root / p for p in val_persons]
        dataset_id = f"webcam-finetune (train={train_persons}, val={val_persons})"
    else:
        raise ValueError(f"unknown stage: {stage!r}, expected 'pretrain' or 'finetune'")

    train_dataset = ClipDataset(
        train_root, gesture_config=gesture_config, training_config=training_config, augment=True, seed=0
    )
    val_dataset = ClipDataset(
        val_root, gesture_config=gesture_config, training_config=training_config, augment=False, seed=1
    )
    return train_dataset, val_dataset, dataset_id


def train(
    stage: str,
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
    gesture_config: GestureConfig = DEFAULT_GESTURE_CONFIG,
    init_from: Path | None = None,
    train_persons: list[str] | None = None,
    val_persons: list[str] | None = None,
    max_epochs: int | None = None,
) -> Path:
    """`stage`를 학습하고 최선(검증 macro-F1 최고) 체크포인트 경로를 반환한다."""
    if stage == "finetune" and init_from is None:
        # 2026-07-20: 이 검사가 없으면 --init-from을 깜빡했을 때 무작위 초기화
        # 가중치로 조용히 학습된 뒤 "파인튜닝"이라는 이름으로 저장돼(체크포인트
        # 파일명·ModelMetadata 둘 다), 실제로는 Jester 사전학습을 전혀 거치지 않은
        # 모델을 파인튜닝 결과물로 오인하게 만든다(development-principles.md 1·2절:
        # 성공을 지어내지 않는다).
        raise ValueError(
            "--stage finetune requires --init-from (Jester로 사전학습한 체크포인트 경로) "
            "— 없으면 무작위 초기화 가중치에서 학습이 시작되는데도 결과물이 '파인튜닝'으로 저장된다"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net, _model_config = _build_model(gesture_config)
    net.to(device)
    if init_from is not None:
        state = torch.load(init_from, map_location=device, weights_only=True)
        net.load_state_dict(state)

    train_dataset, val_dataset, dataset_id = _resolve_datasets(
        stage, training_config, gesture_config, train_persons, val_persons
    )

    class_weights = compute_class_weights(
        [GESTURE_LABEL_TO_INDEX[label] for label in train_dataset.gesture_labels()],
        num_classes=_NUM_GESTURE_CLASSES,
    ).to(device)
    loss_fn = GesturePhaseLoss(class_weights, phase_loss_weight=training_config.phase_loss_weight).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    writer = SummaryWriter(log_dir=str(training_config.runs_dir / stage))

    checkpoint_name = "gesture_tcn_jester.pt" if stage == "pretrain" else "gesture_tcn_finetuned.pt"
    checkpoint_path = training_config.models_dir / checkpoint_name

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    epoch_limit = max_epochs if max_epochs is not None else training_config.max_epochs

    for epoch in range(epoch_limit):
        train_loss, train_report = _run_epoch(net, loss_fn, train_loader, optimizer, device)
        val_loss, val_report = _run_epoch(net, loss_fn, val_loader, None, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("macro_f1/train", train_report.macro_f1, epoch)
        writer.add_scalar("macro_f1/val", val_report.macro_f1, epoch)
        for cls, f1 in val_report.per_class_f1.items():
            writer.add_scalar(f"f1_per_class_val/{DEFAULT_GESTURE_LABELS[cls]}", f1, epoch)

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_report.macro_f1:.4f}"
        )

        if val_report.macro_f1 > best_macro_f1:
            best_macro_f1 = val_report.macro_f1
            epochs_without_improvement = 0
            metadata = ModelMetadata(
                version=f"{stage}-epoch{epoch}",
                trained=True,
                training_data_source=dataset_id,
                evaluation_notes=(
                    f"val macro-F1={best_macro_f1:.4f}, dataset_id={dataset_id!r}, "
                    f"epoch={epoch}, feature_dim={feature_dimension(gesture_config)} "
                    f"(development-principles.md 1.4: dataset·조건·측정 시점 기록)"
                ),
            )
            _save_checkpoint(net, checkpoint_path, metadata)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.early_stopping_patience:
                print(f"early stopping at epoch {epoch} (best val macro-F1={best_macro_f1:.4f})")
                break

    writer.close()
    return checkpoint_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--train-persons", nargs="+", default=None, help="--stage finetune 전용")
    parser.add_argument("--val-persons", nargs="+", default=None, help="--stage finetune 전용")
    parser.add_argument("--epochs", type=int, default=None, help="드라이런용 epoch 수 상한")
    args = parser.parse_args(argv)

    checkpoint_path = train(
        stage=args.stage,
        init_from=args.init_from,
        train_persons=args.train_persons,
        val_persons=args.val_persons,
        max_epochs=args.epochs,
    )
    print(f"checkpoint saved to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

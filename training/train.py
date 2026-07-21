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
from dataclasses import asdict, replace
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
from jarvis.gesture_fusion.model_protocol import (
    BACKGROUND_TIE_TOLERANCE,
    DEFAULT_GESTURE_LABELS,
    ModelMetadata,
)
from training.config import DEFAULT_TRAINING_CONFIG, TrainingConfig
from training.dataset import GESTURE_LABEL_TO_INDEX, IGNORE_INDEX, ClipDataset, collate_fn
from training.losses import GesturePhaseLoss, compute_class_weights
from training.metrics import (
    ClassificationReport,
    collapse_class_indices,
    compute_classification_report,
)

_NUM_GESTURE_CLASSES = len(DEFAULT_GESTURE_LABELS)

# 배경 클래스 index는 `ModelConfig`가 라벨 집합에서 유도한다 — 여기서 다시 세지 않아
# 라벨이 바뀌어도 학습·추론이 같은 정의를 본다. feature_dim은 이 유도와 무관하므로
# 자리표시자를 쓴다(shape 검증만 통과하면 된다).
_LABEL_LAYOUT = ModelConfig(feature_dim=1)
_BACKGROUND_INDICES = _LABEL_LAYOUT.background_indices
_FOREGROUND_INDICES = _LABEL_LAYOUT.foreground_indices
_NUM_COLLAPSED_CLASSES = 1 + len(_FOREGROUND_INDICES)
# 텐서 인덱싱에는 list를 쓴다 — tuple은 numpy에서 "다차원 인덱스"로 해석될 여지가
# 있어(축 하나에 대한 fancy indexing이 아니라) 의미가 모호하다.
_BACKGROUND_SELECTOR = list(_BACKGROUND_INDICES)
_FOREGROUND_SELECTOR = list(_FOREGROUND_INDICES)


def _build_model(gesture_config: GestureConfig) -> tuple[CausalTCN, ModelConfig]:
    """`feature_dimension(gesture_config)`을 동적으로 읽어 모델 입력 차원을 맞춘다.

    하드코딩하지 않는다 — `GestureConfig`가 나중에 조정돼도(예: feature 그룹
    on/off) 이 스크립트를 고칠 필요가 없다.
    """
    model_config = ModelConfig(feature_dim=feature_dimension(gesture_config))
    return CausalTCN(model_config), model_config


def _compute_input_stats(
    dataset: ClipDataset, max_clips: int = 2000
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """학습셋 표본에서 차원별 (평균, 표준편차)를 구한다 — `CausalTCN` 입력 표준화용.

    feature 그룹 간 스케일이 수십 배 차이나(위치 RMS ~0.9 vs 손목 평행이동 ~30)
    정규화 없이는 판별력이 큰 그룹이 큰 값에 묻힌다. 전량이 아니라 표본을 쓰는 이유는
    통계가 표본 수천 클립이면 충분히 안정적이고, 매 학습마다 전체를 훑는 비용이
    크기 때문이다(표본은 고정 stride라 실행마다 재현된다 — 난수 미사용).
    """
    count = min(len(dataset), max_clips)
    stride = max(1, len(dataset) // count)
    total = torch.zeros(0)
    total_sq = torch.zeros(0)
    n_frames = 0
    for i in range(0, len(dataset), stride):
        features, _, _ = dataset[i]
        if total.numel() == 0:
            total = torch.zeros(features.shape[1], dtype=torch.float64)
            total_sq = torch.zeros(features.shape[1], dtype=torch.float64)
        f64 = features.to(torch.float64)
        total += f64.sum(dim=0)
        total_sq += (f64**2).sum(dim=0)
        n_frames += features.shape[0]
    if n_frames == 0:
        raise ValueError("cannot compute input stats from an empty dataset")
    mean = total / n_frames
    var = torch.clamp(total_sq / n_frames - mean**2, min=0.0)
    return mean.to(torch.float32), torch.sqrt(var).to(torch.float32)


def _collapsed_predictions(gesture_logits: "torch.Tensor") -> "torch.Tensor":
    """런타임 결정 규칙과 **동일하게** 배경 확률을 합산해 접은 공간의 예측을 만든다.

    `argmax` 후에 배경 index를 0으로 접는 것과는 결과가 다르다 — 그렇게 하면 배경
    표 분산이 그대로 남아, 실제 런타임(`collapse_background_probabilities`)보다
    낙관적이거나 비관적인 엉뚱한 점수가 나온다. 지표는 배포되는 결정 규칙을 재야
    한다. 동점 시 배경이 이기는 것도 `torch.cat`에서 배경을 앞에 두어 일치시킨다 —
    다만 배경(합산)과 전경 최댓값(단일)은 다른 부동소수점 경로를 거쳐 나오므로,
    수학적으로 동점인 경우도 실제로는 미세하게(~1e-7) 어긋난다. `argmax`는 엄격한
    최댓값만 보므로 그 미세한 잡음에 밀려 "동점이면 배경" 규칙이 발동하지 않는다 —
    `BACKGROUND_TIE_TOLERANCE`만큼 배경에 여유를 줘 `collapse_background_probabilities`
    와 같은 실질적 동작을 보장한다.
    """
    probs = torch.softmax(gesture_logits, dim=1)
    background = probs[:, _BACKGROUND_SELECTOR, :].sum(dim=1, keepdim=True)
    foreground = probs[:, _FOREGROUND_SELECTOR, :]
    collapsed = torch.cat([background + BACKGROUND_TIE_TOLERANCE, foreground], dim=1)
    return collapsed.argmax(dim=1)


def _run_epoch(
    net: CausalTCN,
    loss_fn: GesturePhaseLoss,
    loader: DataLoader,
    optimizer: "torch.optim.Optimizer | None",
    device: str,
) -> tuple[float, ClassificationReport, ClassificationReport]:
    """한 epoch(학습 또는 검증)을 돈다. `optimizer=None`이면 검증(gradient 없음).

    리포트를 둘 반환한다 — 원본 클래스 기준(진단용 클래스별 F1·혼동행렬)과 배경을
    접은 기준(모델 선택·early stopping용). 접은 쪽이 우리가 실제로 원하는 성능이다.
    """
    is_train = optimizer is not None
    net.train(is_train)

    total_loss = 0.0
    n_batches = 0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_collapsed_preds: list[np.ndarray] = []

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
        detached_logits = gesture_logits.detach()
        all_preds.append(detached_logits.argmax(dim=1).cpu().numpy())
        all_collapsed_preds.append(_collapsed_predictions(detached_logits).cpu().numpy())
        all_targets.append(gesture_targets.cpu().numpy())

    flat_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    flat_targets = np.concatenate([t.reshape(-1) for t in all_targets])
    flat_collapsed_preds = np.concatenate([p.reshape(-1) for p in all_collapsed_preds])
    report = compute_classification_report(
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
    return total_loss / max(1, n_batches), report, collapsed_report


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

    # 입력 표준화 통계는 **학습셋에서만** 구한다(검증셋 통계가 새면 평가가 낙관적으로
    # 편향된다). augment=False인 복제본을 쓰지 않고 train_dataset을 그대로 표본하되,
    # augmentation은 좌우반전·시간축 리샘플이라 차원별 스케일을 바꾸지 않으므로 통계에
    # 미치는 영향이 작다. init_from(파인튜닝)일 때는 사전학습 통계를 그대로 이어쓴다.
    if init_from is None:
        mean, std = _compute_input_stats(train_dataset)
        net.set_input_normalization(mean.to(device), std.to(device))

    # 클립 수가 아니라 **유효 프레임 수**로 센다 — cross-entropy가 프레임마다 걸리므로
    # 클립 수로 세면 클립당 유효 프레임 수의 클래스별 편차가 그대로 가중치 왜곡이 된다
    # (2026-07-20 수정, `ClipDataset.valid_frames_per_label` 참조).
    class_weights = compute_class_weights(
        {
            GESTURE_LABEL_TO_INDEX[label]: frames
            for label, frames in train_dataset.valid_frames_per_label().items()
        },
        num_classes=_NUM_GESTURE_CLASSES,
    )
    # 배경이 여러 클래스로 나뉜 데서 따라온 암묵적 강조를 명시적 파라미터로 드러낸다
    # (TrainingConfig.background_class_weight_scale 문서 참조). 기본 1.0은 무보정.
    if training_config.background_class_weight_scale != 1.0:
        class_weights[_BACKGROUND_SELECTOR] *= training_config.background_class_weight_scale
    class_weights = class_weights.to(device)
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

    # 코사인 스케줄(2026-07-20) — 고정 LR로는 val macro-F1이 뚜렷한 수렴 없이
    # 진동만 하는 패턴이 관찰됐다. T_max를 실제 epoch 상한(early stopping으로
    # 더 일찍 끝날 수 있음)에 맞춰, 학습이 진행될수록 step을 줄여 진동을 가라앉힌다.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epoch_limit),
        eta_min=training_config.learning_rate * training_config.lr_min_factor,
    )

    for epoch in range(epoch_limit):
        train_loss, train_report, train_collapsed = _run_epoch(
            net, loss_fn, train_loader, optimizer, device
        )
        val_loss, val_report, val_collapsed = _run_epoch(net, loss_fn, val_loader, None, device)
        scheduler.step()

        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("macro_f1_raw/train", train_report.macro_f1, epoch)
        writer.add_scalar("macro_f1_raw/val", val_report.macro_f1, epoch)
        writer.add_scalar("macro_f1/train", train_collapsed.macro_f1, epoch)
        writer.add_scalar("macro_f1/val", val_collapsed.macro_f1, epoch)
        for cls, f1 in val_report.per_class_f1.items():
            writer.add_scalar(f"f1_per_class_val/{DEFAULT_GESTURE_LABELS[cls]}", f1, epoch)

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_collapsed.macro_f1:.4f} (raw {val_report.macro_f1:.4f})"
        )

        # 선택 기준은 **접은** macro-F1이다 — 원본 기준으로 고르면 서로 구분할 필요가
        # 없는 배경 클래스들을 잘 가르는 epoch이 뽑힌다(collapse_class_indices 참고).
        if val_collapsed.macro_f1 > best_macro_f1:
            best_macro_f1 = val_collapsed.macro_f1
            epochs_without_improvement = 0
            metadata = ModelMetadata(
                version=f"{stage}-epoch{epoch}",
                trained=True,
                training_data_source=dataset_id,
                evaluation_notes=(
                    f"val macro-F1={best_macro_f1:.4f} (배경 클래스 합산 후 "
                    f"{_NUM_COLLAPSED_CLASSES}클래스 기준 — 런타임 결정 규칙과 동일), "
                    f"원본 {_NUM_GESTURE_CLASSES}클래스 기준={val_report.macro_f1:.4f}, "
                    f"dataset_id={dataset_id!r}, epoch={epoch}, "
                    f"feature_dim={feature_dimension(gesture_config)} "
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader 워커 프로세스 수(기본 TrainingConfig.num_workers). GPU 학습에서는 "
            "모델 자체가 가벼워 feature 조립(순수 numpy, CPU)이 병목이 되기 쉽다 — "
            "GPU 사용률이 낮게 나오면 이 값을 코어 수에 맞춰 올린다."
        ),
    )
    args = parser.parse_args(argv)

    training_config = DEFAULT_TRAINING_CONFIG
    if args.num_workers is not None:
        training_config = replace(training_config, num_workers=args.num_workers)

    checkpoint_path = train(
        stage=args.stage,
        training_config=training_config,
        init_from=args.init_from,
        train_persons=args.train_persons,
        val_persons=args.val_persons,
        max_epochs=args.epochs,
    )
    print(f"checkpoint saved to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

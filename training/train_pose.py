"""정적 손 자세 분류기 학습 — 닫힌 자세를 가르는 작은 MLP.

배경: 커서 이동·클릭·스크롤 등을 가르려면 6개 정적 자세를 구분해야 하는데, 손으로
만든 판별자(엄지-검지 거리, 검지 길이 등)는 핀치와 주먹에서 전부 우연 수준이었다
(2026-07-20 측정: 오류율 31~34%, 다수 클래스 기준선 34.5%). 21점 전체를 보는 작은
MLP는 같은 데이터에서 90%대를 낸다. 열린 자세(검지 하나·두 손가락·보)는 휴리스틱으로도
갈리지만, 경계를 한 곳에서 관리하려고 6개 전부를 이 분류기가 맡는다.

전처리는 런타임과 **반드시 같아야 한다**(모델 재현성):
    raw(x, y) → normalize_hand(손목 원점 + palm_scale) → One-Euro(GestureConfig.smoothing_*)
= 디버깅 툴 3번 탭에 뜨는 좌표. 수집기가 전 프레임 raw + timestamp를 남기므로 여기서
그대로 재현한다. 저장하는 메타데이터에 이 설정을 함께 넣어, 추론 측이 다른 전처리를
쓰면 드러나게 한다.

평가는 **에피소드 단위 홀드아웃**이다. 30fps에서 인접 프레임은 거의 같은 이미지라
무작위 분할을 하면 훈련·테스트에 사실상 같은 프레임이 들어가 정확도가 부풀려진다
(실측: 무작위 84.6% vs 에피소드 단위 82%). 한 번 녹화한 구간은 통째로 한쪽에만 넣는다.

사용법:
    python -m training.train_pose --data .claude/tmp/pose_frames.npz
    python -m training.train_pose --data <npz> --out models/hand_pose_classifier.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from jarvis.gesture_fusion.config import (
    DEFAULT_GESTURE_CONFIG,
    LANDMARK_DIMS,
    GestureConfig,
)
from jarvis.gesture_fusion.landmarks import RawHandLandmarks, normalize_hand, palm_tilt_degrees
from jarvis.gesture_fusion.smoothing import OneEuroFilter

FloatArray = npt.NDArray[np.float64]

# 같은 라벨이라도 이보다 긴 공백이 있으면 다른 에피소드(다시 녹화한 구간)로 본다.
EPISODE_GAP_MS = 500
# 기울기 구간 경계 — 자세별 신뢰 구간표를 만들 때 쓴다(3단계 게이트가 소비).
TILT_BUCKETS = ((0.0, 20.0), (20.0, 30.0), (30.0, 90.0))


@dataclass(frozen=True, slots=True)
class PoseSamples:
    """전처리를 마친 학습 표본 — 모델 입력과 평가에 필요한 부속 정보."""

    features: FloatArray          # (N, 21*LANDMARK_DIMS) 정규화·평활된 좌표
    labels: npt.NDArray[np.int64]  # (N,)
    episodes: npt.NDArray[np.int64]  # (N,) 에피소드 id — 분할 단위
    tilt_degrees: FloatArray       # (N,) 손바닥 기울기(도). 신뢰 구간표용
    label_names: tuple[str, ...]


def load_samples(path: Path, config: GestureConfig = DEFAULT_GESTURE_CONFIG) -> PoseSamples:
    """수집 npz를 읽어 런타임과 동일한 전처리를 적용한다.

    라벨이 없는 프레임도 필터에 통과시킨다 — One-Euro는 연속 스트림에 상태를 누적하는
    causal 필터라, 라벨 프레임만 넣으면 런타임과 다른 값이 나온다. 세션 경계와 검출
    실패에서는 필터를 리셋해 끊긴 구간을 잇지 않는다.
    """
    data = np.load(path)
    names = tuple(str(s) for s in data["label_names"])
    points, ts = data["points"], data["ts_ms"]
    detected, labels, sessions = data["detected"], data["label"], data["session"]

    feats: list[FloatArray] = []
    labs: list[int] = []
    eps: list[int] = []
    tilts: list[float] = []
    smoother: OneEuroFilter | None = None
    prev_session: int | None = None
    prev_label: int | None = None
    prev_ts = -(10**9)
    episode = -1

    for i in range(len(labels)):
        if sessions[i] != prev_session or not detected[i]:
            smoother = None
            prev_session = int(sessions[i])
        if not detected[i] or not np.isfinite(points[i]).all():
            continue
        raw = RawHandLandmarks(
            timestamp_ms=int(ts[i]),
            frame_id=i,
            points=points[i][:, :LANDMARK_DIMS].astype(np.float64),
            handedness="Right",
            detection_confidence=1.0,
            handedness_score=1.0,
            palm_tilt_degrees=palm_tilt_degrees(points[i].astype(np.float64), config),
        )
        observation = normalize_hand(raw, config)
        if not observation.hand_detected:
            smoother = None
            continue
        if smoother is None:
            smoother = OneEuroFilter(
                min_cutoff=config.smoothing_min_cutoff,
                beta=config.smoothing_beta,
                d_cutoff=config.smoothing_d_cutoff,
            )
        smoothed = smoother.filter(observation.landmarks, int(ts[i]))
        if labels[i] < 0:
            continue
        if labels[i] != prev_label or ts[i] - prev_ts > EPISODE_GAP_MS:
            episode += 1
        prev_label, prev_ts = int(labels[i]), int(ts[i])
        feats.append(np.asarray(smoothed, dtype=np.float64).reshape(-1))
        labs.append(int(labels[i]))
        eps.append(episode)
        tilts.append(
            float("nan") if observation.palm_tilt_degrees is None else observation.palm_tilt_degrees
        )

    if not feats:
        raise ValueError(f"{path}에 학습 가능한 라벨 프레임이 없다")
    return PoseSamples(
        features=np.array(feats),
        labels=np.array(labs, dtype=np.int64),
        episodes=np.array(eps, dtype=np.int64),
        tilt_degrees=np.array(tilts),
        label_names=names,
    )


def episode_split(
    labels: npt.NDArray[np.int64], episodes: npt.NDArray[np.int64], test_fraction: float, seed: int
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """라벨별로 에피소드의 일부를 통째로 테스트에 배정한다(층화 + 누수 차단)."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        ids = np.unique(episodes[labels == cls])
        if len(ids) < 2:
            raise ValueError(
                f"라벨 {cls}의 에피소드가 {len(ids)}개뿐이라 정직한 분할이 불가능하다. "
                "짧은 에피소드를 여러 번 나눠 다시 수집하라"
            )
        rng.shuffle(ids)
        for episode_id in ids[: max(1, int(round(test_fraction * len(ids))))]:
            test |= episodes == episode_id
    return ~test, test


def build_model(input_dim: int, num_classes: int) -> nn.Sequential:
    """참고 구현과 같은 규모의 MLP. 노트북 CPU로 수 초면 학습된다."""
    return nn.Sequential(
        nn.Linear(input_dim, 20), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, num_classes),
    )


def train(
    samples: PoseSamples,
    train_mask: npt.NDArray[np.bool_],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Sequential, FloatArray, FloatArray]:
    """모델과 표준화 통계를 돌려준다. 통계는 **훈련셋에서만** 구한다(누수 방지)."""
    torch.manual_seed(seed)
    mean = samples.features[train_mask].mean(axis=0)
    std = samples.features[train_mask].std(axis=0) + 1e-8
    x = torch.tensor((samples.features - mean) / std, dtype=torch.float32)
    y = torch.tensor(samples.labels, dtype=torch.long)

    model = build_model(samples.features.shape[1], len(samples.label_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    indices = np.flatnonzero(train_mask)
    for _ in range(epochs):
        model.train()
        # 매 epoch 셔플 — 표본이 시간순이라 셔플하지 않으면 배치 하나가 통째로 같은
        # 라벨의 연속 구간이 되어 경사가 라벨별로 몰린다(실측: 정확도 70% → 83%).
        order = rng.permutation(indices)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(x[batch]), y[batch]).backward()
            optimizer.step()
    return model, mean, std


def evaluate(
    model: nn.Sequential,
    samples: PoseSamples,
    mean: FloatArray,
    std: FloatArray,
    test_mask: npt.NDArray[np.bool_],
) -> dict[str, object]:
    """정확도·혼동행렬과 **자세별 기울기 내성표**를 낸다.

    기울기 내성은 자세마다 크게 다르다(실측: 20~30° 구간에서 two_fingers 83%,
    index_point 0%). 전역 임계 하나로는 스크롤 자세를 막거나 위험한 자세를 통과시키게
    되므로, 예측된 클래스별로 허용 각도를 달리 쓸 수 있게 표를 함께 저장한다.
    """
    model.eval()
    x = torch.tensor((samples.features - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(x).argmax(dim=1).numpy()
    truth, predicted = samples.labels[test_mask], predictions[test_mask]
    num_classes = len(samples.label_names)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(truth, predicted):
        confusion[t, p] += 1

    tilt_table: dict[str, dict[str, float | None]] = {}
    for cls, name in enumerate(samples.label_names):
        per_bucket: dict[str, float | None] = {}
        for low, high in TILT_BUCKETS:
            mask = test_mask & (samples.labels == cls)
            mask &= (samples.tilt_degrees > low) & (samples.tilt_degrees <= high)
            # 표본이 적으면 정확도를 지어내지 않는다 — None이면 근거 없음이 드러난다.
            per_bucket[f"{low:.0f}-{high:.0f}"] = (
                float((predictions[mask] == cls).mean()) if mask.sum() >= 25 else None
            )
        tilt_table[name] = per_bucket

    return {
        "accuracy": float((predicted == truth).mean()),
        "test_samples": int(test_mask.sum()),
        "confusion": confusion.tolist(),
        "recall": {
            name: float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None
            for i, name in enumerate(samples.label_names)
        },
        "tilt_tolerance": tilt_table,
    }


def save(
    path: Path,
    model: nn.Sequential,
    mean: FloatArray,
    std: FloatArray,
    samples: PoseSamples,
    config: GestureConfig,
    metrics: dict[str, object],
) -> None:
    """가중치와 함께 **전처리 설정을 저장한다** — 추론 측이 다르게 쓰면 드러나야 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": samples.features.shape[1],
            "label_names": list(samples.label_names),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "preprocessing": {
                "landmark_dims": LANDMARK_DIMS,
                "origin_index": config.origin_index,
                "palm_scale_root_index": config.palm_scale_root_index,
                "palm_scale_tip_index": config.palm_scale_tip_index,
                "smoothing_min_cutoff": config.smoothing_min_cutoff,
                "smoothing_beta": config.smoothing_beta,
                "smoothing_d_cutoff": config.smoothing_d_cutoff,
            },
            "metrics": metrics,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="정적 손 자세 분류기를 학습한다")
    parser.add_argument("--data", type=Path, required=True, help="수집 npz 경로")
    parser.add_argument("--out", type=Path, default=Path("models/hand_pose_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.data.is_file():
        print(f"데이터 파일이 없다: {args.data}", file=sys.stderr)
        return 1

    config = DEFAULT_GESTURE_CONFIG
    samples = load_samples(args.data, config)
    train_mask, test_mask = episode_split(
        samples.labels, samples.episodes, args.test_fraction, args.seed
    )
    print(
        f"표본 {len(samples.labels)}  특징 {samples.features.shape[1]}차원  "
        f"에피소드 {len(np.unique(samples.episodes))}개\n"
        f"훈련 {train_mask.sum()}  테스트 {test_mask.sum()}"
    )
    model, mean, std = train(
        samples, train_mask, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed
    )
    metrics = evaluate(model, samples, mean, std, test_mask)
    save(args.out, model, mean, std, samples, config, metrics)

    print(f"\n테스트 정확도 {metrics['accuracy']:.1%}  (표본 {metrics['test_samples']})")
    print("\n자세별 재현율")
    for name, value in metrics["recall"].items():  # type: ignore[union-attr]
        print(f"  {name:13s} {'-' if value is None else f'{value:.1%}'}")
    print("\n자세별 기울기 내성 (표본 25개 미만은 '-' — 근거 없음)")
    print(f"  {'자세':13s}{'0-20°':>9s}{'20-30°':>9s}{'30-90°':>9s}")
    for name, buckets in metrics["tilt_tolerance"].items():  # type: ignore[union-attr]
        cells = "".join(
            f"{'-' if v is None else f'{v:.0%}':>9s}" for v in buckets.values()
        )
        print(f"  {name:13s}{cells}")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

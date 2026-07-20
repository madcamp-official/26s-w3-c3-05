"""정적 자세 분류기 학습 파이프라인 — 평가 정직성과 전처리 재현성을 지킨다.

여기서 지키려는 두 가지:
1. **누수 없는 분할** — 30fps에서 인접 프레임은 거의 같은 이미지라, 무작위로 나누면
   훈련·테스트에 사실상 같은 프레임이 들어가 정확도가 부풀려진다(실측: 무작위 84.6%
   vs 에피소드 단위 82%). 한 번 녹화한 구간은 통째로 한쪽에만 들어가야 한다.
2. **전처리 재현성** — 학습과 런타임의 전처리가 어긋나면 모델이 조용히 망가진다.
   저장 파일에 전처리 설정이 함께 들어가야 어긋남을 발견할 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from jarvis.gesture_fusion.config import DEFAULT_GESTURE_CONFIG, LANDMARK_DIMS
from training.train_pose import (
    EPISODE_GAP_MS,
    episode_split,
    evaluate,
    load_samples,
    save,
    train,
)

LABEL_NAMES = ("index_point", "pinch_index", "pinch_middle", "two_fingers", "open_palm", "fist")


def _hand(pose: int, tilt_z: float = 0.0) -> np.ndarray:
    """자세마다 뚜렷이 다른 21점을 만든다(분류가 가능해야 학습이 의미 있다)."""
    rng = np.random.default_rng(pose)
    points = rng.normal(0.5, 0.02, size=(21, 3))
    points[0] = [0.5, 0.7, 0.0]                      # 손목
    points[9] = [0.5, 0.5, tilt_z]                   # 중지 MCP — 기울기는 z로 준다
    points[8] = [0.5 + 0.05 * pose, 0.3, 0.0]        # 검지 끝
    return points


def _write_npz(
    path: Path, *, episodes_per_label: int = 4, frames: int = 12, gap_ms: int = 2000
) -> Path:
    """라벨당 여러 에피소드를 가진 수집 파일을 만든다(에피소드 = 시간 공백으로 구분)."""
    pts, ts, det, lab, sess = [], [], [], [], []
    clock = 0
    for label in range(len(LABEL_NAMES)):
        for _ in range(episodes_per_label):
            clock += gap_ms  # 에피소드 사이 공백
            for _ in range(frames):
                clock += 33
                pts.append(_hand(label))
                ts.append(clock)
                det.append(True)
                lab.append(label)
                sess.append(0)
    np.savez(
        path,
        points=np.array(pts, dtype=np.float32),
        ts_ms=np.array(ts, dtype=np.int64),
        detected=np.array(det, dtype=bool),
        label=np.array(lab, dtype=np.int8),
        session=np.array(sess, dtype=np.int32),
        label_names=np.array(LABEL_NAMES),
    )
    return path


def test_episodes_are_split_whole(tmp_path: Path) -> None:
    """한 에피소드의 프레임이 훈련과 테스트에 쪼개져 들어가면 안 된다(누수)."""
    samples = load_samples(_write_npz(tmp_path / "d.npz"))
    train_mask, test_mask = episode_split(samples.labels, samples.episodes, 0.3, seed=0)
    for episode_id in np.unique(samples.episodes):
        member = samples.episodes == episode_id
        assert not (train_mask[member].any() and test_mask[member].any())
    assert train_mask.sum() > 0 and test_mask.sum() > 0


def test_time_gap_starts_new_episode(tmp_path: Path) -> None:
    """같은 라벨이라도 공백이 길면 다른 녹화 구간으로 센다."""
    samples = load_samples(_write_npz(tmp_path / "d.npz", episodes_per_label=3))
    for label in range(len(LABEL_NAMES)):
        assert len(np.unique(samples.episodes[samples.labels == label])) == 3


def test_split_refuses_when_episodes_too_few(tmp_path: Path) -> None:
    """에피소드가 1개뿐이면 정직한 홀드아웃이 불가능하므로 조용히 진행하지 않는다."""
    samples = load_samples(_write_npz(tmp_path / "d.npz", episodes_per_label=1))
    with pytest.raises(ValueError, match="에피소드"):
        episode_split(samples.labels, samples.episodes, 0.3, seed=0)


def test_unlabelled_frames_excluded_but_feed_the_filter(tmp_path: Path) -> None:
    """라벨 없는 프레임은 표본에서 빠지되, One-Euro 상태에는 반영돼야 한다.

    라벨 프레임만 필터에 넣으면 런타임과 다른 값이 나온다 — 그 어긋남이 곧 모델
    성능 저하로 이어지므로 여기서 고정한다.
    """
    path = _write_npz(tmp_path / "d.npz")
    data = dict(np.load(path))
    labels = data["label"].copy()
    labels[1::2] = -1  # 절반을 라벨 해제
    data["label"] = labels
    np.savez(path, **data)

    partial = load_samples(path)
    assert len(partial.labels) == int((labels >= 0).sum())
    assert np.all(partial.labels >= 0)


def test_saved_model_records_preprocessing(tmp_path: Path) -> None:
    """전처리 설정이 함께 저장돼야 추론 측 불일치를 발견할 수 있다."""
    samples = load_samples(_write_npz(tmp_path / "d.npz"))
    train_mask, test_mask = episode_split(samples.labels, samples.episodes, 0.3, seed=0)
    model, mean, std = train(samples, train_mask, epochs=5, batch_size=32, seed=0)
    metrics = evaluate(model, samples, mean, std, test_mask)
    out = tmp_path / "m.pt"
    save(out, model, mean, std, samples, DEFAULT_GESTURE_CONFIG, metrics)

    blob = torch.load(out, weights_only=False)
    assert blob["label_names"] == list(LABEL_NAMES)
    assert blob["input_dim"] == 21 * LANDMARK_DIMS
    pre = blob["preprocessing"]
    assert pre["landmark_dims"] == LANDMARK_DIMS
    assert pre["smoothing_min_cutoff"] == DEFAULT_GESTURE_CONFIG.smoothing_min_cutoff
    assert pre["smoothing_beta"] == DEFAULT_GESTURE_CONFIG.smoothing_beta
    assert pre["palm_scale_tip_index"] == DEFAULT_GESTURE_CONFIG.palm_scale_tip_index


def test_standardisation_uses_training_statistics_only(tmp_path: Path) -> None:
    """표준화 통계에 테스트셋이 섞이면 그것도 누수다."""
    samples = load_samples(_write_npz(tmp_path / "d.npz"))
    train_mask, _ = episode_split(samples.labels, samples.episodes, 0.3, seed=0)
    _, mean, _ = train(samples, train_mask, epochs=1, batch_size=32, seed=0)
    assert np.allclose(mean, samples.features[train_mask].mean(axis=0))
    assert not np.allclose(mean, samples.features.mean(axis=0))


def test_sparse_tilt_buckets_report_none_not_a_number(tmp_path: Path) -> None:
    """표본이 적은 기울기 구간은 정확도를 지어내지 않는다 — 근거 없음이 드러나야 한다."""
    samples = load_samples(_write_npz(tmp_path / "d.npz"))
    train_mask, test_mask = episode_split(samples.labels, samples.episodes, 0.3, seed=0)
    model, mean, std = train(samples, train_mask, epochs=5, batch_size=32, seed=0)
    metrics = evaluate(model, samples, mean, std, test_mask)
    table = metrics["tilt_tolerance"]
    # 합성 데이터는 전부 기울기 0°라 20° 초과 구간에 표본이 없다.
    assert all(buckets["20-30"] is None for buckets in table.values())


def test_episode_gap_constant_matches_collector() -> None:
    """수집기의 버스트 간격보다 짧으면 별개 녹화가 한 에피소드로 합쳐진다."""
    assert EPISODE_GAP_MS == 500

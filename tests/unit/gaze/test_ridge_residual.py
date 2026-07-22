"""Ridge residual 오프라인 학습·A/B: 선형 편향 복원과 자세 구간 단위 split."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.feature_profile import TargetFeatureSample
from jarvis.gaze.ridge_residual import (
    ResidualDataset,
    _solve_linear,
    evaluate_cross_session,
    evaluate_leave_one_bin_out,
    train_kernel_residual,
    train_ridge_residual,
)

CENTER = (1.0, 5.0)


def _sample(head_yaw: float, index: int, *, bias_slope: float = 0.2) -> TargetFeatureSample:
    """head yaw에 비례하는 선형 편향이 낀 중앙 응시 샘플."""
    bias = bias_slope * head_yaw
    return TargetFeatureSample(
        gaze_yaw=CENTER[0] - bias + 0.01 * (index % 5),
        gaze_pitch=CENTER[1] + 0.5 * bias,
        head_yaw=head_yaw,
        head_pitch=0.1 * (index % 3),
        head_roll=0.0,
        face_scale=0.10,
        face_center_x=0.5,
        face_center_y=0.5,
    )


def _sweep(bias_slope: float = 0.2) -> list[TargetFeatureSample]:
    samples = []
    index = 0
    for head_yaw in (-25.0, -15.0, 0.0, 15.0, 25.0):
        for _ in range(20):
            samples.append(_sample(head_yaw + 0.1 * (index % 7), index, bias_slope=bias_slope))
            index += 1
    return samples


def test_solve_linear_matches_known_solution() -> None:
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    rhs = np.array([[5.0], [10.0]])
    solution = _solve_linear(matrix, rhs)
    assert np.allclose(matrix @ solution, rhs)


def test_ridge_recovers_linear_pose_bias() -> None:
    model = train_ridge_residual(_sweep(), CENTER, ridge_lambda=1e-6)
    probe = _sample(20.0, index=0)
    corrected = model.corrected_gaze(probe)
    raw_error = math.hypot(probe.gaze_yaw - CENTER[0], probe.gaze_pitch - CENTER[1])
    corrected_error = math.hypot(corrected[0] - CENTER[0], corrected[1] - CENTER[1])
    assert raw_error > 3.0
    assert corrected_error < 0.2


def test_leave_one_bin_out_ridge_beats_raw_on_linear_bias() -> None:
    report = evaluate_leave_one_bin_out(_sweep(), CENTER, GazeConfig(), ridge_lambda=1e-3)
    assert len(report.bins) >= 4
    for item in report.bins:
        assert item.ridge_error_deg < item.raw_error_deg
    assert any("이깁니다" in line for line in report.verdict_lines)


def test_leave_one_bin_out_reports_no_gain_on_unstructured_noise() -> None:
    samples = []
    for index in range(100):
        head_yaw = -25.0 + (index % 5) * 12.5
        samples.append(
            TargetFeatureSample(
                gaze_yaw=CENTER[0] + 4.0 * math.sin(index * 2.1),
                gaze_pitch=CENTER[1] + 4.0 * math.cos(index * 1.7),
                head_yaw=head_yaw,
                head_pitch=0.0,
                head_roll=0.0,
                face_scale=0.10,
            )
        )
    report = evaluate_leave_one_bin_out(samples, CENTER, GazeConfig(), ridge_lambda=1.0)
    assert any("활성화하지 않습니다" in line for line in report.verdict_lines) or all(
        abs(item.ridge_error_deg - item.raw_error_deg) < 2.0 for item in report.bins
    )


def _nonlinear_dataset(target_id: str, center: tuple[float, float]) -> ResidualDataset:
    """yaw에는 2차(포물선) 편향, pitch에는 head_pitch 비례 편향이 낀 스윕.

    포물선은 선형회귀가 원리적으로 못 맞추고, head-pitch 의존은 yaw 단독
    bin 보정표가 못 맞춘다 — 6D 커널만 둘 다 잡을 수 있는 구성이다.
    """
    samples = []
    index = 0
    for head_yaw in (-25.0, -15.0, 0.0, 15.0, 25.0):
        for head_pitch in (-12.0, 0.0, 12.0):
            for _ in range(8):
                yaw_jitter = head_yaw + 0.2 * (index % 5)
                bias_yaw = 0.012 * yaw_jitter * yaw_jitter
                bias_pitch = 0.25 * head_pitch
                samples.append(
                    TargetFeatureSample(
                        gaze_yaw=center[0] - bias_yaw + 0.02 * (index % 3),
                        gaze_pitch=center[1] - bias_pitch,
                        head_yaw=yaw_jitter,
                        head_pitch=head_pitch,
                        head_roll=0.0,
                        face_scale=0.10,
                        face_center_x=0.5,
                        face_center_y=0.5,
                    )
                )
                index += 1
    return ResidualDataset(target_id=target_id, center_yaw_pitch=center, samples=tuple(samples))


def test_kernel_beats_linear_and_table_on_nonlinear_bias() -> None:
    train = [_nonlinear_dataset("left", (-8.0, 4.0)), _nonlinear_dataset("right", (8.0, 4.0))]
    evaluation = [_nonlinear_dataset("held-out", (0.0, 6.0))]
    report = evaluate_cross_session(train, evaluation, GazeConfig(), kernel_bandwidth=0.5)
    assert len(report.bins) >= 4
    for item in report.bins:
        assert item.kernel_error_deg < item.raw_error_deg
    lines = report.verdict_lines
    assert any(line.startswith("Kernel") and "PASS" in line for line in lines)
    # 포물선 편향은 전역 선형이 못 맞춘다 — 가장자리 bin에서 커널이 ridge보다 낫다.
    edge_bins = [b for b in report.bins if "[+20" in b.label or "[-30" in b.label]
    assert edge_bins
    for item in edge_bins:
        assert item.kernel_error_deg < item.ridge_error_deg
    # 활성화 기준(서로 다른 날 2세션) 안내가 항상 붙는다.
    assert any("2개" in line for line in lines)


def test_kernel_refuses_to_extrapolate_far_from_training_poses() -> None:
    dataset = _nonlinear_dataset("center", (0.0, 5.0))
    kernel = train_kernel_residual(dataset.delta_pairs(), bandwidth=0.5)
    far_query = TargetFeatureSample(
        gaze_yaw=0.0,
        gaze_pitch=0.0,
        head_yaw=70.0,
        head_pitch=40.0,
        head_roll=25.0,
        face_scale=0.30,
        face_center_x=0.9,
        face_center_y=0.1,
    )
    assert kernel.predict_delta(far_query) == (0.0, 0.0)


def test_train_validates_inputs() -> None:
    with pytest.raises(ValueError):
        train_ridge_residual([_sample(0.0, 0)], CENTER)
    with pytest.raises(ValueError):
        train_ridge_residual(_sweep(), CENTER, ridge_lambda=-1.0)

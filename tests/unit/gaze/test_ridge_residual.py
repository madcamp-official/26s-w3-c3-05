"""Ridge residual 오프라인 학습·A/B: 선형 편향 복원과 자세 구간 단위 split."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.feature_profile import TargetFeatureSample
from jarvis.gaze.ridge_residual import (
    _solve_linear,
    evaluate_leave_one_bin_out,
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


def test_train_validates_inputs() -> None:
    with pytest.raises(ValueError):
        train_ridge_residual([_sample(0.0, 0)], CENTER)
    with pytest.raises(ValueError):
        train_ridge_residual(_sweep(), CENTER, ridge_lambda=-1.0)

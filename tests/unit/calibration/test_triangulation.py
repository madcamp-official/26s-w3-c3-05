"""10초 등록 동안 모은 시선 광선의 삼각측량 + 품질 게이트 검증.

리스트 순서: 좋은 조건(충분한 baseline·각도 다양성)은 채택되고, 서로 다른
퇴화 상황(머리 거의 안 움직임, 먼 물체+모자란 baseline)은 개별적으로 거부됨을
확인한다 — 하나의 지표만으로는 두 퇴화 상황을 동시에 잡아낼 수 없기 때문에
(documents/decisions.md), baseline_mm과 min_eigenvalue 게이트가 독립적으로
동작하는지가 이 테스트의 핵심이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.calibration.triangulation import triangulate_rays
from jarvis.gaze.config import GazeConfig


def _rays_at(
    target: np.ndarray, baseline_radius_mm: float, n: int = 24, depth_offset_mm: float = 0.0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """`target`을 향하는 n개의 정확한(잡음 없는) 광선을 원형 궤적의 원점에서 만든다."""
    origins = []
    directions = []
    for i in range(n):
        angle = i / n * 2 * np.pi
        origin = np.array(
            [
                baseline_radius_mm * np.cos(angle),
                baseline_radius_mm * np.sin(angle),
                depth_offset_mm,
            ]
        )
        direction = target - origin
        direction = direction / np.linalg.norm(direction)
        origins.append(origin)
        directions.append(direction)
    return origins, directions


def test_recovers_exact_point_from_noiseless_converging_rays() -> None:
    target = np.array([50.0, -20.0, 1500.0])
    origins, directions = _rays_at(target, baseline_radius_mm=150.0)

    result = triangulate_rays(origins, directions)

    np.testing.assert_allclose(result.center_mm, target, atol=1e-6)
    assert result.residual_rms_mm == pytest.approx(0.0, abs=1e-6)
    assert result.frame_count == 24


def test_good_parallax_passes_quality_gates() -> None:
    target = np.array([50.0, -20.0, 1500.0])
    origins, directions = _rays_at(target, baseline_radius_mm=150.0)

    result = triangulate_rays(origins, directions)

    assert result.passes_quality_gates(GazeConfig())


def test_barely_moved_head_fails_baseline_gate() -> None:
    """머리는 거의 안 움직이고(원점이 거의 고정) 눈만 돌린 경우.

    잔차는 여전히 거의 0(광선들이 실제로 한 점에 수렴)이라 residual_rms_mm만
    보면 오히려 "확신도가 높다"고 오판할 수 있다 — baseline_mm이 이 경우를
    독립적으로 잡아내야 한다.
    """
    target = np.array([50.0, -20.0, 1500.0])
    origins, directions = _rays_at(target, baseline_radius_mm=1.0)

    result = triangulate_rays(origins, directions)

    assert result.residual_rms_mm == pytest.approx(0.0, abs=1e-6)
    assert result.baseline_mm < GazeConfig().minimum_triangulation_baseline_mm
    assert not result.passes_quality_gates(GazeConfig())


def test_far_target_with_modest_baseline_fails_eigenvalue_gate() -> None:
    """물체가 멀어 baseline은 최소 기준을 넘겨도 광선이 여전히 거의 평행한 경우.

    baseline_mm만으로는 이 경우를 잡아내지 못한다 — min_eigenvalue가 독립적인
    두 번째 게이트로 필요하다.
    """
    config = GazeConfig()
    far_target = np.array([50.0, -20.0, 6000.0])
    origins, directions = _rays_at(far_target, baseline_radius_mm=65.0)

    result = triangulate_rays(origins, directions)

    assert result.baseline_mm >= config.minimum_triangulation_baseline_mm
    assert result.min_eigenvalue < config.minimum_triangulation_eigenvalue
    assert not result.passes_quality_gates(config)


def test_requires_at_least_two_rays() -> None:
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="at least two"):
        triangulate_rays([origin], [direction])


def test_requires_matching_lengths() -> None:
    origins = [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])]
    directions = [np.array([0.0, 0.0, 1.0])]
    with pytest.raises(ValueError, match="same length"):
        triangulate_rays(origins, directions)

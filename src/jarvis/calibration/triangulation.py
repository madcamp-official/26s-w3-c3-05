"""Multi-ray triangulation for 3D object registration.

10초 등록 동안 머리를 움직이며 얻은 여러 시선 광선(origin, direction)으로부터
물체의 카메라 기준 3D 위치를 추정한다. 사용자가 "다양한 각도·자세로 바라본다"는
요구사항의 핵심은 parallax다 — 머리가 충분히 움직여야 광선들이 한 점으로 정확히
수렴하고, 그렇지 않으면 :meth:`TriangulationResult.passes_quality_gates`가
실패해 `calibration/target_registration.py`가 각도 기반(mean_direction +
variance) 등록으로 자동 대체한다(documents/decisions.md).

여기서 나오는 `radius_mm`(호출자가 `residual_rms_mm`에서 유도)는 실제로 측정한
물체 크기가 아니라 삼각측량 잔차에서 유도한 판정 허용 오차다 — 그리고 원점 좌표
자체도 MediaPipe 표준 얼굴 모델 크기 가정에 기반한 근사 스케일이지 눈금으로
검증된 계량값이 아니다(models/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import Vector3


@dataclass(frozen=True, slots=True)
class TriangulationResult:
    """N개 광선의 최소자승 교차점과 그 신뢰도 지표."""

    center_mm: Vector3
    residual_rms_mm: float
    baseline_mm: float
    min_eigenvalue: float
    frame_count: int

    def passes_quality_gates(self, config: GazeConfig) -> bool:
        """등록에 3D 위치를 채택해도 좋을 만큼 조건이 좋았는지 판정한다.

        세 조건을 모두 만족해야 한다 — 하나만으로는 서로 다른 퇴화 상황을 걸러내지
        못한다: baseline(원점 퍼짐)만 보면 머리는 거의 안 움직이고 눈만 돌린 경우를
        "잘 퍼졌다"고 오판할 수 있고(광선들이 카메라 바로 앞 한 점으로 수렴),
        min_eigenvalue(광선 방향의 각도 다양성)만 보면 머리는 움직였지만 물체가
        멀어 광선이 여전히 거의 평행한 경우를 놓친다. residual은 두 경우 모두를
        간접적으로 잡아내는 최종 안전망이다.
        """
        return (
            self.frame_count >= config.minimum_triangulation_frames
            and self.baseline_mm >= config.minimum_triangulation_baseline_mm
            and self.min_eigenvalue >= config.minimum_triangulation_eigenvalue
            and self.residual_rms_mm <= config.maximum_triangulation_residual_mm
        )


def _robust_spread_mm(points: Vector3) -> float:
    """중앙값 기준 90퍼센타일 편차의 2배.

    `target_registration.py`가 이미 쓰는 강건 통계 관례(median + 90th-percentile
    deviation)를 그대로 따른다 — 프레임 하나의 튄 head-position 추정치가 전체
    baseline을 왜곡하지 않도록 한다.
    """
    center = np.median(points, axis=0)
    deviations = np.linalg.norm(points - center, axis=1)
    return float(2.0 * np.percentile(deviations, 90))


def triangulate_rays(origins: list[Vector3], directions: list[Vector3]) -> TriangulationResult:
    """N개 광선(origin, unit direction)의 최소자승 교차점을 계산한다.

    각 광선에 대해 투영 행렬 `(I - d d^T)`를 누적해 3x3 선형계 `A p = b`를 풀어
    모든 광선까지의 수직 거리 제곱합을 최소화하는 점 `p`를 구한다 — 여러 직선의
    최근접 교차점을 구하는 표준 공식이다. `np.linalg.lstsq`를 조건 분기 없이 항상
    사용해, 광선들이 거의 평행해 `A`가 거의 특이 행렬이 되는 경우에도 예외 없이
    최소-노름 해로 안전하게 수렴한다.

    `min_eigenvalue`는 프레임 수로 나눈 평균값이다 — 나누지 않으면 `A`가 광선
    수만큼 선형으로 커져 프레임을 많이 모을수록 같은 임계값을 항상 통과하게
    되므로, 프레임 수와 무관한 "광선 하나당 평균 각도 다양성"으로 정규화한다.
    """
    if len(origins) != len(directions):
        raise ValueError("origins and directions must have the same length")
    if len(origins) < 2:
        raise ValueError("triangulation requires at least two rays")

    identity = np.eye(3, dtype=np.float64)
    a_matrix = np.zeros((3, 3), dtype=np.float64)
    b_vector = np.zeros(3, dtype=np.float64)
    for origin, direction in zip(origins, directions):
        projector = identity - np.outer(direction, direction)
        a_matrix += projector
        b_vector += projector @ origin

    solution, _residuals, _rank, _singular_values = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
    center_mm: Vector3 = solution

    perpendicular_distances = np.empty(len(origins), dtype=np.float64)
    for i, (origin, direction) in enumerate(zip(origins, directions)):
        to_point = center_mm - origin
        along = float(np.dot(to_point, direction))
        perpendicular = to_point - along * direction
        perpendicular_distances[i] = float(np.linalg.norm(perpendicular))
    residual_rms_mm = float(np.sqrt(np.mean(perpendicular_distances**2)))

    origins_array = np.stack(origins)
    baseline_mm = _robust_spread_mm(origins_array)

    eigenvalues = np.linalg.eigvalsh(a_matrix)
    min_eigenvalue = float(eigenvalues[0]) / len(origins)

    return TriangulationResult(
        center_mm=center_mm,
        residual_rms_mm=residual_rms_mm,
        baseline_mm=baseline_mm,
        min_eigenvalue=min_eigenvalue,
        frame_count=len(origins),
    )

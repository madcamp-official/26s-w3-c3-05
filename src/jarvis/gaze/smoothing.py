"""Temporal smoothing over the composed gaze-direction vector.

README 7장 "최종 방식"의 첫 단계인 "최근 시선 방향 시퀀스 → Temporal smoothing"을
구현한다. 여기서 계산하는 `stability`는 Gaze→Fusion 계약(interface-contract.md)의
`stability` 필드로 그대로 출력된다.

추적이 끊긴 프레임(`GazeVector`가 None)이 들어오면 버퍼를 비운다 — 끊긴 구간
너머로 오래된 프레임을 이어붙여 매끈하게 보이도록 속이지 않는다
(development-principles.md 5절 "지연된 오래된 프레임을 무한히 처리하지 않는다").
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from jarvis.gaze.config import GazeConfig
from jarvis.gaze.features import GazeVector, Vector3


@dataclass(frozen=True, slots=True)
class SmoothedGaze:
    """평활화된 시선 방향 단위 벡터와 최근 구간의 안정성."""

    direction: Vector3
    stability: float
    timestamp_ms: int
    frame_id: int


class GazeSmoother:
    """최근 `window_frames`개 프레임에 대한 confidence-가중 이동 평균."""

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self._buffer: deque[GazeVector] = deque(maxlen=config.smoothing_window_frames)

    def reset(self) -> None:
        """추적 손실 시 버퍼를 비운다."""
        self._buffer.clear()

    def update(self, gaze_vector: GazeVector | None) -> SmoothedGaze | None:
        """새 프레임을 반영하고 평활화된 결과를 반환한다.

        `gaze_vector`가 None이면(추적 손실) 버퍼를 비우고 None을 반환한다.
        """
        if gaze_vector is None:
            self.reset()
            return None

        self._buffer.append(gaze_vector)

        weights = np.array([v.confidence for v in self._buffer], dtype=np.float64)
        directions = np.stack([v.direction for v in self._buffer])

        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            return None

        weighted_mean = (directions * weights[:, None]).sum(axis=0) / weight_sum
        norm = float(np.linalg.norm(weighted_mean))
        if norm == 0.0:
            return None
        mean_direction = weighted_mean / norm

        cosine_similarities = directions @ mean_direction
        stability = float((cosine_similarities * weights).sum() / weight_sum)
        stability = max(0.0, min(1.0, stability))

        latest = self._buffer[-1]
        return SmoothedGaze(
            direction=mean_direction,
            stability=stability,
            timestamp_ms=latest.timestamp_ms,
            frame_id=latest.frame_id,
        )

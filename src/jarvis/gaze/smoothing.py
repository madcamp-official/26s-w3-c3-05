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
from itertools import islice
import math

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
    origin: Vector3 | None = None
    """시선 광선의 평활화된 원점(mm 근사). 버퍼 안의 모든 프레임이 origin을
    가지고 있을 때만 계산되고, 그렇지 않으면 이 프레임은 None이다(3D 삼각측량·
    매칭에서만 쓰이며 각도 기반 경로는 이 값 없이도 그대로 동작한다)."""


class GazeSmoother:
    """최근 `window_frames`개 프레임에 대한 confidence-가중 이동 평균."""

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self._buffer: deque[GazeVector] = deque(maxlen=config.smoothing_window_frames)
        self._last_result: SmoothedGaze | None = None

    def reset(self) -> None:
        """추적 손실 시 버퍼를 비운다."""
        self._buffer.clear()
        self._last_result = None

    def hold(self, timestamp_ms: int, frame_id: int) -> SmoothedGaze | None:
        """Return the last result during a short blink without updating the buffer."""
        return self._hold_for(timestamp_ms, frame_id, self._config.blink_hold_ms)

    def hold_tracking_loss(self, timestamp_ms: int, frame_id: int) -> SmoothedGaze | None:
        """Return the last result during a short full face-tracking dropout."""
        return self._hold_for(timestamp_ms, frame_id, self._config.tracking_loss_hold_ms)

    def _hold_for(self, timestamp_ms: int, frame_id: int, hold_ms: int) -> SmoothedGaze | None:
        if self._last_result is None:
            return None
        if timestamp_ms - self._last_result.timestamp_ms > hold_ms:
            self.reset()
            return None
        self._last_result = SmoothedGaze(
            direction=self._last_result.direction,
            stability=self._last_result.stability,
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            origin=self._last_result.origin,
        )
        return self._last_result

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

        # Confidence-aware EMA is recomputed inside the bounded window. This gives
        # low-confidence frames less influence without retaining stale state after
        # every old frame has left the window.
        mean_direction = directions[0]
        alpha_range = self._config.ema_max_alpha - self._config.ema_min_alpha
        for item in islice(self._buffer, 1, None):
            alpha = self._config.ema_min_alpha + alpha_range * item.confidence
            blended = alpha * item.direction + (1.0 - alpha) * mean_direction
            blended_norm = float(np.linalg.norm(blended))
            if blended_norm == 0.0:
                return None
            mean_direction = blended / blended_norm

        cosine_similarities = directions @ mean_direction
        stability = float((cosine_similarities * weights).sum() / weight_sum)
        stability = max(0.0, min(1.0, stability))

        if self._last_result is not None and self._config.small_motion_deadzone_deg > 0.0:
            similarity = float(np.clip(mean_direction @ self._last_result.direction, -1.0, 1.0))
            angle_deg = math.degrees(math.acos(similarity))
            if angle_deg < self._config.small_motion_deadzone_deg:
                mean_direction = self._last_result.direction

        mean_origin = self._smoothed_origin(weights, weight_sum)

        latest = self._buffer[-1]
        self._last_result = SmoothedGaze(
            direction=mean_direction,
            stability=stability,
            timestamp_ms=latest.timestamp_ms,
            frame_id=latest.frame_id,
            origin=mean_origin,
        )
        return self._last_result

    def _smoothed_origin(self, weights: Vector3, weight_sum: float) -> Vector3 | None:
        """버퍼의 모든 프레임이 origin을 가질 때만 confidence-가중 평균을 반환한다.

        일부 프레임에만 origin이 있으면 신뢰할 수 없는 부분 평균을 만들지 않고
        None을 반환한다 — 이 프레임은 3D 매칭에서 제외되고 각도 기반 경로로만
        처리된다(값을 지어내지 않는다, development-principles.md 1·2절).
        """
        origins = [item.origin for item in self._buffer]
        if any(origin is None for origin in origins):
            return None
        stacked = np.stack([origin for origin in origins if origin is not None])
        mean_origin = (stacked * weights[:, None]).sum(axis=0) / weight_sum
        result: Vector3 = mean_origin
        return result

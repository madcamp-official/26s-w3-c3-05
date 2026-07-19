"""One-Euro filter — cheap adaptive low-pass to de-noise hand landmarks.

README 8장 처리 과정에서 랜드마크 좌표의 프레임별 고주파 지터(MediaPipe의 프레임
단위 회귀 특성)를 줄인다. 이 지터를 그대로 두면 다음 단계인 속도·가속도(이산 미분)가
노이즈를 증폭해 모델 입력이 크게 흔들리므로, **미분 전에** 위치를 평활화하는 것이
핵심이다(development-principles.md 7.2: 모델 입력의 비정상 신호를 그대로 흘리지 않음).

1€ 필터(Casiez, Roussel & Vogel, CHI 2012)는 EMA 두 번 + 나눗셈 한 번 수준의
연산으로, 고정 α EMA의 지터↔지연 트레이드오프를 속도 적응형 컷오프로 해결한다:
손이 정지하면 강하게 평활화(저컷오프), 빠르게 움직이면 컷오프를 열어 지연을 줄인다.
과거 표본만 쓰는 causal 필터이며 상태는 좌표당 (직전 출력, 직전 미분)뿐이다(O(1)).

배열 전체(예: 21×3 랜드마크)를 한 번에 처리하는 벡터화 구현이다 — mediapipe·torch에
의존하지 않아 카메라·모델 없이 단위 테스트할 수 있다.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _alpha(cutoff: FloatArray, dt_s: float) -> FloatArray:
    """주어진 컷오프(Hz)·dt에 대한 1차 저역통과 평활 계수 α ∈ (0, 1)."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return dt_s / (dt_s + tau)


class OneEuroFilter:
    """적응형 저역통과 필터. 프레임마다 값과 timestamp로 호출한다.

    `min_cutoff`는 기본 평활 강도(낮을수록 정지 시 더 부드러움), `beta`는 속도에
    따라 컷오프가 열리는 정도(높을수록 빠른 동작에서 지연이 적음), `d_cutoff`는 내부
    속도 추정의 평활 강도다. 기본값은 손목 기준·손바닥 크기 정규화 좌표를 가정하며,
    신호 스케일에 따라 GestureConfig에서 조절한다.
    """

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0.0 or d_cutoff <= 0.0:
            raise ValueError("min_cutoff and d_cutoff must be positive")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._x_prev: FloatArray | None = None
        self._dx_prev: FloatArray | None = None
        self._t_prev_ms: int | None = None

    def reset(self) -> None:
        """history를 비운다 — 추적 손실·프레임 공백에서 호출해 공백을 가로지르지 않는다."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev_ms = None

    def filter(self, value: npt.ArrayLike, timestamp_ms: int) -> FloatArray:
        """이 프레임의 평활화된 값을 반환한다.

        첫 표본(또는 reset 직후 첫 표본)은 그대로 통과한다. timestamp가 증가하지
        않으면(중복·역전 프레임) 상태를 갱신하지 않고 직전 출력을 돌려준다.
        """
        x = np.asarray(value, dtype=np.float64)
        if self._x_prev is None or self._dx_prev is None or self._t_prev_ms is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            self._t_prev_ms = timestamp_ms
            return x

        dt_ms = timestamp_ms - self._t_prev_ms
        if dt_ms <= 0:
            return self._x_prev
        dt_s = dt_ms / 1000.0

        dx = (x - self._x_prev) / dt_s
        a_d = _alpha(np.array(self._d_cutoff, dtype=np.float64), dt_s)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self._min_cutoff + self._beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt_s)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev_ms = timestamp_ms
        return x_hat

"""Cursor Control Mapper — 손 위치를 커서 좌표에 연속 매핑한다(README 6장).

이산 명령 경로(제스처 → Intent → Command)와 달리, 커서 조작은 매 프레임 손 위치를
커서 이동으로 바꾸는 연속 스트림이라 Fusion·Protocol·dedup을 경유하지 않는다. 대신
세 게이트를 통과할 때만 `InputSink.move_cursor`로 직접 이동을 낸다:

1. 시선이 노트북에 Lock돼 있다(`gaze_locked_to_laptop`). Lock이 풀리면 즉시 멈춘다.
2. 손이 추적되고 있다(`PointerSample.hand_detected`).
3. 이산 제스처가 진행 중이 아니다(`gesture_active`). ONSET~ENDING 동안은 커서가
   제스처에 우선권을 넘긴다(README 6장 커서↔제스처 분기).

게이트가 하나라도 막히면 추적 기준점을 버려, 다시 활성화될 때 손의 절대 위치로
커서가 순간이동(teleport)하지 않는다 — 재개 첫 프레임은 이동 없이 기준점만 다시
잡는다. 입력 좌표는 손 랜드마크의 이미지 정규화 좌표([0,1])이며, 그 좌표를 어디서
뽑을지(예: index fingertip)는 소스 어댑터의 몫이고 이 매퍼는 순수 매핑만 한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jarvis.runtime_protocol.adapters.windows import InputSink


@dataclass(frozen=True, slots=True)
class PointerSample:
    """한 프레임의 커서 입력 — 추적 대상 랜드마크의 이미지 정규화 위치.

    `x`/`y`는 [0,1] 범위(이미지 좌상단 원점). `hand_detected=False`이면 추적 손실
    프레임이라 위치는 무의미하다 — 매퍼는 이동을 내지 않고 기준점을 버린다.
    """

    x: float
    y: float
    hand_detected: bool


@dataclass(frozen=True, slots=True)
class PointerConfig:
    """커서 매핑 파라미터. 실기기 감도 튜닝 대상이며 변경은 설정이다."""

    screen_width: int
    screen_height: int
    sensitivity: float = 1.0
    """정규화 이동량에 곱하는 배율. 화면 대비 손 이동 폭을 조절한다."""

    smoothing: float = 0.35
    """EMA 계수 [0,1). 이전 위치를 이만큼 유지한다(0=평활 없음, 클수록 부드럽고 느림)."""

    deadzone_px: float = 1.0
    """이 픽셀 미만의 이동은 떨림으로 보고 무시한다(기준점은 계속 따라간다)."""

    max_step_px: int = 120
    """한 프레임 최대 이동. 추적 튐으로 인한 커서 점프를 막는 상한."""

    invert_x: bool = False
    """셀피 미러 뷰 보정. 손을 오른쪽으로 옮겼을 때 커서도 오른쪽으로 가게 하려면
    카메라 미러 여부에 맞춰 켠다(실기기 튜닝 대상)."""

    invert_y: bool = False

    def __post_init__(self) -> None:
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("screen dimensions must be positive")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be within [0, 1)")
        if not math.isfinite(self.sensitivity) or self.sensitivity <= 0.0:
            raise ValueError("sensitivity must be finite and positive")
        if self.deadzone_px < 0.0:
            raise ValueError("deadzone_px must be non-negative")
        if self.max_step_px <= 0:
            raise ValueError("max_step_px must be positive")


@dataclass(frozen=True, slots=True)
class PointerUpdate:
    """한 프레임 커서 매핑 결과(디버깅·모니터링용)."""

    moved: bool
    dx: int
    dy: int
    active: bool
    """이번 프레임에 커서 모드가 스트림을 소유했는지(세 게이트 통과 여부)."""

    reason: str


def _clamp(value: int, limit: int) -> int:
    return max(-limit, min(limit, value))


class CursorControlMapper:
    """손 위치 스트림을 상대 커서 이동으로 바꾸는 상태 유지 매퍼.

    상태는 마지막으로 본 (평활된) 정규화 위치뿐이다. 게이트가 막히면 이 기준점을
    버려 재개 시 순간이동을 막는다.
    """

    def __init__(self, sink: InputSink, config: PointerConfig) -> None:
        self._sink = sink
        self._config = config
        self._reference: tuple[float, float] | None = None

    @property
    def active(self) -> bool:
        """현재 커서 모드가 활성(기준점을 잡고 이동을 내는 중)인지."""
        return self._reference is not None

    def _deactivate(self, reason: str) -> PointerUpdate:
        self._reference = None
        return PointerUpdate(moved=False, dx=0, dy=0, active=False, reason=reason)

    def update(
        self,
        *,
        gaze_locked_to_laptop: bool,
        hand: PointerSample,
        gesture_active: bool,
    ) -> PointerUpdate:
        """한 프레임을 처리해 필요하면 커서를 이동시킨다.

        세 게이트(노트북 Lock·손 추적·제스처 비활성)를 통과할 때만 이동한다.
        활성화 첫 프레임은 기준점만 잡고 이동하지 않는다.
        """
        if not gaze_locked_to_laptop:
            return self._deactivate("gaze not locked to laptop")
        if not hand.hand_detected:
            return self._deactivate("hand not detected")
        if gesture_active:
            # 제스처에 우선권을 넘긴다. 커서 기준점을 버려, 제스처가 끝나 커서가
            # 재개될 때 손이 옮겨간 만큼 순간이동하지 않게 한다.
            return self._deactivate("gesture in progress")

        cfg = self._config
        if self._reference is None:
            # 재개(또는 최초) 첫 프레임: 기준점만 잡고 이동은 내지 않는다.
            self._reference = (hand.x, hand.y)
            return PointerUpdate(moved=False, dx=0, dy=0, active=True, reason="reacquired reference")

        prev_x, prev_y = self._reference
        # EMA 평활: 새 기준점은 이전과 현재의 가중 평균이다.
        smoothed_x = cfg.smoothing * prev_x + (1.0 - cfg.smoothing) * hand.x
        smoothed_y = cfg.smoothing * prev_y + (1.0 - cfg.smoothing) * hand.y

        delta_x_norm = smoothed_x - prev_x
        delta_y_norm = smoothed_y - prev_y
        self._reference = (smoothed_x, smoothed_y)

        sign_x = -1.0 if cfg.invert_x else 1.0
        sign_y = -1.0 if cfg.invert_y else 1.0
        dx_px = delta_x_norm * cfg.screen_width * cfg.sensitivity * sign_x
        dy_px = delta_y_norm * cfg.screen_height * cfg.sensitivity * sign_y

        if math.hypot(dx_px, dy_px) < cfg.deadzone_px:
            return PointerUpdate(moved=False, dx=0, dy=0, active=True, reason="within deadzone")

        dx = _clamp(int(round(dx_px)), cfg.max_step_px)
        dy = _clamp(int(round(dy_px)), cfg.max_step_px)
        if dx == 0 and dy == 0:
            return PointerUpdate(moved=False, dx=0, dy=0, active=True, reason="sub-pixel move")

        self._sink.move_cursor(dx, dy)
        return PointerUpdate(moved=True, dx=dx, dy=dy, active=True, reason="moved")

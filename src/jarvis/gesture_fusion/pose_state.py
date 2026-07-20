"""정적 자세 판정 → 시간축 동작 — 순수 상태기계(torch·카메라 무관).

프레임별 `PosePrediction`만으로는 동작이 정해지지 않는다. 같은 `pinch_index`라도 짧게
떼면 클릭, 유지하면 드래그다. 이 모듈이 그 시간 구조를 담당한다.

설계 규칙 세 가지:

1. **진입은 느리게, 이탈은 빠르게.** 자세가 `DWELL_MS` 연속 유지돼야 상태로 진입한다.
   전이 프레임은 짧아서(보통 100~200ms) 여기서 걸러진다. 반대로 `none`이 몇 프레임
   들어와도 바로 끊지 않는다(`RELEASE_FRAMES`) — 분류기의 놓침(실측 15.4%)이 조작
   중단으로 이어지면 안 된다.

2. **믿을 수 없는 판정은 상태를 건드리지 않는다.** `trusted=False`(기울기 게이트에
   걸린 프레임)는 상태를 바꾸지도, 유지를 끊지도 않고 그냥 흘려보낸다.

3. **`none`은 자세 이력을 지우지 않는다.** 주먹에서 손바닥으로 갈 때 중간의 어중간하게
   펴진 상태가 `none`으로 분류되기 때문에, `fist → open_palm`을 인접 상태로 보면 절대
   성립하지 않는다. 그래서 "마지막 명령 자세"를 따로 기억하고 `none`은 그걸 덮어쓰지
   않는다 — 전이 판정이 중간의 빈 구간을 건너뛴다.

여기서 쓰는 임계는 전부 **시간**이다. 이 프로젝트에서 반복적으로 실패한 것은 기하학적
임계(palm_scale로 나눈 거리 등)였고, 그것들은 손 각도·거리에 따라 값이 8.85배까지
흔들렸다. 시간 임계는 투영에 영향받지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.pose_protocol import NONE_POSE, PosePrediction

FloatArray = npt.NDArray[np.float64]

INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP = 5, 8, 9, 12

# 자세별 진입 유지 시간(ms). 클릭류는 빠른 반응이 중요해 짧고, 연속 동작인 스크롤은
# 오발동 비용이 커서 길다(우클릭 준비 중 손가락이 펴진 구간을 확실히 넘기려면 필요).
DWELL_MS: dict[str, int] = {
    "index_point": 150,
    "pinch_index": 120,
    "pinch_middle": 120,
    "two_fingers": 300,
    "open_palm": 200,
    "fist": 200,
}
DEFAULT_DWELL_MS = 200

# 이 프레임 수만큼 연속으로 자세가 사라져야 상태를 해제한다. 분류기가 정상 자세를
# none으로 흘리는 비율이 15.4%라, 1프레임만에 끊으면 조작이 계속 끊긴다.
RELEASE_FRAMES = 3

# 핀치를 이보다 오래 쥐고 있으면 클릭이 아니라 드래그로 본다.
CLICK_MAX_MS = 400
# `fist → open_palm` 전이로 인정하는 최대 간격. 중간의 `none` 구간을 건너뛴다.
TRANSITION_WINDOW_MS = 800
# 스크롤 방향을 인정할 최소 수직성(|dy| / 길이). 손가락이 옆을 가리키면 위아래를
# 지어내지 않는다 — 0.5는 수평에서 30° 이상 기울어야 방향을 인정한다는 뜻이다.
MIN_VERTICALITY = 0.5


@dataclass(frozen=True, slots=True)
class PoseEvent:
    """상태기계가 내보내는 동작. `value`는 스크롤에서만 의미가 있다(부호=방향)."""

    kind: str
    timestamp_ms: int
    value: float = 0.0


def pointing_direction(landmarks: FloatArray) -> tuple[float, float] | None:
    """검지·중지가 **가리키는 방향** 단위벡터. 손의 이동이 아니라 자세에서 나온다.

    스크롤은 손을 움직이는 것이 아니라 두 손가락이 위/아래를 가리키는 동안 계속
    일어난다. 그래서 방향은 MCP→끝 벡터로 구한다 — 손을 멈춰도 방향은 유지되고,
    이동 추적이 필요 없어 카메라 흔들림·손목 미세 이동에 영향받지 않는다.

    이미지 좌표는 y축이 아래로 향하므로, 화면 위쪽을 가리키면 dy < 0이다.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= MIDDLE_TIP:
        return None
    vectors = [points[INDEX_TIP] - points[INDEX_MCP], points[MIDDLE_TIP] - points[MIDDLE_MCP]]
    mean = np.mean(vectors, axis=0)[:2]
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return float(mean[0] / norm), float(mean[1] / norm)


@dataclass
class PoseStateMachine:
    """자세 판정을 받아 동작 이벤트를 낸다. 프레임마다 `update()`를 부른다."""

    dwell_ms: dict[str, int] = field(default_factory=lambda: dict(DWELL_MS))
    release_frames: int = RELEASE_FRAMES
    click_max_ms: int = CLICK_MAX_MS
    transition_window_ms: int = TRANSITION_WINDOW_MS

    # 확정된 현재 상태(진입 조건을 통과한 자세). 없으면 "".
    state: str = ""
    _state_since: int = 0
    # 진입 대기 중인 후보
    _pending: str = ""
    _pending_since: int = 0
    _missing: int = 0
    # `none`이 덮어쓰지 않는 "마지막 명령 자세" — 전이 판정이 빈 구간을 건너뛴다.
    _last_pose: str = ""
    _last_pose_end: int = 0
    _dragging: bool = False

    def reset(self) -> None:
        """추적 손실 등으로 이력을 신뢰할 수 없을 때 — 상태를 지어내지 않는다."""
        self.state = ""
        self._pending = ""
        self._missing = 0
        self._last_pose = ""
        self._dragging = False

    def update(
        self,
        prediction: PosePrediction,
        timestamp_ms: int,
        landmarks: FloatArray | None = None,
    ) -> list[PoseEvent]:
        """한 프레임을 처리해 발생한 이벤트를 돌려준다(없으면 빈 리스트)."""
        # 규칙 2: 믿을 수 없는 판정은 상태를 바꾸지도 끊지도 않는다.
        if not prediction.trusted:
            return self._continuous(timestamp_ms, landmarks)

        label = prediction.label
        if label in ("", NONE_POSE):
            return self._absent(timestamp_ms, landmarks)
        return self._present(label, timestamp_ms, landmarks)

    # --- 내부 ---

    def _present(
        self, label: str, timestamp_ms: int, landmarks: FloatArray | None
    ) -> list[PoseEvent]:
        self._missing = 0
        if label == self.state:
            return self._continuous(timestamp_ms, landmarks)

        if label != self._pending:
            self._pending, self._pending_since = label, timestamp_ms
            return self._continuous(timestamp_ms, landmarks)

        dwell = self.dwell_ms.get(label, DEFAULT_DWELL_MS)
        if timestamp_ms - self._pending_since < dwell:
            return self._continuous(timestamp_ms, landmarks)

        events = self._leave(timestamp_ms)
        events.extend(self._enter(label, timestamp_ms))
        return events

    def _absent(self, timestamp_ms: int, landmarks: FloatArray | None) -> list[PoseEvent]:
        """`none`/미판정 프레임. 규칙 1(관용)과 규칙 3(이력 보존)이 여기 걸린다."""
        self._pending = ""
        if not self.state:
            return []
        self._missing += 1
        if self._missing < self.release_frames:
            return self._continuous(timestamp_ms, landmarks)
        return self._leave(timestamp_ms)

    def _enter(self, label: str, timestamp_ms: int) -> list[PoseEvent]:
        events: list[PoseEvent] = []
        # 규칙 3: 중간의 none 구간을 건너뛰고 직전 **명령 자세**와 비교한다.
        if (
            label == "open_palm"
            and self._last_pose == "fist"
            and timestamp_ms - self._last_pose_end <= self.transition_window_ms
        ):
            events.append(PoseEvent("media_toggle", timestamp_ms))
        self.state, self._state_since = label, timestamp_ms
        self._pending, self._missing = "", 0
        return events

    def _leave(self, timestamp_ms: int) -> list[PoseEvent]:
        events: list[PoseEvent] = []
        held = timestamp_ms - self._state_since
        if self.state == "pinch_index":
            if self._dragging:
                events.append(PoseEvent("drag_end", timestamp_ms))
            elif held <= self.click_max_ms:
                events.append(PoseEvent("click", timestamp_ms))
        elif self.state == "pinch_middle" and held <= self.click_max_ms:
            events.append(PoseEvent("right_click", timestamp_ms))
        if self.state:
            self._last_pose, self._last_pose_end = self.state, timestamp_ms
        self.state, self._dragging, self._missing = "", False, 0
        return events

    def _continuous(
        self, timestamp_ms: int, landmarks: FloatArray | None
    ) -> list[PoseEvent]:
        """상태를 유지하는 동안 계속 나가는 이벤트(스크롤·드래그 승격)."""
        events: list[PoseEvent] = []
        held = timestamp_ms - self._state_since
        if self.state == "pinch_index" and not self._dragging and held > self.click_max_ms:
            # 오래 쥐고 있으면 클릭이 아니라 드래그다. 진입 시점으로 소급해 알린다.
            self._dragging = True
            events.append(PoseEvent("drag_start", self._state_since))
        elif self.state == "two_fingers" and landmarks is not None:
            direction = pointing_direction(landmarks)
            if direction is not None and abs(direction[1]) >= MIN_VERTICALITY:
                # 화면 y는 아래로 증가하므로 부호를 뒤집어 위쪽을 +로 만든다.
                events.append(PoseEvent("scroll", timestamp_ms, -direction[1]))
        return events

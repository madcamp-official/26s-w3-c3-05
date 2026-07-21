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
from jarvis.gesture_fusion.smoothing import OneEuroFilter

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
# 직전 클릭과 이 간격 안에 다음 클릭이 나오면 더블클릭으로 승격한다(마우스와 동일 UX).
DOUBLE_CLICK_MS = 400
# `fist → open_palm` 전이로 인정하는 최대 간격. 중간의 `none` 구간을 건너뛴다.
TRANSITION_WINDOW_MS = 800
# 스크롤 방향을 인정할 최소 수직성(|dy| / 길이). 손가락이 옆을 가리키면 위아래를
# 지어내지 않는다 — 0.5는 수평에서 30° 이상 기울어야 방향을 인정한다는 뜻이다.
MIN_VERTICALITY = 0.5

# 커서 이동: index_point(이동) 또는 pinch_index(드래그) 상태에서 손 이동을 커서로 옮긴다.
# 좌표를 1:1로 대응시키지 않고, 마우스처럼 손 이동 **델타**에 이득을 곱한다.
CURSOR_POSES = ("index_point", "pinch_index")
CURSOR_BASE_GAIN = 480.0      # 손 이동(팜 단위) → 픽셀 기본 배율. 낮을수록 미세조정 여지↑
CURSOR_ACCEL_GAIN = 1.4       # 속도가 빠를수록 이득이 커진다(정밀↔빠른 이동 양립). 클수록 공격적
CURSOR_MAX_ACCEL = 4.0        # 이득 상한(급격한 튐 방지). 클수록 고속 큰 이동이 시원
CURSOR_DEADZONE = 0.007       # 이보다 작은 손 떨림은 무시(팜 단위)
CURSOR_MAX_STEP_PX = 220      # 한 프레임 최대 이동(검출 튐이 커서를 순간이동시키지 않게)
CURSOR_INVERT_X = True        # 거울 뷰가 아닌 실제 손 기준 — 왼손 이동 = 커서 왼쪽
CURSOR_Y_GAIN_SCALE = 0.5     # y축 이동 감도 배율. x 대비 세로가 과민해 절반으로 낮춘다

# 위 gain은 "이 해상도의 화면에서" px가 되도록 튜닝됐다(기준 디스플레이). _cursor_move가
# 내는 delta는 이 기준 화면 기준의 px이며, 실제 방출 직전 pose_control이 실기기 해상도
# 비율(actual/reference)로 스케일해 **화면 대비 이동 비율**을 기기 무관하게 맞춘다. 그래야
# 해상도가 다른 노트북에서도 같은 손 이동이 화면의 같은 비율을 가로질러 체감 속도가 같다.
# 값은 이 gain을 튜닝한 기기(1440×900 macOS 논리 해상도, CGEvent 좌표계)다. 이 기기에서는
# actual==reference라 스케일이 1.0 → 이동량이 종전과 완전히 동일하다.
CURSOR_REFERENCE_WIDTH = 1440
CURSOR_REFERENCE_HEIGHT = 900

# soft deadzone: 데드존을 하드컷(distance<dz면 0)하지 않고, 넘는 순간 이동량이 0부터
# 연속으로 살아나게 한다 — `distance - deadzone`만큼만 이동에 반영해 경계의 급점프를 없앤다.
# palm_scale 평활: 커서 speed 분모(raw palm_scale)의 프레임 지터를 One-Euro로 줄인다
# (features 모델 경로와 동일 파라미터). 정지 잡음 자체엔 효과가 작지만, 이동 중 카메라
# 거리 변화·손 각도에 따른 palm_scale 흔들림이 이동량으로 새는 것을 완화한다.
CURSOR_PALM_SMOOTHING_MIN_CUTOFF = 1.0
CURSOR_PALM_SMOOTHING_BETA = 0.0
CURSOR_PALM_SMOOTHING_D_CUTOFF = 1.0


def _make_cursor_palm_smoother() -> OneEuroFilter:
    return OneEuroFilter(
        min_cutoff=CURSOR_PALM_SMOOTHING_MIN_CUTOFF,
        beta=CURSOR_PALM_SMOOTHING_BETA,
        d_cutoff=CURSOR_PALM_SMOOTHING_D_CUTOFF,
    )


@dataclass(frozen=True, slots=True)
class PoseEvent:
    """상태기계가 내보내는 동작.

    `value`는 스크롤에서 방향(부호). `delta`는 move에서 커서 이동 픽셀 (dx, dy).
    """

    kind: str
    timestamp_ms: int
    value: float = 0.0
    delta: tuple[float, float] = (0.0, 0.0)


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
    double_click_ms: int = DOUBLE_CLICK_MS
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
    # 직전에 확정된 클릭 시각 — 다음 클릭이 double_click_ms 안이면 더블클릭으로 승격한다.
    # 첫 클릭이 오인되지 않도록 "아주 오래전"으로 시작한다(0은 작은 timestamp에서 위험).
    _last_click_ms: int = -1_000_000
    _dragging: bool = False
    # 커서 이동 참조점(이미지 좌표)과 시각 — 델타 계산용. 상태 진입 때 초기화한다.
    _cursor_ref: tuple[float, float] | None = None
    _cursor_ref_ms: int = 0
    # 커서 speed 분모로 쓰는 palm_scale의 One-Euro 평활기(raw palm_scale 지터 완화).
    _palm_smoother: OneEuroFilter = field(default_factory=_make_cursor_palm_smoother)

    def reset(self) -> None:
        """추적 손실 등으로 이력을 신뢰할 수 없을 때 — 상태를 지어내지 않는다."""
        self.state = ""
        self._pending = ""
        self._missing = 0
        self._last_pose = ""
        self._last_click_ms = -1_000_000
        self._dragging = False
        self._palm_smoother.reset()

    def update(
        self,
        prediction: PosePrediction,
        timestamp_ms: int,
        landmarks: FloatArray | None = None,
        reference_point: tuple[float, float] | None = None,
        palm_scale: float | None = None,
    ) -> list[PoseEvent]:
        """한 프레임을 처리해 발생한 이벤트를 돌려준다(없으면 빈 리스트).

        `reference_point`는 커서 이동 기준이 되는 이미지 좌표(손 전체 위치, 예: 손목).
        정규화 좌표는 손목이 원점이라 손 전체 이동이 사라지므로, 이동은 이 값으로 잰다.
        `palm_scale`로 나눠 카메라 거리에 무관하게 만든다(멀든 가깝든 같은 손 이동 = 같은
        커서 이동).
        """
        # palm_scale(커서 speed 분모)을 One-Euro로 평활한다. 손 손실 프레임(palm 없음)엔
        # 평활기를 리셋해 재개 시 옛 상태가 새 값에 섞이지 않게 한다.
        if palm_scale is not None and palm_scale > 0.0:
            palm_scale = float(self._palm_smoother.filter(palm_scale, timestamp_ms))
        else:
            self._palm_smoother.reset()
        self._cursor_ctx = (reference_point, palm_scale)
        # 규칙 2: 믿을 수 없는 판정은 상태를 바꾸지도 끊지도 않는다.
        if not prediction.trusted:
            return self._continuous(timestamp_ms, landmarks)

        label = prediction.label
        if label in ("", NONE_POSE):
            return self._absent(timestamp_ms, landmarks)
        return self._present(label, timestamp_ms, landmarks)

    def _cursor_move(self, timestamp_ms: int) -> PoseEvent | None:
        """참조점 델타를 커서 이동으로 바꾼다 — 1:1 대응이 아니라 마우스식 상대 이동.

        속도가 빠를수록 이득이 커져(포인터 가속) 큰 이동과 정밀 조작을 양립시킨다.
        검출 튐이 커서를 순간이동시키지 않도록 프레임당 이동을 제한한다.
        """
        reference_point, palm_scale = getattr(self, "_cursor_ctx", (None, None))
        if reference_point is None or not palm_scale or palm_scale <= 0.0:
            self._cursor_ref = None
            return None
        if self._cursor_ref is None:
            self._cursor_ref, self._cursor_ref_ms = reference_point, timestamp_ms
            return None
        # 팜 단위 이동(카메라 거리 독립)
        dx = (reference_point[0] - self._cursor_ref[0]) / palm_scale
        dy = (reference_point[1] - self._cursor_ref[1]) / palm_scale
        self._cursor_ref, prev_ms = reference_point, self._cursor_ref_ms
        self._cursor_ref_ms = timestamp_ms
        distance = math.hypot(dx, dy)
        if distance <= CURSOR_DEADZONE:  # 정지 시 손 떨림 무시
            return None
        # soft deadzone: 데드존만큼 뺀 이동량만 반영해 경계에서 0부터 연속으로 살아나게
        # 한다(하드컷의 급점프 제거). 방향은 유지하고 크기만 (distance-dz)/distance로 줄인다.
        soft = (distance - CURSOR_DEADZONE) / distance
        dx, dy = dx * soft, dy * soft
        dt_s = max((timestamp_ms - prev_ms) / 1000.0, 1e-3)
        speed = distance / dt_s
        gain = CURSOR_BASE_GAIN * min(CURSOR_MAX_ACCEL, 1.0 + CURSOR_ACCEL_GAIN * speed)
        px = (-dx if CURSOR_INVERT_X else dx) * gain
        py = dy * gain * CURSOR_Y_GAIN_SCALE
        step = math.hypot(px, py)
        if step > CURSOR_MAX_STEP_PX:  # 검출 튐 방지
            scale = CURSOR_MAX_STEP_PX / step
            px, py = px * scale, py * scale
        return PoseEvent("move", timestamp_ms, 0.0, (px, py))

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
                # 직전 클릭과 간격이 짧으면 더블클릭으로 승격한다. 첫 클릭은 이미
                # 나갔지만(마우스와 동일) 두 번째를 double_click으로 낸다.
                if timestamp_ms - self._last_click_ms <= self.double_click_ms:
                    events.append(PoseEvent("double_click", timestamp_ms))
                    self._last_click_ms = -1_000_000  # 3연속 핀치가 또 더블클릭 되지 않게 초기화
                else:
                    events.append(PoseEvent("click", timestamp_ms))
                    self._last_click_ms = timestamp_ms
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
        # 커서 이동: index_point(이동)·pinch_index(드래그) 상태에서 손 이동을 옮긴다.
        # 드래그도 여기서 커서가 따라 움직인다 — pinch_index가 CURSOR_POSES에 있다.
        if self.state in CURSOR_POSES:
            move = self._cursor_move(timestamp_ms)
            if move is not None:
                events.append(move)
        else:
            self._cursor_ref = None  # 이동 자세를 벗어나면 참조점을 버린다
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

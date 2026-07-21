"""자세 상태기계 — 시간 구조가 동작을 정한다.

여기서 지키려는 것은 실사용에서 발견된 세 가지다:
1. 전이 프레임이 명령을 발동시키면 안 된다(우클릭 준비 중 스크롤이 튀던 문제).
2. 분류기가 정상 자세를 none으로 흘려도(실측 15.4%) 조작이 끊기면 안 된다.
3. 주먹→손바닥 전이 중간은 none으로 분류된다 — 인접 상태로 보면 절대 성립 안 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.gesture_fusion.pose_protocol import NONE_POSE, PosePrediction
from jarvis.gesture_fusion.pose_state import (
    PoseStateMachine,
    pointing_direction,
)


def _pose(label: str, *, trusted: bool = True) -> PosePrediction:
    return PosePrediction(
        label=label,
        confidence=0.95,
        trusted=trusted,
        reason="" if trusted else "기울기 초과",
    )


def _hand(dx: float, dy: float) -> np.ndarray:
    """검지·중지가 (dx, dy) 방향을 가리키는 랜드마크.

    방향은 (dx, dy)를 따르되, 손가락 길이는 편 손가락에 해당하는 현실적 값(palm_scale
    정규화 0.7)으로 뻗어 스크롤 폄 게이트(MIN_FINGER_EXTENSION)를 통과하게 한다.
    """
    d = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(d))
    unit = d / norm if norm > 0 else d
    tip = unit * 0.7  # 편 손가락 길이(palm 단위)
    points = np.zeros((21, 2), dtype=np.float64)
    points[5], points[9] = [0.0, 0.0], [0.1, 0.0]              # MCP
    points[8], points[12] = tip, np.array([0.1, 0.0]) + tip     # 끝
    return points


def _feed(machine: PoseStateMachine, label: str, *, ms: int, start: int = 0, step: int = 33,
          landmarks: np.ndarray | None = None) -> list:
    events = []
    t = start
    while t <= start + ms:
        events.extend(machine.update(_pose(label), t, landmarks))
        t += step
    return events


# --- 규칙 1: 진입은 느리게(전이 차단) ---

def test_short_pose_does_not_trigger() -> None:
    """전이 중 스쳐 지나가는 자세는 상태가 되지 않는다(스크롤 오발동 차단)."""
    machine = PoseStateMachine()
    _feed(machine, "two_fingers", ms=80, landmarks=_hand(0.0, -0.2))  # dwell 120ms 미만
    assert machine.state == ""


def test_held_pose_enters_state() -> None:
    machine = PoseStateMachine()
    _feed(machine, "two_fingers", ms=400, landmarks=_hand(0.0, -0.2))
    assert machine.state == "two_fingers"


# --- 규칙 2: 이탈은 관용적으로(놓침 흡수) ---

def test_brief_none_does_not_break_state() -> None:
    """분류기가 한두 프레임 none으로 흘려도 조작이 끊기면 안 된다."""
    machine = PoseStateMachine()
    _feed(machine, "index_point", ms=300)
    assert machine.state == "index_point"
    for t in (400, 433):
        machine.update(_pose(NONE_POSE), t)
    assert machine.state == "index_point"


def test_sustained_none_releases_state() -> None:
    machine = PoseStateMachine()
    _feed(machine, "index_point", ms=300)
    for t in (400, 433, 466, 500):
        machine.update(_pose(NONE_POSE), t)
    assert machine.state == ""


# --- 규칙 3: none이 자세 이력을 지우지 않는다 ---

def test_fist_to_open_palm_through_none_fires_media_toggle() -> None:
    """주먹→손바닥 전이 중간은 none이다. 인접 상태로 보면 절대 성립하지 않는다."""
    machine = PoseStateMachine()
    _feed(machine, "fist", ms=300)
    assert machine.state == "fist"
    # 어중간하게 펴진 중간 구간 — none으로 분류된다
    for t in (400, 433, 466, 500, 533):
        machine.update(_pose(NONE_POSE), t)
    assert machine.state == ""
    events = _feed(machine, "open_palm", ms=300, start=566)
    assert [e.kind for e in events] == ["media_toggle"]


def test_open_palm_alone_does_not_fire_media_toggle() -> None:
    """주먹을 거치지 않은 손바닥은 아무것도 아니다 — 손만 펴도 재생이 토글되면 안 된다."""
    machine = PoseStateMachine()
    events = _feed(machine, "open_palm", ms=400)
    assert events == []


def test_media_toggle_expires_after_window() -> None:
    """한참 전의 주먹은 전이로 치지 않는다(우연한 순서에 반응하지 않는다)."""
    machine = PoseStateMachine()
    _feed(machine, "fist", ms=300)
    for t in range(400, 1800, 33):
        machine.update(_pose(NONE_POSE), t)
    events = _feed(machine, "open_palm", ms=300, start=1800)
    assert events == []


# --- 클릭 / 드래그 ---

def test_short_pinch_is_click() -> None:
    machine = PoseStateMachine()
    _feed(machine, "pinch_index", ms=200)
    events = [e for t in (300, 333, 366, 400)
              for e in machine.update(_pose(NONE_POSE), t)]
    assert [e.kind for e in events] == ["click"]


def _pinch_click(machine: PoseStateMachine, *, start: int) -> list:
    """핀치를 짧게 쥐었다 떼는 한 사이클 — click/double_click 이벤트를 낸다."""
    events = _feed(machine, "pinch_index", ms=150, start=start)  # dwell(120ms) 통과
    t = start + 150 + 33
    for _ in range(4):  # RELEASE_FRAMES(3) 넘겨 확실히 이탈
        events.extend(machine.update(_pose(NONE_POSE), t))
        t += 33
    return events


def test_two_quick_pinches_are_double_click() -> None:
    """두 클릭 간격이 double_click_ms 안이면 두 번째가 더블클릭으로 승격된다."""
    machine = PoseStateMachine()
    first = _pinch_click(machine, start=0)
    second = _pinch_click(machine, start=300)  # 첫 클릭에서 ~300ms 뒤
    assert [e.kind for e in first] == ["click"]
    assert [e.kind for e in second] == ["double_click"]


def test_slow_second_pinch_is_plain_click() -> None:
    """간격이 double_click_ms를 넘으면 둘 다 그냥 클릭이다."""
    machine = PoseStateMachine()
    first = _pinch_click(machine, start=0)
    second = _pinch_click(machine, start=1000)  # 충분히 늦음
    assert [e.kind for e in first] == ["click"]
    assert [e.kind for e in second] == ["click"]


def test_triple_pinch_does_not_chain_double_clicks() -> None:
    """세 번 연속 핀치는 (클릭, 더블클릭, 클릭) — 더블클릭이 연쇄되지 않는다."""
    machine = PoseStateMachine()
    kinds = [
        e.kind
        for start in (0, 300, 600)
        for e in _pinch_click(machine, start=start)
    ]
    assert kinds == ["click", "double_click", "click"]


def test_long_pinch_becomes_drag() -> None:
    """오래 쥐면 드래그다. 시작 시점은 핀치 진입 순간으로 소급된다."""
    machine = PoseStateMachine()
    events = _feed(machine, "pinch_index", ms=700)
    starts = [e for e in events if e.kind == "drag_start"]
    assert len(starts) == 1
    assert starts[0].timestamp_ms < 700  # 소급 적용

    release = [e for t in (800, 833, 866, 900)
               for e in machine.update(_pose(NONE_POSE), t)]
    assert [e.kind for e in release] == ["drag_end"]


def test_pinch_middle_is_right_click() -> None:
    machine = PoseStateMachine()
    _feed(machine, "pinch_middle", ms=200)
    events = [e for t in (300, 333, 366, 400)
              for e in machine.update(_pose(NONE_POSE), t)]
    assert [e.kind for e in events] == ["right_click"]


# --- 스크롤: 가리키는 방향 ---

def test_scroll_follows_pointing_direction_not_movement() -> None:
    """손을 움직이지 않아도, 가리키는 방향으로 계속 스크롤된다."""
    machine = PoseStateMachine()
    up = _feed(machine, "two_fingers", ms=500, landmarks=_hand(0.0, -0.2))
    scrolls = [e for e in up if e.kind == "scroll"]
    assert scrolls and all(e.value > 0 for e in scrolls)  # 화면 위쪽 = 양수

    machine.reset()
    down = _feed(machine, "two_fingers", ms=500, landmarks=_hand(0.0, 0.2))
    assert all(e.value < 0 for e in down if e.kind == "scroll")


def test_sideways_pointing_does_not_scroll() -> None:
    """옆을 가리키면 위아래를 지어내지 않는다."""
    machine = PoseStateMachine()
    events = _feed(machine, "two_fingers", ms=500, landmarks=_hand(0.2, 0.02))
    assert [e for e in events if e.kind == "scroll"] == []


def test_pointing_direction_is_unit_and_none_when_degenerate() -> None:
    direction = pointing_direction(_hand(0.0, -0.2))
    assert direction == pytest.approx((0.0, -1.0))
    assert pointing_direction(np.zeros((21, 2))) is None


# --- 규칙: 신뢰 못 하는 판정은 무시 ---

def test_untrusted_prediction_neither_enters_nor_breaks() -> None:
    """기울기 게이트에 걸린 프레임은 상태를 바꾸지도, 유지를 끊지도 않는다."""
    machine = PoseStateMachine()
    for t in range(0, 500, 33):
        machine.update(_pose("two_fingers", trusted=False), t)
    assert machine.state == ""

    _feed(machine, "index_point", ms=300, start=500)
    assert machine.state == "index_point"
    for t in range(900, 1200, 33):
        machine.update(_pose("open_palm", trusted=False), t)
    assert machine.state == "index_point"


# --- 커서 이동 ---

def _feed_cursor(machine, label, ref, *, ms, start=0, step=33, palm=0.15):
    """참조점을 고정하거나 이동시키며 자세를 유지한다. ref는 (x0,y0)→(x1,y1) 또는 고정점."""
    events, t = [], start
    while t <= start + ms:
        frac = (t - start) / max(ms, 1)
        if isinstance(ref[0], tuple):
            point = (ref[0][0] + (ref[1][0] - ref[0][0]) * frac,
                     ref[0][1] + (ref[1][1] - ref[0][1]) * frac)
        else:
            point = ref
        events.extend(machine.update(_pose(label), t, reference_point=point, palm_scale=palm))
        t += step
    return events


def test_index_point_moves_cursor() -> None:
    """검지 폄 상태에서 손을 옮기면 커서 이동 이벤트가 나온다."""
    machine = PoseStateMachine()
    _feed_cursor(machine, "index_point", (0.5, 0.5), ms=200)  # 진입
    events = _feed_cursor(machine, "index_point", ((0.5, 0.5), (0.7, 0.5)), ms=300, start=233)
    moves = [e for e in events if e.kind == "move"]
    assert moves and all(e.delta[0] != 0 for e in moves)


def test_pinch_drag_also_moves_cursor() -> None:
    """드래그(핀치 유지) 중에도 커서가 따라 움직인다."""
    machine = PoseStateMachine()
    events = _feed_cursor(machine, "pinch_index", ((0.4, 0.4), (0.6, 0.6)), ms=700)
    assert any(e.kind == "move" for e in events)
    assert any(e.kind == "drag_start" for e in events)


def test_stationary_hand_does_not_move_cursor() -> None:
    """손이 가만히 있으면(참조점 고정) 커서가 떨지 않는다 — 데드존."""
    machine = PoseStateMachine()
    _feed_cursor(machine, "index_point", (0.5, 0.5), ms=200)
    events = _feed_cursor(machine, "index_point", (0.5, 0.5), ms=300, start=233)
    assert [e for e in events if e.kind == "move"] == []


def test_open_palm_does_not_move_cursor() -> None:
    """이동 자세가 아니면 손을 옮겨도 커서는 그대로다."""
    machine = PoseStateMachine()
    events = _feed_cursor(machine, "open_palm", ((0.4, 0.4), (0.7, 0.7)), ms=400)
    assert [e for e in events if e.kind == "move"] == []


def test_sustained_untrusted_stops_cursor_but_keeps_state() -> None:
    """기울기 초과가 관용 프레임을 넘겨 지속되면 커서가 멈추되 상태는 유지된다.

    각도를 다시 낮추면 dwell 재대기 없이 즉시 이동이 재개돼야 한다(상태 보존의 목적).
    """
    machine = PoseStateMachine()
    _feed_cursor(machine, "index_point", (0.5, 0.5), ms=200)  # 진입
    assert machine.state == "index_point"

    # 손은 계속 이동하지만 기울기 초과(untrusted)가 지속 — 관용(3프레임) 이후엔 정지.
    t, moves = 233, []
    for k in range(8):
        point = (0.5 + 0.01 * k, 0.5)
        out = machine.update(_pose("index_point", trusted=False), t, reference_point=point, palm_scale=0.15)
        moves.append(any(e.kind == "move" for e in out))
        t += 33
    assert any(moves[:3]), "관용 구간에는 이동이 유지돼야 한다"
    assert not any(moves[machine.untrusted_grace_frames + 1:]), "관용을 넘기면 이동이 멈춰야 한다"
    assert machine.state == "index_point", "정지해도 상태는 유지된다"

    # 각도 회복(trusted) — dwell 재대기 없이 즉시 재개.
    resumed = _feed_cursor(machine, "index_point", ((0.6, 0.5), (0.8, 0.5)), ms=150, start=t)
    assert any(e.kind == "move" for e in resumed), "각도 회복 시 dwell 없이 즉시 재개"

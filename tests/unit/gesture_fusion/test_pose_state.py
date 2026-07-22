"""자세 상태기계 — 시간 구조가 동작을 정한다.

여기서 지키려는 것은 실사용에서 발견된 세 가지다:
1. 전이 프레임이 명령을 발동시키면 안 된다(우클릭 준비 중 스크롤이 튀던 문제).
2. 분류기가 정상 자세를 none으로 흘려도(실측 15.4%) 조작이 끊기면 안 된다.
3. 주먹→손바닥 전이 중간은 none으로 분류된다 — 인접 상태로 보면 절대 성립 안 한다.
"""

from __future__ import annotations

import math

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
    """검지·중지가 (dx, dy) 방향을 가리키는, 곧게 편 손가락 랜드마크.

    방향은 (dx, dy)를 따르되, 두 손가락을 곧게(MCP·PIP·DIP·끝을 일직선으로) 펴
    직진도 게이트(TWO_FINGER_STRAIGHTNESS_MIN)를 통과하게 한다. PIP·DIP를 MCP→끝
    선분 위 등간격에 두므로 straightness=1.0이 되어 경계값과 무관하게 안정적이다.
    """
    d = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(d))
    unit = d / norm if norm > 0 else d
    tip = unit * 0.7  # 편 손가락 길이(palm 단위)
    points = np.zeros((21, 2), dtype=np.float64)
    for mcp, pip, dip, end, base in ((5, 6, 7, 8, [0.0, 0.0]), (9, 10, 11, 12, [0.1, 0.0])):
        origin = np.array(base, dtype=np.float64)
        points[mcp] = origin
        points[pip] = origin + tip * (1.0 / 3.0)  # MCP→끝 선분 위 등간격 → 곧음(straightness 1.0)
        points[dip] = origin + tip * (2.0 / 3.0)
        points[end] = origin + tip
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


# --- 미션 컨트롤(open_palm) / OK(재생·일시정지) ---

def test_ok_pose_fires_play_pause() -> None:
    """OK 사인(ok) 진입은 재생/일시정지를 한 번 발화한다(주먹 자리에 넣은 새 자세)."""
    machine = PoseStateMachine()
    events = _feed(machine, "ok", ms=300)
    assert [e.kind for e in events] == ["play_pause"]


def test_any_pose_to_open_palm_fires_media_toggle() -> None:
    """주먹뿐 아니라 다른 아무 명령 자세 → 보(open_palm) 전이도 미션 컨트롤을 발화한다."""
    machine = PoseStateMachine()
    _feed(machine, "index_point", ms=300)  # 주먹이 아닌 다른 자세
    for t in (400, 433, 466):  # RELEASE_FRAMES 넘겨 index_point 이탈
        machine.update(_pose(NONE_POSE), t)
    assert machine.state == ""
    events = _feed(machine, "open_palm", ms=300, start=500)
    assert [e.kind for e in events] == ["media_toggle"]


def test_open_palm_from_none_fires_media_toggle() -> None:
    """맨손(none)에서 바로 보를 펴도 미션 컨트롤이 발화한다(none에서 가도 인정)."""
    machine = PoseStateMachine()
    events = _feed(machine, "open_palm", ms=400)
    assert [e.kind for e in events] == ["media_toggle"]


# --- 클릭 / 드래그 ---

def _kv(events: list) -> list:
    """(kind, click_state) 목록 — mouse_down/up의 clickState까지 검증한다."""
    return [(e.kind, int(e.value)) for e in events]


def test_short_pinch_emits_down_then_up() -> None:
    """짧은 핀치: 진입에서 버튼 down, 릴리즈에서 up (둘 다 clickState=1)."""
    machine = PoseStateMachine()
    down = _feed(machine, "pinch_index", ms=200)
    up = [e for t in (300, 333, 366, 400)
          for e in machine.update(_pose(NONE_POSE), t)]
    assert _kv(down) == [("mouse_down", 1)]
    assert _kv(up) == [("mouse_up", 1)]


def _pinch_click(machine: PoseStateMachine, *, start: int) -> list:
    """핀치를 짧게 쥐었다 떼는 한 사이클 — mouse_down→mouse_up 한 쌍을 낸다."""
    events = _feed(machine, "pinch_index", ms=150, start=start)  # dwell 통과 + 진입 down
    t = start + 150 + 33
    for _ in range(4):  # RELEASE_FRAMES(3) 넘겨 확실히 이탈 → up
        events.extend(machine.update(_pose(NONE_POSE), t))
        t += 33
    return events


def test_two_quick_pinches_are_double_click() -> None:
    """두 핀치 간격이 double_click_ms 안이면 두 번째 down/up이 clickState=2다."""
    machine = PoseStateMachine()
    first = _pinch_click(machine, start=0)
    second = _pinch_click(machine, start=300)  # 첫 핀치 진입에서 ~300ms 뒤
    assert _kv(first) == [("mouse_down", 1), ("mouse_up", 1)]
    assert _kv(second) == [("mouse_down", 2), ("mouse_up", 2)]


def test_slow_second_pinch_is_plain_click() -> None:
    """간격이 double_click_ms를 넘으면 둘 다 clickState=1이다."""
    machine = PoseStateMachine()
    first = _pinch_click(machine, start=0)
    second = _pinch_click(machine, start=1000)  # 충분히 늦음
    assert _kv(first) == [("mouse_down", 1), ("mouse_up", 1)]
    assert _kv(second) == [("mouse_down", 1), ("mouse_up", 1)]


def test_triple_pinch_does_not_chain_double_clicks() -> None:
    """세 번 연속 핀치: 두 번째 눌림만 clickState=2, 세 번째는 다시 1로 돌아간다."""
    machine = PoseStateMachine()
    states = [
        int(e.value)
        for start in (0, 300, 600)
        for e in _pinch_click(machine, start=start)
        if e.kind == "mouse_down"
    ]
    assert states == [1, 2, 1]


def test_long_pinch_holds_button_then_releases() -> None:
    """오래 쥐면 버튼이 눌린 채 유지(드래그)되고, 릴리즈에서 한 번만 up이 나간다.

    down/up 모델에선 진입에서 이미 버튼이 down이라 별도 drag 이벤트가 없다 — 유지 중
    커서 move가 곧 드래그이고, 릴리즈의 mouse_up이 끝낸다.
    """
    machine = PoseStateMachine()
    held = _feed(machine, "pinch_index", ms=700)
    button = [e for e in held if e.kind in ("mouse_down", "mouse_up")]
    assert _kv(button) == [("mouse_down", 1)]  # 유지 내내 down 한 번, up 없음

    release = [e for t in (800, 833, 866, 900)
               for e in machine.update(_pose(NONE_POSE), t)]
    assert _kv(release) == [("mouse_up", 1)]


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
    """드래그(핀치 유지) 중에도 커서가 따라 움직인다 — 진입 down 뒤 move가 이어진다."""
    machine = PoseStateMachine()
    events = _feed_cursor(machine, "pinch_index", ((0.4, 0.4), (0.6, 0.6)), ms=700)
    assert any(e.kind == "move" for e in events)
    assert any(e.kind == "mouse_down" for e in events)  # 진입에서 버튼 down(=드래그 시작)


def _index_hand(*, straight: bool) -> np.ndarray:
    """검지를 곧게 편(True) / 애매하게 굽힌(False) index_point 랜드마크."""
    points = np.zeros((21, 2), dtype=np.float64)
    tip = np.array([0.0, -0.7])
    if straight:  # MCP·PIP·DIP·끝 일직선 → straightness 1.0
        points[5], points[6], points[7], points[8] = [0.0, 0.0], tip / 3, 2 * tip / 3, tip
    else:  # 끝이 손바닥으로 말려 straightness가 게이트 아래로 떨어진다
        points[5], points[6], points[7], points[8] = [0.0, 0.0], [0.0, -0.35], [0.25, -0.45], [0.45, -0.25]
    return points


def _feed_cursor_lm(machine, ref, landmarks, *, ms, start=0, step=33, palm=0.15):
    """`_feed_cursor`와 같되 매 프레임 landmarks도 넘긴다(폄 게이트 검증용)."""
    events, t = [], start
    while t <= start + ms:
        frac = (t - start) / max(ms, 1)
        if isinstance(ref[0], tuple):
            point = (ref[0][0] + (ref[1][0] - ref[0][0]) * frac,
                     ref[0][1] + (ref[1][1] - ref[0][1]) * frac)
        else:
            point = ref
        events.extend(
            machine.update(_pose("index_point"), t, landmarks, reference_point=point, palm_scale=palm)
        )
        t += step
    return events


def test_bent_index_gates_cursor() -> None:
    """검지를 애매하게 굽히면 index_point로 분류돼도 커서 이동에 진입하지 않는다."""
    machine = PoseStateMachine()
    bent = _index_hand(straight=False)
    _feed_cursor_lm(machine, (0.5, 0.5), bent, ms=200)  # 진입
    events = _feed_cursor_lm(machine, ((0.5, 0.5), (0.8, 0.5)), bent, ms=300, start=233)
    assert machine.state == "index_point"  # 상태는 유지된다
    assert [e for e in events if e.kind == "move"] == []  # 커서 이동은 차단


def test_straight_index_moves_cursor_through_gate() -> None:
    """검지를 곧게 펴면 폄 게이트를 통과해 커서가 이동한다(landmarks가 있어도)."""
    machine = PoseStateMachine()
    straight = _index_hand(straight=True)
    _feed_cursor_lm(machine, (0.5, 0.5), straight, ms=200)  # 진입
    events = _feed_cursor_lm(machine, ((0.5, 0.5), (0.8, 0.5)), straight, ms=300, start=233)
    assert any(e.kind == "move" for e in events)


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


# --- 검지 회전 → 볼륨 ---

def _hand_pointing(theta_deg: float) -> np.ndarray:
    """검지 MCP(5)→TIP(8)가 theta_deg 방향을 가리키는 최소 랜드마크."""
    p = np.zeros((21, 3))
    p[5] = (0.5, 0.5, 0.0)
    th = math.radians(theta_deg)
    p[8] = (0.5 + 0.1 * math.cos(th), 0.5 + 0.1 * math.sin(th), 0.0)
    return p


def _enter_index(machine: PoseStateMachine, *, start: int = 0) -> int:
    """낮은 tilt(trusted) index_point로 진입시키고 다음 timestamp를 돌려준다."""
    t = start
    while t <= start + 200:
        machine.update(_pose("index_point"), t, _hand_pointing(0.0))
        t += 33
    return t


def _rotate(machine: PoseStateMachine, *, sign: int, frames: int, start: int, trusted: bool = False):
    """검지 방향을 sign*15°/frame으로 돌리며 이벤트 종류를 모은다. 다음 timestamp도 반환."""
    events, t, theta = [], start, 0.0
    for _ in range(frames):
        theta += sign * 15.0
        events += [e.kind for e in machine.update(_pose("index_point", trusted=trusted), t, _hand_pointing(theta))]
        t += 33
    return events, t


def test_index_rotation_drives_volume() -> None:
    """검지를 돌리면 볼륨 스텝이 나오고, 반대로 돌리면 반대 방향이 나온다(고tilt·untrusted 포함).

    CW/CCW ↔ up/down 부호는 `ROT_SIGN`(실기기 확인값)에 달렸으므로, 여기서는 두 방향이
    **서로 반대**라는 불변만 검증한다.
    """
    machine = PoseStateMachine()
    t = _enter_index(machine)
    assert machine.state == "index_point"

    # frames=30(450°): 앞 ~24프레임(360°)은 워밍업, 이후부터 볼륨 스텝이 나온다.
    cw, t = _rotate(machine, sign=+1, frames=30, start=t)
    cw_vol = {e for e in cw if e.startswith("volume")}
    assert len(cw_vol) == 1, "한 방향 회전은 한 종류의 볼륨 스텝만"
    assert machine.state == "index_point"  # 회전 내내 상태 유지

    # 세션이 이미 활성화됐으므로(연속 index_point) 반대 회전은 워밍업 없이 바로 볼륨.
    ccw, t = _rotate(machine, sign=-1, frames=20, start=t)
    ccw_vol = {e for e in ccw if e.startswith("volume")}
    assert len(ccw_vol) == 1
    assert cw_vol != ccw_vol, "반대로 돌리면 볼륨 방향도 반대여야 한다"


def test_index_rotation_starts_at_high_tilt_without_entering_state() -> None:
    """상태 진입(낮은 tilt) 없이도, 고tilt(untrusted) 라벨만으로 회전 볼륨이 시작된다."""
    machine = PoseStateMachine()
    events, _ = _rotate(machine, sign=+1, frames=30, start=0, trusted=False)  # 워밍업 통과
    assert any(e.startswith("volume") for e in events)
    assert machine.state == ""  # 상태로 진입하지 않았어도 동작


def test_small_rotation_does_not_change_volume() -> None:
    """워밍업(한 바퀴 ROT_ACTIVATION_DEG) 미만의 회전은 볼륨을 바꾸지 않는다.

    일상 동작의 작은 검지 회전이 갑자기 볼륨을 한두 칸 바꾸던 것을 막는 게이트다.
    """
    machine = PoseStateMachine()
    events, _ = _rotate(machine, sign=+1, frames=16, start=0, trusted=False)  # 240° < ROT_ACTIVATION_DEG
    assert not any(e.startswith("volume") for e in events)


def test_volume_knob_locks_during_warmup() -> None:
    """워밍업(활성화 전) 중에도 회전이 감지되면 볼륨 노브 락이 걸린다 — 게이지 채우는
    동안 순간 오인식이 클릭·커서로 새지 않게(볼륨 자체는 활성화 후에만 나간다)."""
    machine = PoseStateMachine()
    events, _ = _rotate(machine, sign=+1, frames=10, start=0, trusted=False)  # 150° < ROT_ACTIVATION_DEG
    assert not any(e.startswith("volume") for e in events)  # 활성화 전 → 볼륨 없음
    assert not machine._rot_activated  # 아직 워밍업 중
    assert machine._rot_active_until > 0  # 그래도 락은 걸렸다(다른 동작 차단)


def test_volume_knob_mode_blocks_other_actions() -> None:
    """회전(볼륨 노브 모드) 중에는 순간 오인식(pinch 등)이 섞여도 클릭·드래그가 나오지 않는다."""
    machine = PoseStateMachine()
    events, t, theta = [], 0, 0.0
    # 워밍업(360°)을 index_point 프레임(3중 2)만으로 넘기려면 넉넉히 돌려야 한다.
    for i in range(48):
        theta += 15.0
        label = "pinch_index" if i % 3 == 0 else "index_point"  # 회전 중 오인식 섞기
        events += [e.kind for e in machine.update(_pose(label, trusted=False), t, _hand_pointing(theta))]
        t += 33
    assert not any(e in ("mouse_down", "mouse_up", "right_click") for e in events)
    assert any(e.startswith("volume") for e in events)


def test_pointing_without_rotation_does_not_change_volume() -> None:
    """검지가 한 방향을 가리킨 채(회전 없음) 있으면 볼륨 이벤트가 없다 — 포인팅과 회전의 분리."""
    machine = PoseStateMachine()
    t = _enter_index(machine)
    events = []
    for _ in range(30):
        events += [e.kind for e in machine.update(_pose("index_point", trusted=False), t, _hand_pointing(0.0))]
        t += 33
    assert not any(e.startswith("volume") for e in events)

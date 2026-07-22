"""자세 이벤트 → 실제 입력 브리지.

핵심 회귀: move·scroll은 UI 기록에서 빠지지만 실행은 되어야 한다. 예전에 로그를
줄이려고 _describe가 빈 문자열을 반환했고, apply가 그걸 보고 continue해 커서가 아예
움직이지 않았다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.gesture_fusion.pose_state import PoseEvent
from jarvis.monitoring.pose_control import SCROLL_INTERVAL_MS, PoseControlBridge
from jarvis.runtime_protocol.adapters.windows import InputKey, MouseButton


@dataclass
class _FakeSink:
    calls: list[tuple] = field(default_factory=list)

    def move_cursor(self, dx: int, dy: int, *, dragging: bool = False) -> None:
        self.calls.append(("move", dx, dy, dragging))

    def click(self, button: MouseButton) -> None:
        self.calls.append(("click", button))

    def double_click(self, button: MouseButton) -> None:
        self.calls.append(("double_click", button))

    def press(self, button: MouseButton, *, down: bool, click_state: int = 1) -> None:
        self.calls.append(("press", button, down, click_state))

    def scroll(self, ticks: int) -> None:
        self.calls.append(("scroll", ticks))

    def tap_key(self, key: InputKey) -> None:
        self.calls.append(("key", key))


def test_move_executes_even_though_it_is_not_logged() -> None:
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("move", 0, delta=(12.0, -4.0))])
    assert ("move", 12, -4, False) in sink.calls
    assert bridge.history == []  # 로그는 남지 않는다


def test_move_uses_absolute_pixels_regardless_of_resolution() -> None:
    # delta는 절대 픽셀이라 화면 해상도로 스케일하지 않는다 — 어떤 기기에서도 delta 그대로.
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("move", 0, delta=(37.0, -19.0))])
    assert ("move", 37, -19, False) in sink.calls


def test_disabled_bridge_executes_nothing() -> None:
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=False)
    bridge.apply([PoseEvent("mouse_down", 0, value=1.0), PoseEvent("move", 0, delta=(5.0, 5.0))])
    assert sink.calls == []
    assert "실행 안 함" in bridge.last_action  # 기록은 남아 관찰 가능


def test_scroll_is_throttled_for_execution_not_just_logging() -> None:
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("scroll", 1000, value=1.0)])
    bridge.apply([PoseEvent("scroll", 1000 + SCROLL_INTERVAL_MS - 10, value=1.0)])  # 너무 이름
    bridge.apply([PoseEvent("scroll", 1000 + SCROLL_INTERVAL_MS + 10, value=1.0)])
    scrolls = [c for c in sink.calls if c[0] == "scroll"]
    assert len(scrolls) == 2  # 가운데 것은 솎아진다


def test_pinch_down_up_and_right_click() -> None:
    """핀치 down/up은 버튼 press로, 우클릭(pinch_middle)은 그대로 click으로 나간다."""
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([
        PoseEvent("mouse_down", 0, value=1.0),
        PoseEvent("mouse_up", 0, value=1.0),
        PoseEvent("right_click", 0),
    ])
    assert ("press", MouseButton.LEFT, True, 1) in sink.calls
    assert ("press", MouseButton.LEFT, False, 1) in sink.calls
    assert ("click", MouseButton.RIGHT) in sink.calls


def test_double_click_carries_click_state() -> None:
    """더블클릭 눌림(value=2)은 press에 clickState=2를 실어 보낸다(OS가 합침)."""
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("mouse_down", 0, value=2.0)])
    assert ("press", MouseButton.LEFT, True, 2) in sink.calls
    assert bridge.last_action == "더블클릭 ↓"


def test_pinch_press_and_release() -> None:
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("mouse_down", 0, value=1.0)])
    bridge.apply([PoseEvent("mouse_up", 0, value=1.0)])
    assert ("press", MouseButton.LEFT, True, 1) in sink.calls
    assert ("press", MouseButton.LEFT, False, 1) in sink.calls


def test_release_lifts_held_button() -> None:
    """핀치(버튼 down) 중 손을 놓치거나 제어를 끄면 버튼이 눌린 채 남으면 안 된다."""
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("mouse_down", 0, value=1.0)])
    sink.calls.clear()
    bridge.release()
    assert any(c[:3] == ("press", MouseButton.LEFT, False) for c in sink.calls)


def test_media_toggle_sends_transition_key() -> None:
    """media_toggle은 플랫폼별 전이 키를 보낸다(Windows=재생/일시정지, macOS=F11).

    프로덕션과 같은 `_transition_key()`로 기대값을 잡아, 하드코딩한 F11이 Windows에서
    실패하던 문제를 없앤다(양 플랫폼에서 실제 계약을 검증한다).
    """
    from jarvis.monitoring.pose_control import _transition_key

    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("media_toggle", 0)])
    assert ("key", _transition_key()) in sink.calls

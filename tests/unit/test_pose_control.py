"""자세 이벤트 → 실제 입력 브리지.

핵심 회귀: move·scroll은 UI 기록에서 빠지지만 실행은 되어야 한다. 예전에 로그를
줄이려고 _describe가 빈 문자열을 반환했고, apply가 그걸 보고 continue해 커서가 아예
움직이지 않았다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.gesture_fusion.pose_state import (
    CURSOR_REFERENCE_HEIGHT,
    CURSOR_REFERENCE_WIDTH,
    PoseEvent,
)
from jarvis.monitoring.pose_control import SCROLL_INTERVAL_MS, PoseControlBridge
from jarvis.runtime_protocol.adapters.windows import InputKey, MouseButton


@dataclass
class _FakeSink:
    calls: list[tuple] = field(default_factory=list)
    # 기본은 기준 화면(스케일 1.0) — 기존 테스트가 delta를 그대로 관찰하도록.
    screen: tuple[int, int] = (CURSOR_REFERENCE_WIDTH, CURSOR_REFERENCE_HEIGHT)

    def screen_size(self) -> tuple[int, int]:
        return self.screen

    def move_cursor(self, dx: int, dy: int, *, dragging: bool = False) -> None:
        self.calls.append(("move", dx, dy, dragging))

    def click(self, button: MouseButton) -> None:
        self.calls.append(("click", button))

    def double_click(self, button: MouseButton) -> None:
        self.calls.append(("double_click", button))

    def press(self, button: MouseButton, *, down: bool) -> None:
        self.calls.append(("press", button, down))

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


def test_move_scaled_to_device_resolution() -> None:
    # 기준(1440×900)보다 큰 화면에선 같은 delta가 화면 비율만큼 커진다 — 체감 속도 동일.
    sink = _FakeSink(screen=(CURSOR_REFERENCE_WIDTH * 2, CURSOR_REFERENCE_HEIGHT * 2))
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("move", 0, delta=(12.0, -4.0))])
    assert ("move", 24, -8, False) in sink.calls


def test_move_on_reference_screen_is_unchanged() -> None:
    # 기준 화면과 같은 기기(튜닝한 macOS 1440×900)에선 스케일 1.0 → delta 그대로.
    sink = _FakeSink(screen=(CURSOR_REFERENCE_WIDTH, CURSOR_REFERENCE_HEIGHT))
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("move", 0, delta=(37.0, -19.0))])
    assert ("move", 37, -19, False) in sink.calls


def test_disabled_bridge_executes_nothing() -> None:
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=False)
    bridge.apply([PoseEvent("click", 0), PoseEvent("move", 0, delta=(5.0, 5.0))])
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


def test_click_and_right_click() -> None:
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("click", 0), PoseEvent("right_click", 0)])
    assert ("click", MouseButton.LEFT) in sink.calls
    assert ("click", MouseButton.RIGHT) in sink.calls


def test_double_click_executes_and_logs() -> None:
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("double_click", 0)])
    assert ("double_click", MouseButton.LEFT) in sink.calls
    assert bridge.last_action == "더블클릭"


def test_drag_press_and_release() -> None:
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("drag_start", 0)])
    bridge.apply([PoseEvent("drag_end", 0)])
    assert ("press", MouseButton.LEFT, True) in sink.calls
    assert ("press", MouseButton.LEFT, False) in sink.calls


def test_release_lifts_held_drag_button() -> None:
    """드래그 중 손을 놓치거나 제어를 끄면 버튼이 눌린 채 남으면 안 된다."""
    sink = _FakeSink()
    bridge = PoseControlBridge(sink=sink, enabled=True)
    bridge.apply([PoseEvent("drag_start", 0)])
    sink.calls.clear()
    bridge.release()
    assert ("press", MouseButton.LEFT, False) in sink.calls


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

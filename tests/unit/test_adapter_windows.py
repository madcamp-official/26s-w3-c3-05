"""Unit tests for the Windows adapter's command→input mapping."""

from __future__ import annotations

from unittest.mock import MagicMock

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.adapters.base import AdapterStatus
from jarvis.runtime_protocol.adapters.windows import InputKey, Win32InputSink, WindowsAdapter
from jarvis.runtime_protocol.protocol.capability import DeviceProfile

_LAPTOP = DeviceProfile(device_id="laptop", adapter="windows", capabilities={})


class RecordingSink:
    """Fake input sink that records calls instead of emitting real input."""

    def __init__(self) -> None:
        self.scrolls: list[int] = []
        self.keys: list[InputKey] = []
        self.moves: list[tuple[int, int]] = []
        self.window_switches: list[tuple[bool, int]] = []
        self.raise_on_scroll = False

    def scroll(self, ticks: int) -> None:
        if self.raise_on_scroll:
            raise OSError("simulated user32 failure")
        self.scrolls.append(ticks)

    def tap_key(self, key: InputKey) -> None:
        self.keys.append(key)

    def move_cursor(self, dx: int, dy: int) -> None:
        self.moves.append((dx, dy))

    def switch_window(self, forward: bool, repeat: int) -> None:
        self.window_switches.append((forward, repeat))


def _command(capability: str, operation: str, value: int | float | bool) -> Command:
    return Command(
        command_id="cmd-1",
        intent_id="intent-1",
        device_id="laptop",
        capability=capability,
        operation=operation,
        value=value,
        expires_at_ms=10_000,
    )


def test_scroll_increment_scrolls_up() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("scroll", "increment", 3), _LAPTOP)
    assert result.status == AdapterStatus.ACKNOWLEDGED
    assert sink.scrolls == [3]


def test_scroll_decrement_scrolls_down() -> None:
    sink = RecordingSink()
    WindowsAdapter(sink).execute(_command("scroll", "decrement", 2), _LAPTOP)
    assert sink.scrolls == [-2]


def test_volume_increment_taps_volume_up_n_times() -> None:
    sink = RecordingSink()
    WindowsAdapter(sink).execute(_command("volume", "increment", 3), _LAPTOP)
    assert sink.keys == [InputKey.VOLUME_UP] * 3


def test_media_toggle_taps_play_pause() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("media", "toggle", True), _LAPTOP)
    assert result.status == AdapterStatus.ACKNOWLEDGED
    assert sink.keys == [InputKey.PLAY_PAUSE]


def test_unknown_capability_fails_without_touching_sink() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("brightness", "set", 50), _LAPTOP)
    assert result.status == AdapterStatus.FAILED
    assert sink.scrolls == [] and sink.keys == []


def test_unsupported_operation_fails() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("scroll", "set", 5), _LAPTOP)
    assert result.status == AdapterStatus.FAILED
    assert sink.scrolls == []


def test_invalid_value_fails() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("scroll", "increment", True), _LAPTOP)
    assert result.status == AdapterStatus.FAILED
    assert sink.scrolls == []


def test_sink_error_reported_as_failed_not_raised() -> None:
    sink = RecordingSink()
    sink.raise_on_scroll = True
    result = WindowsAdapter(sink).execute(_command("scroll", "increment", 1), _LAPTOP)
    assert result.status == AdapterStatus.FAILED
    assert "user32 failure" in result.detail


def test_window_switch_increment_goes_forward() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("window_switch", "increment", 1), _LAPTOP)
    assert result.status == AdapterStatus.ACKNOWLEDGED
    assert sink.window_switches == [(True, 1)]


def test_window_switch_decrement_goes_backward_with_repeat() -> None:
    sink = RecordingSink()
    WindowsAdapter(sink).execute(_command("window_switch", "decrement", 2), _LAPTOP)
    assert sink.window_switches == [(False, 2)]


def test_window_switch_rejects_unsupported_operation() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("window_switch", "toggle", 1), _LAPTOP)
    assert result.status == AdapterStatus.FAILED
    assert sink.window_switches == []


def test_window_switch_rejects_invalid_count() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("window_switch", "increment", 0), _LAPTOP)
    assert result.status == AdapterStatus.FAILED
    assert sink.window_switches == []


def test_win32_move_cursor_uses_absolute_position_not_relative() -> None:
    """커서 이동은 현재 좌표+델타의 절대 이동(SetCursorPos)이라 포인터 가속을 우회한다.

    상대 `mouse_event`(MOUSEEVENTF_MOVE)는 Windows 포인터 가속에 왜곡돼 지시한
    픽셀 델타가 그대로 적용되지 않는다(macOS sink와 감도가 어긋남). 실제 user32
    호출은 하드웨어 경계라 여기서는 현재 위치를 (500, 300)으로 고정하고, 절대
    목표가 (현재+델타)로 계산돼 SetCursorPos에 넘어가는지만 검증한다.
    """
    sink = Win32InputSink()
    fake_user32 = MagicMock()
    sink._user32 = lambda: fake_user32  # type: ignore[method-assign]
    sink._cursor_position = lambda: (500, 300)  # type: ignore[method-assign]

    sink.move_cursor(30, -12)

    fake_user32.SetCursorPos.assert_called_once_with(530, 288)
    # 상대 이동(mouse_event)로는 절대 내지 않는다 — 가속 우회가 핵심.
    fake_user32.mouse_event.assert_not_called()


def test_win32_move_cursor_dragging_uses_same_absolute_move() -> None:
    """드래그 중에도 이동 방식은 같다 — 눌린 버튼 + 이동이면 Windows가 알아서 드래그.

    (macOS는 LeftMouseDragged로 분기하지만 Windows는 버튼 상태가 WM_MOUSEMOVE에
    실려 별도 분기가 필요 없다.)
    """
    sink = Win32InputSink()
    fake_user32 = MagicMock()
    sink._user32 = lambda: fake_user32  # type: ignore[method-assign]
    sink._cursor_position = lambda: (100, 100)  # type: ignore[method-assign]

    sink.move_cursor(5, 5, dragging=True)

    fake_user32.SetCursorPos.assert_called_once_with(105, 105)


def test_win32_move_cursor_skips_when_position_unavailable() -> None:
    """GetCursorPos가 실패하면(위치 None) 조용히 넘긴다 — 잘못된 좌표로 튀지 않는다."""
    sink = Win32InputSink()
    fake_user32 = MagicMock()
    sink._user32 = lambda: fake_user32  # type: ignore[method-assign]
    sink._cursor_position = lambda: None  # type: ignore[method-assign]

    sink.move_cursor(10, 10)

    fake_user32.SetCursorPos.assert_not_called()

"""Unit tests for the Windows adapter's command→input mapping."""

from __future__ import annotations

import pytest

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
        self.desktop_switches: list[tuple[bool, int]] = []
        self.raise_on_scroll = False

    def scroll(self, ticks: int) -> None:
        if self.raise_on_scroll:
            raise OSError("simulated user32 failure")
        self.scrolls.append(ticks)

    def tap_key(self, key: InputKey) -> None:
        self.keys.append(key)

    def move_cursor(self, dx: int, dy: int) -> None:
        self.moves.append((dx, dy))

    def switch_desktop(self, forward: bool, repeat: int) -> None:
        self.desktop_switches.append((forward, repeat))


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


def test_desktop_switch_increment_goes_forward() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("desktop_switch", "increment", 1), _LAPTOP)
    assert result.status == AdapterStatus.ACKNOWLEDGED
    assert sink.desktop_switches == [(True, 1)]


def test_desktop_switch_decrement_goes_backward_with_repeat() -> None:
    sink = RecordingSink()
    WindowsAdapter(sink).execute(_command("desktop_switch", "decrement", 2), _LAPTOP)
    assert sink.desktop_switches == [(False, 2)]


def test_desktop_switch_rejects_unsupported_operation() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("desktop_switch", "toggle", 1), _LAPTOP)
    assert result.status == AdapterStatus.FAILED
    assert sink.desktop_switches == []


def test_desktop_switch_rejects_invalid_count() -> None:
    sink = RecordingSink()
    result = WindowsAdapter(sink).execute(_command("desktop_switch", "increment", 0), _LAPTOP)
    assert result.status == AdapterStatus.FAILED
    assert sink.desktop_switches == []


# --- Win32InputSink 가상 데스크톱 순환 (2026-07-22) -------------------------------
# 데스크톱을 **새로 만들지 않고 기존 것들 사이만 순환**해야 한다(사용자 지시). 실제
# user32/레지스트리는 하드웨어·OS 경계라 여기서는 배치(_desktop_layout)를 고정하고
# "몇 칸을 어느 방향으로 누르는가"(_desktop_steps)만 검증한다.
# 방향 규약: forward(오른쪽 슬라이드)는 인덱스가 **감소**한다(windows.py 주석 참고).


def _sink_with_layout(count: int, current: int) -> Win32InputSink:
    sink = Win32InputSink()
    sink._desktop_layout = lambda: (count, current)  # type: ignore[method-assign]
    return sink


@pytest.mark.parametrize(
    ("current", "forward", "expected_target"),
    [
        (0, True, 2),  # 첫 데스크톱에서 forward → 마지막으로 감김(wrap)
        (1, True, 0),
        (2, True, 1),
        (0, False, 1),
        (1, False, 2),
        (2, False, 0),  # 마지막에서 backward → 처음으로 감김(wrap)
    ],
)
def test_desktop_steps_rotate_within_existing_desktops(
    current: int, forward: bool, expected_target: int
) -> None:
    """3개 데스크톱에서 양 끝이 서로 감기며 순환한다 — 새로 만들지 않는다."""
    sink = _sink_with_layout(3, current)
    assert current + sink._desktop_steps(forward, 1) == expected_target


def test_desktop_steps_single_desktop_does_not_move() -> None:
    """데스크톱이 하나뿐이면 이동하지 않는다(화살표를 눌러 새로 만들지 않는다)."""
    assert _sink_with_layout(1, 0)._desktop_steps(True, 1) == 0
    assert _sink_with_layout(1, 0)._desktop_steps(False, 1) == 0


def test_desktop_steps_large_repeat_stays_within_bounds() -> None:
    """repeat이 데스크톱 수보다 커도 감아서 범위 안에 머문다 — 끝을 넘어서지 않는다."""
    sink = _sink_with_layout(3, 0)
    target = 0 + sink._desktop_steps(True, 5)
    assert 0 <= target < 3
    assert abs(sink._desktop_steps(True, 5)) < 3  # 누르는 횟수도 count 미만


def test_desktop_steps_falls_back_when_layout_unavailable() -> None:
    """배치를 못 읽으면 예전처럼 요청 방향으로 repeat칸 이동한다(순환만 안 됨)."""
    sink = Win32InputSink()
    sink._desktop_layout = lambda: None  # type: ignore[method-assign]
    assert sink._desktop_steps(True, 2) == -2  # forward = 왼쪽(인덱스 감소)
    assert sink._desktop_steps(False, 2) == 2

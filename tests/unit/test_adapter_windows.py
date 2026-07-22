"""Unit tests for the Windows adapter's command→input mapping."""

from __future__ import annotations

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.adapters.base import AdapterStatus
from jarvis.runtime_protocol.adapters.windows import InputKey, WindowsAdapter
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

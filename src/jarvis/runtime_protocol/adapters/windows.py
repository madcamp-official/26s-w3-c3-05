"""Windows laptop adapter: discrete commands → OS input.

Maps validated commands (scroll / volume / media) to low-level OS input through
an injectable :class:`InputSink`. The real sink (:class:`Win32InputSink`) is the
hardware boundary that calls into user32; tests use a fake sink so the mapping is
verified without emitting real input.

Local synthetic input is accepted by the OS but its effect is not independently
read back, so successful laptop commands report ``ACKNOWLEDGED`` — never a
fabricated ``VERIFIED`` (development-principles 1.1). Anything this adapter does
not know how to translate is a ``FAILED`` result, not a guess.

The continuous cursor path (Cursor Control Mapper, README 6장) is a separate
concern that will reuse :meth:`InputSink.move_cursor`; this adapter handles only
the discrete command path from the protocol.

``WindowsAdapter`` itself only calls :class:`InputSink` methods, so it is not
actually OS-specific — only the sink implementation is. :func:`default_input_sink`
picks :class:`Win32InputSink` on Windows (unchanged) or
:class:`jarvis.runtime_protocol.adapters.macos.MacOSInputSink` on macOS (`macos`
extra), so the same adapter and command mapping run on both without duplicating
the scroll/volume/media logic.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from typing import Any, Protocol

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.adapters.base import AdapterResult, AdapterStatus
from jarvis.runtime_protocol.protocol.capability import DeviceProfile


class InputKey(StrEnum):
    """The system input keys this adapter emits."""

    PLAY_PAUSE = "play_pause"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MUTE = "mute"


class InputSink(Protocol):
    """Low-level OS input operations. The real implementation touches hardware."""

    def scroll(self, ticks: int) -> None:
        """Scroll the wheel by ``ticks`` (positive up, negative down)."""
        ...

    def tap_key(self, key: InputKey) -> None:
        """Press and release a single system key."""
        ...

    def move_cursor(self, dx: int, dy: int) -> None:
        """Move the cursor by a relative delta (used by the future pointer path)."""
        ...

    def switch_window(self, forward: bool, repeat: int) -> None:
        """Switch between application windows ``repeat`` times.

        ``forward`` advances to the next window (Alt+Tab / Cmd+Tab), else the
        previous one. The modifier-chord details live in each OS implementation.
        """
        ...


def _as_count(value: int | float | bool) -> int | None:
    """Interpret a command value as a positive repeat count, or ``None`` if invalid.

    bool is rejected explicitly: it is an ``int`` subclass but never a valid count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    count = int(value)
    return count if count > 0 else None


class WindowsAdapter:
    """Executes laptop commands via an :class:`InputSink`."""

    name = "windows"

    def __init__(self, sink: InputSink) -> None:
        self._sink = sink

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult:
        try:
            return self._execute(command)
        except Exception as exc:  # noqa: BLE001 - report any OS failure honestly
            return AdapterResult(
                AdapterStatus.FAILED, f"input sink error: {type(exc).__name__}: {exc}"
            )

    def _execute(self, command: Command) -> AdapterResult:
        if command.capability == "scroll":
            return self._scroll(command)
        if command.capability == "volume":
            return self._volume(command)
        if command.capability == "media":
            return self._media(command)
        if command.capability == "window_switch":
            return self._window_switch(command)
        return AdapterResult(
            AdapterStatus.FAILED,
            f"windows adapter does not handle capability {command.capability!r}",
        )

    def _scroll(self, command: Command) -> AdapterResult:
        count = _as_count(command.value)
        if count is None:
            return AdapterResult(AdapterStatus.FAILED, f"invalid scroll amount {command.value!r}")
        if command.operation == "increment":
            self._sink.scroll(count)
        elif command.operation == "decrement":
            self._sink.scroll(-count)
        else:
            return AdapterResult(
                AdapterStatus.FAILED, f"scroll does not support {command.operation!r}"
            )
        return AdapterResult(AdapterStatus.ACKNOWLEDGED, f"scrolled {command.operation} {count}")

    def _volume(self, command: Command) -> AdapterResult:
        count = _as_count(command.value)
        if count is None:
            return AdapterResult(AdapterStatus.FAILED, f"invalid volume step {command.value!r}")
        if command.operation == "increment":
            key = InputKey.VOLUME_UP
        elif command.operation == "decrement":
            key = InputKey.VOLUME_DOWN
        else:
            return AdapterResult(
                AdapterStatus.FAILED, f"volume does not support {command.operation!r}"
            )
        for _ in range(count):
            self._sink.tap_key(key)
        return AdapterResult(AdapterStatus.ACKNOWLEDGED, f"volume {command.operation} x{count}")

    def _media(self, command: Command) -> AdapterResult:
        if command.operation not in ("toggle", "set"):
            return AdapterResult(
                AdapterStatus.FAILED, f"media does not support {command.operation!r}"
            )
        self._sink.tap_key(InputKey.PLAY_PAUSE)
        return AdapterResult(AdapterStatus.ACKNOWLEDGED, "media play/pause toggled")

    def _window_switch(self, command: Command) -> AdapterResult:
        count = _as_count(command.value)
        if count is None:
            return AdapterResult(
                AdapterStatus.FAILED, f"invalid window_switch count {command.value!r}"
            )
        if command.operation == "increment":
            forward = True
        elif command.operation == "decrement":
            forward = False
        else:
            return AdapterResult(
                AdapterStatus.FAILED, f"window_switch does not support {command.operation!r}"
            )
        self._sink.switch_window(forward, count)
        direction = "next" if forward else "previous"
        return AdapterResult(AdapterStatus.ACKNOWLEDGED, f"window switch {direction} x{count}")


# --- Hardware boundary ------------------------------------------------------

_VK = {
    InputKey.PLAY_PAUSE: 0xB3,
    InputKey.VOLUME_UP: 0xAF,
    InputKey.VOLUME_DOWN: 0xAE,
    InputKey.MUTE: 0xAD,
}
_KEYEVENTF_KEYUP = 0x0002
_MOUSEEVENTF_WHEEL = 0x0800
_WHEEL_DELTA = 120
_VK_TAB = 0x09
_VK_MENU = 0x12  # ALT
_VK_SHIFT = 0x10


class Win32InputSink:
    """Real OS input via ``user32`` (Windows only, manually verified).

    Uses the classic ``keybd_event``/``mouse_event`` entry points. ``ctypes`` is
    imported lazily so importing this module never fails on a non-Windows host or
    in tests; the calls themselves only work on Windows.
    """

    def _user32(self) -> Any:
        import ctypes

        # ctypes function pointers are resolved dynamically at call time; typing
        # the handle as Any is the honest description of this boundary. `windll`
        # only exists on Windows, so reach it via getattr — a static
        # ``ctypes.windll`` trips [attr-defined] when mypy runs on a non-Windows
        # host (a per-line ignore would flip to unused when checked on Windows).
        return getattr(ctypes, "windll").user32

    def scroll(self, ticks: int) -> None:
        self._user32().mouse_event(_MOUSEEVENTF_WHEEL, 0, 0, ticks * _WHEEL_DELTA, 0)

    def tap_key(self, key: InputKey) -> None:
        user32 = self._user32()
        vk = _VK[key]
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)

    def move_cursor(self, dx: int, dy: int) -> None:
        # MOUSEEVENTF_MOVE (0x0001) moves relative to the current cursor position.
        self._user32().mouse_event(0x0001, dx, dy, 0, 0)

    def switch_window(self, forward: bool, repeat: int) -> None:
        # Alt+Tab (forward) / Alt+Shift+Tab (backward). Hold Alt for the whole
        # sequence so the window switcher stays up across repeated Tab taps.
        user32 = self._user32()
        user32.keybd_event(_VK_MENU, 0, 0, 0)
        if not forward:
            user32.keybd_event(_VK_SHIFT, 0, 0, 0)
        for _ in range(repeat):
            user32.keybd_event(_VK_TAB, 0, 0, 0)
            user32.keybd_event(_VK_TAB, 0, _KEYEVENTF_KEYUP, 0)
        if not forward:
            user32.keybd_event(_VK_SHIFT, 0, _KEYEVENTF_KEYUP, 0)
        user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)


def default_input_sink() -> InputSink:
    """이 프로세스가 돌고 있는 OS에 맞는 실제 `InputSink`를 고른다.

    Windows에서는 지금까지와 똑같이 `Win32InputSink`를 반환한다(이 함수 도입으로
    Windows 쪽 동작·의존성은 전혀 바뀌지 않는다). macOS는 `adapters.macos.
    MacOSInputSink`를 지연 import해 반환한다 — `macos` extra가 없는 환경에서는
    이 브랜치를 타지 않는 한(=macOS가 아닌 한) import 자체가 실패하지 않는다.
    지원하지 않는 OS(Linux 등)는 추측 대신 `RuntimeError`로 정직하게 실패한다
    (development-principles.md 1.1: 성공을 가장하지 않는다).
    """
    if sys.platform == "win32":
        return Win32InputSink()
    if sys.platform == "darwin":
        from jarvis.runtime_protocol.adapters.macos import MacOSInputSink

        return MacOSInputSink()
    raise RuntimeError(
        f"no InputSink implementation for platform {sys.platform!r} "
        "(supported: win32, darwin)"
    )

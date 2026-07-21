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
    SHOW_DESKTOP = "show_desktop"  # F11 — 바탕화면 표시(누를 때마다 토글)
    MISSION_CONTROL = "mission_control"  # macOS 전용 — Mission Control 열기(Ctrl+↑)


class MouseButton(StrEnum):
    """클릭에 쓰는 마우스 버튼."""

    LEFT = "left"
    RIGHT = "right"


class InputSink(Protocol):
    """Low-level OS input operations. The real implementation touches hardware."""

    def click(self, button: MouseButton) -> None:
        """버튼 하나를 누름→뗌으로 전송한다(현재 커서 위치)."""
        ...

    def double_click(self, button: MouseButton) -> None:
        """버튼을 두 번 빠르게 눌러 더블클릭을 전송한다(현재 커서 위치)."""
        ...

    def press(self, button: MouseButton, *, down: bool) -> None:
        """버튼을 누르거나 뗀다 — 드래그는 누른 채 커서를 옮겨야 하므로 분리한다."""
        ...

    def scroll(self, ticks: int) -> None:
        """Scroll the wheel by ``ticks`` (positive up, negative down)."""
        ...

    def tap_key(self, key: InputKey) -> None:
        """Press and release a single system key."""
        ...

    def move_cursor(self, dx: int, dy: int, *, dragging: bool = False) -> None:
        """Move the cursor by a relative delta. ``dragging``이면 드래그로 이동한다."""
        ...

    def screen_size(self) -> tuple[int, int]:
        """주 디스플레이의 (너비, 높이). 커서 이동을 화면 대비 비율로 정규화하는 데 쓴다.

        move_cursor가 쓰는 좌표계와 같은 단위여야 한다(Windows=픽셀, macOS=points).
        """
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
    InputKey.SHOW_DESKTOP: 0x7A,  # VK_F11
}
_KEYEVENTF_KEYUP = 0x0002
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_WHEEL_DELTA = 120
_VK_TAB = 0x09
_VK_MENU = 0x12  # ALT
_VK_SHIFT = 0x10
_SM_CXSCREEN = 0  # GetSystemMetrics: 주 디스플레이 너비(px)
_SM_CYSCREEN = 1  # 〃 높이(px)


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

    def press(self, button: MouseButton, *, down: bool) -> None:
        """버튼 상태를 바꾼다. 드래그는 누른 채 이동해야 해서 click과 분리돼 있다."""
        flags = {
            (MouseButton.LEFT, True): _MOUSEEVENTF_LEFTDOWN,
            (MouseButton.LEFT, False): _MOUSEEVENTF_LEFTUP,
            (MouseButton.RIGHT, True): _MOUSEEVENTF_RIGHTDOWN,
            (MouseButton.RIGHT, False): _MOUSEEVENTF_RIGHTUP,
        }[(button, down)]
        self._user32().mouse_event(flags, 0, 0, 0, 0)

    def click(self, button: MouseButton) -> None:
        self.press(button, down=True)
        self.press(button, down=False)

    def double_click(self, button: MouseButton) -> None:
        # 두 번의 클릭을 연속으로 보낸다. Windows는 두 클릭의 간격이 GetDoubleClickTime
        # (기본 500ms) 이내면 OS가 더블클릭으로 합쳐 인식한다 — 상태기계가 이미 그 안에서
        # 승격했으므로 여기선 지연 없이 두 번 보내면 된다.
        self.click(button)
        self.click(button)

    def scroll(self, ticks: int) -> None:
        self._user32().mouse_event(_MOUSEEVENTF_WHEEL, 0, 0, ticks * _WHEEL_DELTA, 0)

    def tap_key(self, key: InputKey) -> None:
        user32 = self._user32()
        vk = _VK[key]
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)

    def move_cursor(self, dx: int, dy: int, *, dragging: bool = False) -> None:
        # 절대좌표 warp로 이동한다(상대 mouse_event가 아니라). MOUSEEVENTF_MOVE 상대
        # 이동은 Windows 포인터 가속("포인터 정확도 향상", 기본 ON) 곡선을 타서, 앱이
        # 이미 준 가속 위에 OS가 2차 가속을 또 겹친다 — 이게 macOS 대비 커서가 훨씬
        # 빠르고 튀는 주원인이다. 현재 위치를 읽어 델타를 더한 절대좌표로 SetCursorPos를
        # 부르면 탄도(ballistics)를 우회해 macOS(절대 warp)와 동일하게 동작한다.
        # 버튼이 눌린 채 SetCursorPos로 옮기면 드래그는 그대로 따라오므로 dragging은
        # 경로에 영향을 주지 않는다.
        import ctypes
        import ctypes.wintypes

        user32 = self._user32()
        point = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        user32.SetCursorPos(int(point.x + dx), int(point.y + dy))

    def screen_size(self) -> tuple[int, int]:
        user32 = self._user32()
        return user32.GetSystemMetrics(_SM_CXSCREEN), user32.GetSystemMetrics(_SM_CYSCREEN)

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

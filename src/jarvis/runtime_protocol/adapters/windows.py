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

import ctypes
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
        """현재 위치에서 (dx, dy)px만큼 커서를 옮긴다.

        ``dragging``이면 버튼이 눌린 상태의 드래그 이동임을 sink에 알린다 — macOS는
        ``LeftMouseDragged``로 분기해야 창·선택이 실시간으로 따라오고, Windows는
        눌린 버튼 상태가 이동 이벤트에 실려 별도 분기가 필요 없다.
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


class _POINT(ctypes.Structure):
    """``GetCursorPos``가 채우는 화면 좌표(px). 커서 절대 이동 계산용.

    ``ctypes`` 자체는 모든 OS에서 import되므로(Windows 전용은 ``windll`` 접근뿐)
    이 구조체를 모듈 최상위에 둬도 non-Windows import는 깨지지 않는다.
    """

    _fields_ = (("x", ctypes.c_long), ("y", ctypes.c_long))


class Win32InputSink:
    """Real OS input via ``user32`` (Windows only, manually verified).

    Keys·scroll·click은 classic ``keybd_event``/``mouse_event`` 경로를, 커서 이동은
    ``GetCursorPos``+``SetCursorPos`` 절대 좌표 경로를 쓴다(상대 ``mouse_event``는
    포인터 가속에 왜곡되므로 — `move_cursor` 참고). ``windll`` 접근만 Windows
    전용이라 ``_user32``에서 지연 resolve하고, 이 모듈 import 자체는 non-Windows
    호스트·테스트에서도 항상 성공한다(실제 호출 시에만 Windows가 필요).
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

    def scroll(self, ticks: int) -> None:
        self._user32().mouse_event(_MOUSEEVENTF_WHEEL, 0, 0, ticks * _WHEEL_DELTA, 0)

    def tap_key(self, key: InputKey) -> None:
        user32 = self._user32()
        vk = _VK[key]
        user32.keybd_event(vk, 0, 0, 0)
        user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)

    def _cursor_position(self) -> tuple[int, int] | None:
        """현재 커서의 화면 좌표(px). 못 읽으면 None(다음 프레임에 재시도)."""
        point = _POINT()
        if not self._user32().GetCursorPos(ctypes.byref(point)):
            return None
        return int(point.x), int(point.y)

    def move_cursor(self, dx: int, dy: int, *, dragging: bool = False) -> None:
        """현재 위치에서 정확히 (dx, dy)px 이동 — 현재 좌표+델타의 절대 이동으로 낸다.

        상대 ``mouse_event(MOUSEEVENTF_MOVE)``는 Windows "포인터 정확도 향상"
        (포인터 가속) 곡선을 타서, 컨트롤러가 계산한 픽셀 델타가 그대로 적용되지
        않는다 — 느린 이동은 덜, 빠른 이동은 더 나가 손 랜드마크 기준 감도가
        어긋난다(실측: 5px×20회=100px 지시가 81px로 적용). ``GetCursorPos``로 현재
        위치를 읽어 델타를 더한 절대 좌표를 ``SetCursorPos``해 가속을 우회한다 —
        macOS sink(현재 위치+델타 절대 이동)와 같은 의미라 두 OS 감도가 일치한다.

        ``dragging``이면 pose_control이 이미 왼쪽 버튼을 누른 상태다(``press`` 참고).
        버튼이 눌린 채 커서가 움직이면 Windows가 ``WM_MOUSEMOVE``에 ``MK_LBUTTON``을
        실어 보내 창·선택이 실시간으로 따라오므로, 이동 방식은 드래그 여부와
        무관하게 같다(macOS는 별도 ``LeftMouseDragged``가 필요해 분기하지만 Windows는
        불필요).
        """
        current = self._cursor_position()
        if current is None:
            return
        self._user32().SetCursorPos(current[0] + dx, current[1] + dy)

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

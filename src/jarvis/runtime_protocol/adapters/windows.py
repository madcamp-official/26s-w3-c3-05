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
    TASK_VIEW = "task_view"  # Windows 전용 — Task View 열기(Win+Tab). macOS Mission Control 대응
    CLOSE_TAB = "close_tab"  # 탭 닫기 — Windows Ctrl+W, macOS Cmd+W. 앱이 정의한 탭/창 닫기


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

    def press(self, button: MouseButton, *, down: bool, click_state: int = 1) -> None:
        """버튼을 누르거나 뗀다 — 드래그는 누른 채 커서를 옮겨야 하므로 분리한다.

        `click_state`(1=단일, 2=더블)는 macOS가 더블클릭을 합치는 데 쓰는 힌트다.
        Windows는 두 클릭의 간격(GetDoubleClickTime)으로 OS가 알아서 합치므로 무시한다.
        """
        ...

    def scroll(self, ticks: int) -> None:
        """Scroll the wheel by ``ticks`` (positive up, negative down)."""
        ...

    def center_cursor_on_foreground(self) -> None:
        """Move the cursor to the center of the current foreground window.

        A synthesized wheel event is a hardware-level signal delivered to
        whatever window is under the cursor — not to the focused window. The
        Intent-driven scroll path (unlike pose control, which already tracks
        the cursor to the hand) never positions the cursor at all, so without
        a physical mouse the cursor can sit anywhere (e.g. over this app's own
        window), and scroll commands silently land nowhere useful. Call this
        immediately before :meth:`scroll` to make the command actually visible.
        """
        ...

    def tap_key(self, key: InputKey) -> None:
        """Press and release a single system key."""
        ...

    def move_cursor(self, dx: int, dy: int, *, dragging: bool = False) -> None:
        """Move the cursor by a relative delta. ``dragging``이면 드래그로 이동한다."""
        ...

    def switch_desktop(self, forward: bool, repeat: int) -> None:
        """Switch between virtual desktops ``repeat`` times.

        ``forward`` advances to the next desktop (Ctrl+Win+→ on Windows,
        Ctrl+→ / 다음 Space on macOS), else the previous one. The
        modifier-chord details live in each OS implementation.
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
        if command.capability == "desktop_switch":
            return self._desktop_switch(command)
        return AdapterResult(
            AdapterStatus.FAILED,
            f"windows adapter does not handle capability {command.capability!r}",
        )

    def _scroll(self, command: Command) -> AdapterResult:
        count = _as_count(command.value)
        if count is None:
            return AdapterResult(AdapterStatus.FAILED, f"invalid scroll amount {command.value!r}")
        # 이 경로(제스처→Intent)는 pose 제어와 달리 손 움직임을 커서에 반영하지 않으므로,
        # 물리 마우스가 없으면 커서가 어디 있는지 알 수 없다 — 휠 이벤트를 실제로 보이게
        # 하려면 보내기 직전에 포그라운드 창으로 커서를 옮겨야 한다.
        if command.operation in ("increment", "decrement"):
            self._sink.center_cursor_on_foreground()
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

    def _desktop_switch(self, command: Command) -> AdapterResult:
        count = _as_count(command.value)
        if count is None:
            return AdapterResult(
                AdapterStatus.FAILED, f"invalid desktop_switch count {command.value!r}"
            )
        if command.operation == "increment":
            forward = True
        elif command.operation == "decrement":
            forward = False
        else:
            return AdapterResult(
                AdapterStatus.FAILED, f"desktop_switch does not support {command.operation!r}"
            )
        self._sink.switch_desktop(forward, count)
        direction = "next" if forward else "previous"
        return AdapterResult(AdapterStatus.ACKNOWLEDGED, f"desktop switch {direction} x{count}")


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
_VK_LWIN = 0x5B  # 왼쪽 Windows 키 — Task View(Win+Tab)·가상 데스크톱 전환(Ctrl+Win+←/→)용
_VK_CONTROL = 0x11
_VK_W = 0x57  # 탭 닫기(Ctrl+W)용
_VK_LEFT = 0x25  # ← 화살표 — 이전 가상 데스크톱
_VK_RIGHT = 0x27  # → 화살표 — 다음 가상 데스크톱

# 탐색기가 가상 데스크톱 배치를 저장하는 레지스트리 키(HKCU 하위). 문서화되지 않은
# 값이라 빌드에 따라 없을 수 있다 — `_desktop_layout`이 그 경우를 None으로 처리한다.
_VIRTUAL_DESKTOP_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\VirtualDesktops"
_DESKTOP_GUID_BYTES = 16  # 데스크톱 GUID 하나의 바이트 수(VirtualDesktopIDs가 이 단위로 이어짐)


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

    def press(self, button: MouseButton, *, down: bool, click_state: int = 1) -> None:
        """버튼 상태를 바꾼다. 드래그는 누른 채 이동해야 해서 click과 분리돼 있다.

        `click_state`는 macOS 전용 힌트라 여기선 무시한다 — Windows는 두 down/up 쌍의
        간격이 GetDoubleClickTime 이내면 OS가 알아서 더블클릭으로 합친다.
        """
        del click_state  # Windows에선 불필요(OS가 타이밍으로 판정)
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

    def center_cursor_on_foreground(self) -> None:
        # 포그라운드 창이 없거나(바탕화면 등) 좌표를 못 얻으면 조용히 넘어간다 — 커서
        # 위치 보정 하나가 실패했다고 스크롤 명령 자체를 막지 않는다.
        import ctypes
        import ctypes.wintypes

        user32 = self._user32()
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return
        rect = ctypes.wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return
        center_x = (rect.left + rect.right) // 2
        center_y = (rect.top + rect.bottom) // 2
        user32.SetCursorPos(center_x, center_y)

    def tap_key(self, key: InputKey) -> None:
        user32 = self._user32()
        if key is InputKey.TASK_VIEW:
            # Task View는 Win+Tab 조합키라 modifier(Win)로 감싼다(switch_desktop의
            # Ctrl+Win+화살표와 같은 modifier-hold 구조). macOS Mission Control에 대응하는 창 개요 화면이다.
            user32.keybd_event(_VK_LWIN, 0, 0, 0)
            user32.keybd_event(_VK_TAB, 0, 0, 0)
            user32.keybd_event(_VK_TAB, 0, _KEYEVENTF_KEYUP, 0)
            user32.keybd_event(_VK_LWIN, 0, _KEYEVENTF_KEYUP, 0)
            return
        if key is InputKey.CLOSE_TAB:
            # 탭 닫기는 Ctrl+W 조합키라 modifier(Ctrl)로 감싼다(TASK_VIEW의 Win+Tab과
            # 같은 구조). 앱이 정의한 "현재 탭 또는 창 닫기"로 동작한다.
            user32.keybd_event(_VK_CONTROL, 0, 0, 0)
            user32.keybd_event(_VK_W, 0, 0, 0)
            user32.keybd_event(_VK_W, 0, _KEYEVENTF_KEYUP, 0)
            user32.keybd_event(_VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0)
            return
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

    def switch_desktop(self, forward: bool, repeat: int) -> None:
        # 가상 데스크톱 전환: Ctrl+Win+← (forward) / Ctrl+Win+→ (backward).
        # 방향은 제스처와 반대로 매핑한다(사용자 지시): 오른쪽 슬라이드(forward)는
        # 왼쪽 데스크톱으로 간다.
        #
        # **기존 데스크톱 사이를 순환한다 — 새로 만들지 않는다**(2026-07-22 사용자 지시).
        # 레지스트리로 데스크톱 개수와 현재 위치를 읽어 목표 인덱스를 `% count`로 감아
        # 계산하고, 그 차이만큼만 화살표를 누른다. 누르는 횟수가 항상 `count` 미만이라
        # 끝을 넘어설 수 없다 — repeat이 커도 마지막 데스크톱 밖으로 나가지 않는다.
        steps = self._desktop_steps(forward, repeat)
        if steps == 0:
            return  # 데스크톱이 하나뿐이거나 이동할 필요가 없음
        # Ctrl+Win을 시퀀스 전체 동안 누른 채로 화살표를 눌러, 여러 칸을 이동해도
        # 조합이 유지되게 한다.
        user32 = self._user32()
        arrow = _VK_RIGHT if steps > 0 else _VK_LEFT
        user32.keybd_event(_VK_CONTROL, 0, 0, 0)
        user32.keybd_event(_VK_LWIN, 0, 0, 0)
        for _ in range(abs(steps)):
            user32.keybd_event(arrow, 0, 0, 0)
            user32.keybd_event(arrow, 0, _KEYEVENTF_KEYUP, 0)
        user32.keybd_event(_VK_LWIN, 0, _KEYEVENTF_KEYUP, 0)
        user32.keybd_event(_VK_CONTROL, 0, _KEYEVENTF_KEYUP, 0)

    def _desktop_steps(self, forward: bool, repeat: int) -> int:
        """이번 전환에서 실제로 눌러야 할 화살표 이동량(+=오른쪽, -=왼쪽).

        데스크톱 배치를 읽을 수 있으면 순환(wrap)까지 계산하고, 못 읽으면 예전처럼
        요청된 방향으로 `repeat`칸만 이동한다 — Ctrl+Win+화살표는 끝에서 아무 일도
        하지 않을 뿐 데스크톱을 만들지 않으므로, 이 fallback도 새 데스크톱을 만들지는
        않는다(순환만 안 될 뿐이다).
        """
        # 인덱스 기준 방향: forward는 왼쪽(인덱스 감소)으로 간다(위 주석의 반대 매핑).
        delta = -repeat if forward else repeat
        layout = self._desktop_layout()
        if layout is None:
            return delta
        count, current = layout
        if count <= 1:
            return 0
        return ((current + delta) % count) - current

    def _desktop_layout(self) -> tuple[int, int] | None:
        """`(데스크톱 개수, 현재 인덱스)` — 읽지 못하면 None.

        Windows는 가상 데스크톱 개수·현재 위치를 공개 API로 주지 않아 탐색기가 쓰는
        레지스트리 값을 읽는다. `VirtualDesktopIDs`는 데스크톱 GUID(16바이트)를 화면
        왼쪽→오른쪽 순서로 이어 붙인 값이고, `CurrentVirtualDesktop`은 그중 현재
        데스크톱의 GUID다. 문서화되지 않은 값이라 Windows 빌드에 따라 없을 수 있어,
        읽기 실패·형식 불일치는 전부 None으로 돌려 호출부가 fallback하게 한다
        (development-principles.md 1.1: 못 읽은 것을 아는 척하지 않는다).
        """
        import winreg  # Windows 전용 stdlib — `_user32`와 같은 이유로 지연 import한다.

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _VIRTUAL_DESKTOP_KEY) as key:
                ids, _ = winreg.QueryValueEx(key, "VirtualDesktopIDs")
                current, _ = winreg.QueryValueEx(key, "CurrentVirtualDesktop")
        except OSError:
            return None
        if not isinstance(ids, bytes) or not isinstance(current, bytes):
            return None
        if len(current) != _DESKTOP_GUID_BYTES or not ids or len(ids) % _DESKTOP_GUID_BYTES:
            return None
        count = len(ids) // _DESKTOP_GUID_BYTES
        for index in range(count):
            if ids[index * _DESKTOP_GUID_BYTES : (index + 1) * _DESKTOP_GUID_BYTES] == current:
                return count, index
        return None  # 현재 GUID가 목록에 없음 — 신뢰할 수 없으니 fallback


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

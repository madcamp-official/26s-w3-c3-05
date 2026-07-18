"""macOS `InputSink` implementation — extends the Windows laptop adapter path
to run on macOS without touching Windows behavior.

`WindowsAdapter`(in `windows.py`)의 명령→입력 매핑 로직(scroll/volume/media)은
이미 `InputSink`를 통해서만 하드웨어를 건드리므로 OS 무관하다. Windows 쪽은
그대로 `Win32InputSink`(user32, ctypes)를 계속 쓴다 — 이 파일은 macOS에서
같은 `InputSink` Protocol을 구현하는 별도 구현체만 추가한다(`mediapipe_hands.py`
·`model.py`와 같은 무거운/선택적 의존성 격리 원칙: 이 모듈이 `jarvis.runtime_protocol`
패키지에서 PyObjC(Quartz/AppKit)를 import하는 유일한 곳이다. `pyproject.toml`의
`macos` extra는 이 모듈에서만 필요하다).

Quartz만으로는 볼륨·재생/일시정지 같은 macOS "시스템 정의(system-defined)" 미디어
키 이벤트를 만들 수 없어(표준 키보드 keycode가 아님) `AppKit.NSEvent`도 함께
쓴다(macOS 미디어 키 전송의 표준 관용구, `NX_KEYTYPE_*` 상수 기반).

Win32InputSink와 마찬가지로 로컬 합성 입력은 OS가 받아들이지만 효과를 되읽지
않으므로, 이 sink를 쓰는 adapter의 성공 결과는 `ACKNOWLEDGED`가 정직한 상한이다
(development-principles.md 1.1). 실물 macOS 하드웨어 검증이 필요하다(자동
테스트는 fake sink를 쓴다).
"""

from __future__ import annotations

from typing import Any

from jarvis.runtime_protocol.adapters.windows import InputKey

# NX_KEYTYPE_* — macOS 시스템 정의 미디어 키 코드(IOKit/hidsystem/ev_keymap.h).
_NX_KEYTYPE = {
    InputKey.PLAY_PAUSE: 16,  # NX_KEYTYPE_PLAY
    InputKey.VOLUME_UP: 0,  # NX_KEYTYPE_SOUND_UP
    InputKey.VOLUME_DOWN: 1,  # NX_KEYTYPE_SOUND_DOWN
    InputKey.MUTE: 7,  # NX_KEYTYPE_MUTE
}

# CGScrollWheelEvent 단위: 한 "줄" 단위로 보낸다(Windows의 WHEEL_DELTA 배수 스크롤과
# 감각을 맞추는 정도로 충분 — 정밀 매핑은 실기기 튜닝 대상).
_SCROLL_UNIT_LINE = 1

# 표준 ANSI 키보드 keycode(Carbon HIToolbox/Events.h). Cmd+Tab 창 전환용.
_KEYCODE_TAB = 0x30
_KEYCODE_COMMAND = 0x37
_KEYCODE_SHIFT = 0x38


class MacOSInputSink:
    """Real OS input via Quartz/AppKit CGEvent (macOS only, manually verified).

    `Win32InputSink`와 같은 지연 import 원칙: PyObjC는 이 클래스의 메서드 안에서만
    import해, macOS가 아닌 호스트나 `macos` extra 미설치 환경에서 이 모듈을
    import하는 것 자체는 항상 성공한다(실제 호출 시에만 실패).
    """

    def _post(self, event: Any) -> None:
        import Quartz

        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def scroll(self, ticks: int) -> None:
        import Quartz

        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, _SCROLL_UNIT_LINE, ticks
        )
        self._post(event)

    def tap_key(self, key: InputKey) -> None:
        """미디어 키 하나를 누름→뗌으로 전송한다.

        표준 키보드 keycode가 아니라 macOS의 "시스템 정의" 이벤트(`NSEvent.
        otherEventWithType...`)로 보낸다 — 볼륨·재생/일시정지는 이 경로가 아니면
        OS가 인식하지 않는다. 잘 알려진 관용구(NX_KEYTYPE + data1 인코딩)를 그대로
        따른다.
        """
        from AppKit import NSEvent, NSSystemDefined

        key_code = _NX_KEYTYPE[key]
        for key_down in (True, False):
            flags = 0xA00 if key_down else 0xB00
            data1 = (key_code << 16) | ((0xA if key_down else 0xB) << 8)
            event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                NSSystemDefined, (0, 0), flags, 0, 0, None, 8, data1, -1
            )
            self._post(event.CGEvent())

    def move_cursor(self, dx: int, dy: int) -> None:
        """현재 커서 위치에서 상대 이동(Win32의 `MOUSEEVENTF_MOVE`와 같은 의미).

        CGEvent의 마우스 이동은 절대좌표만 받으므로, 현재 위치를 읽어 델타를
        더한 뒤 그 절대좌표로 이동 이벤트를 만든다.
        """
        import Quartz

        current = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        target = (current.x + dx, current.y + dy)
        event = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved, target, Quartz.kCGMouseButtonLeft
        )
        self._post(event)

    def switch_window(self, forward: bool, repeat: int) -> None:
        """Cmd+Tab (forward) / Cmd+Shift+Tab (backward)로 창을 전환한다.

        Win32의 Alt+Tab hold와 같은 구조: Command를 누른 채로 Tab을 repeat번
        누른다. Quartz 키 이벤트에 Command(+backward면 Shift) 플래그를 실어
        보내며, 앱 스위처가 뜬 상태를 유지하도록 Command down/up으로 감싼다.
        """
        import Quartz

        flags = Quartz.kCGEventFlagMaskCommand
        if not forward:
            flags |= Quartz.kCGEventFlagMaskShift

        def key(code: int, key_down: bool, event_flags: int) -> None:
            event = Quartz.CGEventCreateKeyboardEvent(None, code, key_down)
            Quartz.CGEventSetFlags(event, event_flags)
            self._post(event)

        key(_KEYCODE_COMMAND, True, Quartz.kCGEventFlagMaskCommand)
        if not forward:
            key(_KEYCODE_SHIFT, True, flags)
        for _ in range(repeat):
            key(_KEYCODE_TAB, True, flags)
            key(_KEYCODE_TAB, False, flags)
        if not forward:
            key(_KEYCODE_SHIFT, False, Quartz.kCGEventFlagMaskCommand)
        key(_KEYCODE_COMMAND, False, 0)

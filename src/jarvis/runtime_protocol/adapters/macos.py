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

from jarvis.runtime_protocol.adapters.windows import InputKey, MouseButton

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
_KEYCODE_F11 = 0x67  # 바탕화면 표시


class MacOSInputSink:
    """Real OS input via Quartz/AppKit CGEvent (macOS only, manually verified).

    `Win32InputSink`와 같은 지연 import 원칙: PyObjC는 이 클래스의 메서드 안에서만
    import해, macOS가 아닌 호스트나 `macos` extra 미설치 환경에서 이 모듈을
    import하는 것 자체는 항상 성공한다(실제 호출 시에만 실패).
    """

    def _post(self, event: Any) -> None:
        import Quartz

        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def press(self, button: MouseButton, *, down: bool) -> None:
        """버튼 상태를 바꾼다. 드래그는 누른 채 이동해야 해서 click과 분리돼 있다."""
        import Quartz

        if button is MouseButton.RIGHT:
            event_type = Quartz.kCGEventRightMouseDown if down else Quartz.kCGEventRightMouseUp
            mouse_button = Quartz.kCGMouseButtonRight
        else:
            event_type = Quartz.kCGEventLeftMouseDown if down else Quartz.kCGEventLeftMouseUp
            mouse_button = Quartz.kCGMouseButtonLeft
        location = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        self._post(Quartz.CGEventCreateMouseEvent(None, event_type, location, mouse_button))

    def click(self, button: MouseButton) -> None:
        self.press(button, down=True)
        self.press(button, down=False)

    def double_click(self, button: MouseButton) -> None:
        """더블클릭을 전송한다 — 두 번의 down/up에 clickState를 실어 보낸다.

        macOS는 단순히 click을 두 번 보내는 것만으로는 더블클릭으로 인식하지 않는다.
        CGEvent의 `kCGMouseEventClickState` 필드가 각 이벤트의 '연속 클릭 횟수'를
        나타내며, 두 번째 클릭에 2를 실어야 OS가 하나의 더블클릭으로 합친다.
        """
        import Quartz

        if button is MouseButton.RIGHT:
            down_type = Quartz.kCGEventRightMouseDown
            up_type = Quartz.kCGEventRightMouseUp
            mouse_button = Quartz.kCGMouseButtonRight
        else:
            down_type = Quartz.kCGEventLeftMouseDown
            up_type = Quartz.kCGEventLeftMouseUp
            mouse_button = Quartz.kCGMouseButtonLeft
        location = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        for click_state in (1, 2):
            for event_type in (down_type, up_type):
                event = Quartz.CGEventCreateMouseEvent(
                    None, event_type, location, mouse_button
                )
                Quartz.CGEventSetIntegerValueField(
                    event, Quartz.kCGMouseEventClickState, click_state
                )
                self._post(event)

    def scroll(self, ticks: int) -> None:
        import Quartz

        event = Quartz.CGEventCreateScrollWheelEvent(
            None, Quartz.kCGScrollEventUnitLine, _SCROLL_UNIT_LINE, ticks
        )
        self._post(event)

    def tap_key(self, key: InputKey) -> None:
        """키 하나를 누름→뗌으로 전송한다.

        F11(바탕화면) 같은 표준 키는 일반 키보드 이벤트로, 볼륨·재생 같은 미디어 키는
        시스템 정의 이벤트로 보낸다 — 미디어 키는 표준 keycode 경로로는 OS가 인식하지
        않기 때문에 경로가 갈린다. Mission Control은 키 합성 대신 앱을 직접 실행한다
        (아래 `_tap_mission_control` 참조 — 키보드 단축키 설정에 의존하지 않음).
        """
        if key is InputKey.MISSION_CONTROL:
            self._tap_mission_control()
            return
        if key is InputKey.SHOW_DESKTOP:
            import Quartz

            for key_down in (True, False):
                event = Quartz.CGEventCreateKeyboardEvent(None, _KEYCODE_F11, key_down)
                self._post(event)
            return
        self._tap_media_key(key)

    def _tap_mission_control(self) -> None:
        """Mission Control을 연다 — `/System/Applications/Mission Control.app`을 실행한다.

        키 합성(Ctrl+↑ 또는 F3)은 **시스템 키보드 단축키 설정에 의존**한다: 해당 단축키가
        꺼져 있거나 다른 키로 바뀌었거나(설정 > 키보드 > 단축키 > Mission Control), 다른
        앱이 그 조합을 가로채면 아무 일도 일어나지 않는다 — 실제로 Ctrl+↑·F3가 모두 안 먹는
        환경이 확인됐다. 앱을 직접 실행하면 이 의존성이 사라져 어떤 단축키 설정에서도 열린다.

        `open`은 실패해도 예외를 던지지 않고 조용히 무시한다(제어 경로가 한 번의 실패로
        멈추지 않게) — 앱 부재 등은 반환 코드로만 드러난다.
        """
        import subprocess

        subprocess.run(
            ["open", "-a", "Mission Control"],
            check=False,
            capture_output=True,
        )

    def _tap_media_key(self, key: InputKey) -> None:
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

    def move_cursor(self, dx: int, dy: int, *, dragging: bool = False) -> None:
        """현재 커서 위치에서 상대 이동. `dragging`이면 드래그 이벤트를 보낸다.

        CGEvent의 마우스 이동은 절대좌표만 받으므로, 현재 위치를 읽어 델타를 더한 뒤
        그 절대좌표로 이벤트를 만든다. **드래그 중에는 `MouseMoved`가 아니라
        `LeftMouseDragged`를 보내야** 창·선택 영역이 실시간으로 따라온다 — MouseMoved만
        보내면 버튼을 뗄 때까지 대상이 안 움직이고 최종 위치로만 튄다.
        """
        import Quartz

        current = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        target = (current.x + dx, current.y + dy)
        event_type = Quartz.kCGEventLeftMouseDragged if dragging else Quartz.kCGEventMouseMoved
        event = Quartz.CGEventCreateMouseEvent(
            None, event_type, target, Quartz.kCGMouseButtonLeft
        )
        self._post(event)

    def screen_size(self) -> tuple[int, int]:
        """주 디스플레이 크기 — move_cursor(CGEvent)와 같은 **points** 좌표계로 낸다.

        CGDisplayBounds는 논리 좌표(points)를 돌려줘 CGEventCreateMouseEvent가 쓰는
        좌표계와 일치한다. Retina의 물리 픽셀(CGDisplayPixelsWide)이 아니라 이 값이
        커서 이동 정규화의 기준이어야 한다.
        """
        import Quartz

        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(bounds.size.width), int(bounds.size.height)

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

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

# 표준 ANSI 키보드 keycode(Carbon HIToolbox/Events.h).
_KEYCODE_COMMAND = 0x37
_KEYCODE_F11 = 0x67  # 바탕화면 표시
_KEYCODE_W = 0x0D  # 탭 닫기(Cmd+W)용


def dock_transition(
    y: float, bottom: float, edge_px: float, *, revealed: bool
) -> bool | None:
    """커서 Y가 화면 하단에 닿았는지로 Dock 노출 전환을 결정한다(순수 함수).

    Quartz·osascript에 의존하지 않아 단위 테스트할 수 있다. 반환:
        False → 지금 드러내야 함(autohide 끔). 하단 진입 & 아직 안 드러난 상태.
        True  → 지금 숨겨야 함(autohide 켬). 하단 이탈 & 드러난 상태.
        None  → 전환 없음(이미 원하는 상태). 매 프레임 osascript를 피하는 핵심.

    `bottom`은 커서가 있는 디스플레이의 하단 Y(전역 top-left 좌표, Y는 아래로 증가).
    """
    at_bottom = y >= bottom - edge_px
    if at_bottom and not revealed:
        return False  # 드러내기(autohide off)
    if not at_bottom and revealed:
        return True  # 숨기기(autohide on)
    return None


class MacOSInputSink:
    """Real OS input via Quartz/AppKit CGEvent (macOS only, manually verified).

    `Win32InputSink`와 같은 지연 import 원칙: PyObjC는 이 클래스의 메서드 안에서만
    import해, macOS가 아닌 호스트나 `macos` extra 미설치 환경에서 이 모듈을
    import하는 것 자체는 항상 성공한다(실제 호출 시에만 실패).
    """

    def __init__(self) -> None:
        # 커서가 화면 하단에 있을 때만 Dock을 드러낸다. 합성 마우스 이동은 물리 마우스와
        # 달리 Dock 자동숨김을 깨우지 못하므로(가장자리 "밀어붙임" 신호가 없다), 커서
        # 위치를 직접 감지해 Dock autohide를 토글한다. 전환(진입/이탈) 시점에만 명령을
        # 보내려고 현재 노출 상태를 기억한다 — 매 프레임 osascript를 띄우면 버벅인다.
        self._dock_revealed = False

    def _post(self, event: Any) -> None:
        import Quartz

        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def press(self, button: MouseButton, *, down: bool, click_state: int = 1) -> None:
        """버튼 상태를 바꾼다. 드래그는 누른 채 이동해야 해서 click과 분리돼 있다.

        `click_state`는 연속 클릭 횟수(1=단일, 2=더블)다. down/up 쌍에 같은 값을 실어
        보내면, 두 번째 클릭 쌍(click_state=2)을 OS가 하나의 더블클릭으로 합친다 —
        macOS는 단일 클릭을 두 번 보내는 것만으로는 더블클릭으로 인식하지 않는다
        (`double_click`의 kCGMouseEventClickState 주석과 같은 이유).
        """
        import Quartz

        if button is MouseButton.RIGHT:
            event_type = Quartz.kCGEventRightMouseDown if down else Quartz.kCGEventRightMouseUp
            mouse_button = Quartz.kCGMouseButtonRight
        else:
            event_type = Quartz.kCGEventLeftMouseDown if down else Quartz.kCGEventLeftMouseUp
            mouse_button = Quartz.kCGMouseButtonLeft
        location = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        event = Quartz.CGEventCreateMouseEvent(None, event_type, location, mouse_button)
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, click_state)
        self._post(event)

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
        if key is InputKey.CLOSE_TAB:
            self._tap_close_tab()
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

    def _tap_close_tab(self) -> None:
        """Cmd+W로 현재 탭(또는 창)을 닫는다.

        `switch_desktop`의 Ctrl+화살표와 같은 구조: Command 플래그를 실은 W 키를
        누름→뗌으로 보낸다. 표준 keycode 경로라 미디어 키(NSEvent)와 달리
        Quartz 키보드 이벤트로 충분하다. 앱이 정의한 '탭 또는 창 닫기'로 동작한다.
        """
        import Quartz

        flags = Quartz.kCGEventFlagMaskCommand

        def key(code: int, key_down: bool) -> None:
            event = Quartz.CGEventCreateKeyboardEvent(None, code, key_down)
            Quartz.CGEventSetFlags(event, flags)
            self._post(event)

        key(_KEYCODE_COMMAND, True)
        key(_KEYCODE_W, True)
        key(_KEYCODE_W, False)
        key(_KEYCODE_COMMAND, False)

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
        # 목표를 전체 화면 경계로 클램프한다. CGEventGetLocation은 직전에 **설정한** 좌표를
        # 되돌려주므로(하드웨어 클램프 반영 안 됨), 클램프하지 않으면 같은 방향 이동이
        # 누적돼 커서 논리 위치가 화면 밖으로 무한정 드리프트한다(실측: y가 -4482까지) —
        # 화면 밖으로 나가면 Dock·핫코너가 트리거되지 않고, 반대로 움직여도 그만큼
        # "되감아야" 화면에 돌아온다.
        min_x, min_y, max_x, max_y = self._desktop_bounds()
        tx = min(max(current.x + dx, min_x), max_x - 1)
        ty = min(max(current.y + dy, min_y), max_y - 1)
        target = (tx, ty)
        event_type = Quartz.kCGEventLeftMouseDragged if dragging else Quartz.kCGEventMouseMoved
        event = Quartz.CGEventCreateMouseEvent(
            None, event_type, target, Quartz.kCGMouseButtonLeft
        )
        self._post(event)
        # 클램프된 목표 = 실제 커서 위치이므로 그대로 Dock 노출 판정에 쓴다.
        self._update_dock_reveal(tx, ty)

    _DOCK_EDGE_PX = 2  # 이 이내로 하단에 닿으면 "맨 아래"로 본다

    def _update_dock_reveal(self, x: float, y: float) -> None:
        """커서가 있는 디스플레이의 하단에 닿으면 Dock을 드러내고, 벗어나면 숨긴다.

        `x, y`는 move_cursor가 클램프한 실제 커서 위치(전역 top-left 좌표). 전환 시점에만
        토글한다(진입 시 한 번 노출, 이탈 시 한 번 숨김). 듀얼 모니터에서 커서가 있는 화면
        기준으로 판정하므로, 각 화면 하단에서 그 화면의 Dock이 나온다.
        """
        import Quartz

        err, display_ids, count = Quartz.CGGetDisplaysWithPoint((x, y), 1, None, None)
        display_id = display_ids[0] if (err == 0 and count) else Quartz.CGMainDisplayID()
        bounds = Quartz.CGDisplayBounds(display_id)
        bottom = bounds.origin.y + bounds.size.height  # 전역 좌표 하단(Y는 아래로 증가)

        # 토글할지·어느 방향인지는 순수 함수로 결정한다(테스트 가능). 전환이 없으면 None.
        hide = dock_transition(y, bottom, self._DOCK_EDGE_PX, revealed=self._dock_revealed)
        if hide is None:
            return
        self._set_dock_autohide(hide=hide)
        self._dock_revealed = not hide

    @staticmethod
    def _set_dock_autohide(*, hide: bool) -> None:
        """Dock 자동숨김을 켜고 끈다 — System Events로 부드럽게(killall 불필요).

        실패해도 예외를 던지지 않는다(제어 경로가 Dock 토글 하나로 멈추지 않게).
        """
        import subprocess

        value = "true" if hide else "false"
        subprocess.run(
            ["osascript", "-e",
             f"tell application \"System Events\" to set autohide of dock preferences to {value}"],
            check=False,
            capture_output=True,
            timeout=2,
        )

    def restore_dock(self) -> None:
        """드러낸 Dock을 원래(숨김)대로 되돌린다 — 제어를 끄거나 종료할 때 호출한다.

        커서가 하단에 머문 채 제어가 꺼지면 Dock이 노출된 채 남으므로, 사용자의 원래
        상태(자동숨김)로 확실히 복구한다.
        """
        if self._dock_revealed:
            self._set_dock_autohide(hide=True)
            self._dock_revealed = False

    def _desktop_bounds(self) -> tuple[float, float, float, float]:
        """활성 디스플레이 전체를 감싸는 경계 (min_x, min_y, max_x, max_y). 전역 top-left 좌표.

        디스플레이 구성은 자주 안 바뀌므로 한 번 계산해 캐싱한다.
        """
        cached = getattr(self, "_desktop_bounds_cache", None)
        if cached is not None:
            return cached
        import Quartz

        err, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
        if err != 0 or not count:
            b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            cached = (b.origin.x, b.origin.y, b.origin.x + b.size.width, b.origin.y + b.size.height)
        else:
            bounds = [Quartz.CGDisplayBounds(did) for did in ids[:count]]
            cached = (
                min(b.origin.x for b in bounds),
                min(b.origin.y for b in bounds),
                max(b.origin.x + b.size.width for b in bounds),
                max(b.origin.y + b.size.height for b in bounds),
            )
        self._desktop_bounds_cache = cached
        return cached

    def switch_desktop(self, forward: bool, repeat: int) -> None:
        """옆 Space(가상 데스크톱)로 전환한다. forward=왼쪽(←), else 오른쪽(→).

        방향은 제스처와 반대로 매핑한다(사용자 지시): 오른쪽 슬라이드(forward)는
        왼쪽 Space로, 왼쪽 슬라이드는 오른쪽 Space로 간다.

        **System Events(osascript)로 Ctrl+←/→ 키 코드를 보낸다.** 실측 결론:
        우리 프로세스가 직접 만든 합성 키(CGEvent, HID·session·annotated tap,
        flag-only 등 전부)는 커서·물리 키는 되는데도 Space 전환만 WindowServer가
        무시했다. SkyLight로 공간을 직접 바꾸면(`CGSManagedDisplaySetCurrentSpace`)
        전환은 되지만 창↔Space 소속이 꼬여 미션 컨트롤에 창이 바탕으로 남는
        아티팩트가 생긴다. 반면 Apple 자체 프로세스인 System Events가 보낸 키는
        **진짜 Ctrl+화살표로 인정**돼 네이티브 그대로(애니메이션·창 관리 정상)
        전환된다 — 물리 키를 누른 것과 동일하다.

        "Mission Control > 이전/다음 Space로 이동" 단축키(기본 Ctrl+←/→)와, 이
        프로세스가 System Events를 제어할 **자동화(Automation) 권한**에 의존한다
        (첫 호출 시 권한 프롬프트가 뜰 수 있다). `_tap_mission_control`과 같은
        규약으로, osascript 실패는 예외 없이 조용히 무시한다(제어 경로가 한 번의
        실패로 멈추지 않게).
        """
        import subprocess

        key_code = 123 if forward else 124  # forward=←(123), backward=→(124), 제스처와 반대
        script = (
            f'tell application "System Events" to key code {key_code} using control down'
        )
        for _ in range(repeat):
            subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
            )

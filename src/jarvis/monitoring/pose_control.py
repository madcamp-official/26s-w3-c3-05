"""자세 상태기계 이벤트 → 실제 OS 입력. 디버깅 툴이 컴퓨터를 제어하는 지점.

`PoseStateMachine`은 순수 로직이라 아무것도 실행하지 않는다. 이 모듈이 그 이벤트를
`InputSink`(플랫폼별 실제 입력)로 옮긴다 — 판정과 실행을 분리해 두면 상태기계를
카메라·OS 없이 테스트할 수 있고, 실행을 끄고도 판정을 관찰할 수 있다.

**기본값은 꺼짐이다.** 이 경로는 사용자의 실제 데스크톱을 클릭하고 스크롤한다.
분류 정확도가 87.1%이고 오발동이 1.7% 남아 있는 상태에서 기본으로 켜면, 디버깅 툴을
띄우는 것만으로 의도치 않은 클릭이 일어날 수 있다. 켜고 끄는 것은 사용자가 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.gesture_fusion.pose_state import PoseEvent
from jarvis.runtime_protocol.adapters.windows import InputKey, InputSink, MouseButton

# 스크롤 이벤트는 매 프레임 나온다(30fps). 그대로 보내면 초당 30틱이라 너무 빠르므로
# 이 간격으로 솎아낸다. 값이 커지면 스크롤이 느려지고, 작아지면 걷잡을 수 없어진다.
SCROLL_INTERVAL_MS = 60
SCROLL_TICKS = 1


def default_input_sink() -> InputSink | None:
    """플랫폼에 맞는 입력 싱크. 지원하지 않는 OS에서는 None(제어를 흉내내지 않는다)."""
    import sys

    if sys.platform == "darwin":
        from jarvis.runtime_protocol.adapters.macos import MacOSInputSink

        return MacOSInputSink()
    if sys.platform.startswith("win"):
        from jarvis.runtime_protocol.adapters.windows import Win32InputSink

        return Win32InputSink()
    return None


@dataclass
class PoseControlBridge:
    """이벤트를 입력으로 실행하고, 무엇을 했는지 기록한다(UI 표시용)."""

    sink: InputSink | None = None
    enabled: bool = False
    last_action: str = ""
    _last_scroll_ms: int = 0
    _dragging: bool = False
    history: list[str] = field(default_factory=list)

    def apply(self, events: list[PoseEvent]) -> None:
        """이벤트를 실행한다. 꺼져 있으면 기록만 남기고 아무것도 실행하지 않는다."""
        for event in events:
            # 스크롤은 30fps 그대로면 감당이 안 돼 솎아낸다 — 실행·기록 **양쪽**에
            # 적용해야 한다(예전엔 스로틀이 _describe에만 있어 실행 판단과 얽혔다).
            if event.kind == "scroll":
                if event.timestamp_ms - self._last_scroll_ms < SCROLL_INTERVAL_MS:
                    continue
                self._last_scroll_ms = event.timestamp_ms
            if self.enabled and self.sink is not None:
                self._execute(event)
            label = self._describe(event)
            if not label:
                continue  # move처럼 실행은 하되 UI 기록만 생략하는 경우
            self.last_action = label if self.enabled else f"{label} (실행 안 함)"
            self.history.append(self.last_action)
            del self.history[:-20]

    def release(self) -> None:
        """드래그 중 손을 놓치거나 제어를 끌 때 버튼이 눌린 채로 남지 않게 한다."""
        if self._dragging and self.sink is not None:
            self.sink.press(MouseButton.LEFT, down=False)
        self._dragging = False

    def _describe(self, event: PoseEvent) -> str:
        if event.kind == "move":
            return ""  # 커서 이동은 매 프레임이라 기록/표시하지 않는다(로그가 뒤덮인다)
        if event.kind == "scroll":
            return f"스크롤 {'위' if event.value > 0 else '아래'}"
        return {
            "click": "클릭",
            "right_click": "우클릭",
            "drag_start": "드래그 시작",
            "drag_end": "드래그 끝",
            # 임시: 재생/일시정지 대신 F11(바탕화면 표시). 주먹→보 전이마다 토글된다.
            "media_toggle": "바탕화면 (F11)",
        }.get(event.kind, "")

    def _execute(self, event: PoseEvent) -> None:
        assert self.sink is not None
        if event.kind == "move":
            self.sink.move_cursor(round(event.delta[0]), round(event.delta[1]))
        elif event.kind == "click":
            self.sink.click(MouseButton.LEFT)
        elif event.kind == "right_click":
            self.sink.click(MouseButton.RIGHT)
        elif event.kind == "drag_start":
            self.sink.press(MouseButton.LEFT, down=True)
            self._dragging = True
        elif event.kind == "drag_end":
            self.sink.press(MouseButton.LEFT, down=False)
            self._dragging = False
        elif event.kind == "media_toggle":
            self.sink.tap_key(InputKey.SHOW_DESKTOP)
        elif event.kind == "scroll":
            self.sink.scroll(SCROLL_TICKS if event.value > 0 else -SCROLL_TICKS)

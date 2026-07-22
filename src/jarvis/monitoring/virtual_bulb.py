"""화면상 전구 표시 — 실물에서 읽은 상태를 기준으로, 그 위에 보낸 명령을 얹는다.

실물 전구(Philips WiZ, 로컬 UDP)는 같은 네트워크·전원·설정이 모두 갖춰져야
반응한다. 그 중 하나라도 빠지면 전구 시나리오(같은 slide_down이 노트북에선
스크롤, 전구에선 밝기 감소)가 통째로 빈다. 그래서 이 상태 객체는 실물이 없어도
**dispatch된 Intent를 누적**해 그 명령이 무엇이었는지는 보여준다.

다만 명령 누적만으로는 화면이 실물과 어긋난다 — 시작 시점의 실제 밝기·색을
모르는 채 임의의 초기값에서 출발하기 때문이다(2026-07-22 시연에서 "표시가 실물과
하나도 안 맞는다"로 드러났다). `state_from_pilot()`이 그 공백을 메운다: `getPilot`
응답을 그대로 이 상태로 옮겨, 시작할 때와 명령을 보낸 뒤에 실물 값으로 맞춘다.

정직성 경계는 그대로다: 이 값이 **실물에서 읽은 것인지 보낸 명령 기준인지**를 UI가
구분해 표시해야 한다(`DemoPanel.set_bulb(..., verified=...)`). 실물이 실패했는데
그림만 밝아지는 것을 "전구가 켜졌다"로 읽히게 두면 성공을 지어내는 것이다.

값의 범위·step은 `jarvis.runtime.devices._bulb_profile()`의 capability model과
같다 — 거기서 검증을 통과한 명령만 여기 도달하므로 두 정의가 어긋나면 화면과
실물이 갈린다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from jarvis.contracts.messages import Intent
from jarvis.monitoring.demo_bridge import BULB_DEVICE_ID
from jarvis.runtime_protocol.adapters.wiz import rgb_to_hue
from jarvis.runtime_protocol.protocol.capability import Operation

# 하한이 0이 아니라 10인 것은 실측 WiZ 모델의 `minDimLevel`이 10이기 때문이다 —
# 0은 전구가 거부하며, 끄기는 power capability의 몫이다
# (`jarvis.runtime.devices._bulb_profile()`과 같은 값이어야 화면과 실물이 갈리지 않는다).
BRIGHTNESS_MIN = 10
BRIGHTNESS_MAX = 100
COLOR_TEMPERATURE_MIN = 2700
COLOR_TEMPERATURE_MAX = 6500

# 색상은 순환량이라 상·하한이 벽이 아니라 한 바퀴의 경계다 — 클램프가 아니라 감아 돈다
# (WizAdapter `_apply_color`와 같은 규약).
HUE_DEGREES = 360


def _clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, value)))


@dataclass
class VirtualBulbState:
    """화면에 그릴 전구 상태.

    초기값은 아무 근거 없는 임의의 중간값이다 — 실물을 읽기 전에는 "지금 실물이
    이렇다"고 주장할 수 없다. 그래서 앱은 시작하자마자 `state_from_pilot()`으로
    이 값을 실물에서 읽은 것으로 갈아끼우고, UI는 그 전까지 "확인 전"으로 표시한다.
    """

    power: bool = True
    brightness: int = 60
    color_temperature: int = 4000
    hue: int = 0
    """색상각(도). `color_mode`가 True일 때만 화면에 반영된다."""

    color_mode: bool = False
    """RGB(색상) 모드인가, CCT(색온도) 모드인가.

    실물 WiZ는 둘 중 하나로만 켜진다 — r/g/b를 보내면 색상 모드로, temp를 보내면
    색온도 모드로 넘어간다. 어느 쪽 명령을 마지막에 보냈는지를 그대로 따라가야
    화면과 실물이 갈리지 않는다.
    """

    def apply(self, intent: Intent) -> bool:
        """전구를 향한 Intent 하나를 반영한다. 반영했으면 True.

        다른 기기(`laptop` 등)의 Intent나 모르는 capability/operation은 조용히
        무시하고 False를 돌려준다 — 알 수 없는 명령을 추측해 상태를 바꾸지 않는다.
        """
        if intent.target != BULB_DEVICE_ID:
            return False
        if intent.capability == "power":
            return self._apply_power(intent)
        if intent.capability == "brightness":
            return self._apply_number(intent, "brightness", BRIGHTNESS_MIN, BRIGHTNESS_MAX)
        if intent.capability == "color_temperature":
            applied = self._apply_number(
                intent, "color_temperature", COLOR_TEMPERATURE_MIN, COLOR_TEMPERATURE_MAX
            )
            if applied:
                self.color_mode = False  # temp를 보내면 실물도 CCT 모드로 돌아간다
            return applied
        if intent.capability == "color":
            return self._apply_hue(intent)
        return False

    def _apply_hue(self, intent: Intent) -> bool:
        """색상각. 다른 수치와 달리 클램프가 아니라 **순환**한다 — 회전을 계속하면 한 바퀴 돈다."""
        if isinstance(intent.value, bool) or not isinstance(intent.value, (int, float)):
            return False
        delta = float(intent.value)
        if intent.operation == Operation.INCREMENT:
            updated = self.hue + delta
        elif intent.operation == Operation.DECREMENT:
            updated = self.hue - delta
        elif intent.operation == Operation.SET:
            updated = delta
        else:
            return False
        self.hue = int(round(updated)) % HUE_DEGREES
        self.color_mode = True  # r/g/b를 보내면 실물도 색상 모드로 넘어간다
        return True

    def _apply_power(self, intent: Intent) -> bool:
        if intent.operation == Operation.TOGGLE:
            self.power = not self.power
            return True
        if intent.operation == Operation.SET and isinstance(intent.value, bool):
            self.power = intent.value
            return True
        return False

    def _apply_number(self, intent: Intent, attribute: str, low: int, high: int) -> bool:
        if isinstance(intent.value, bool) or not isinstance(intent.value, (int, float)):
            return False
        current = float(getattr(self, attribute))
        delta = float(intent.value)
        if intent.operation == Operation.INCREMENT:
            updated = current + delta
        elif intent.operation == Operation.DECREMENT:
            updated = current - delta
        elif intent.operation == Operation.SET:
            updated = delta
        else:
            return False
        setattr(self, attribute, _clamp(updated, low, high))
        return True

    def describe(self) -> str:
        """한 줄 요약(로그·툴팁용). 지금 어느 모드인지에 따라 색상/색온도를 골라 보여준다."""
        if not self.power:
            return "전원 꺼짐"
        tint = (
            f"색상 {self.hue}° ({hue_name(self.hue)})"
            if self.color_mode
            else f"색온도 {self.color_temperature}K"
        )
        return f"밝기 {self.brightness}% · {tint}"


def _number(state: Mapping[str, object], key: str) -> float | None:
    """getPilot의 수치 필드 하나. 없거나 수치가 아니면 None — 값을 지어내지 않는다."""
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def state_from_pilot(result: Mapping[str, object]) -> VirtualBulbState | None:
    """WiZ `getPilot` 응답 → 화면 상태. 전원조차 못 읽으면 ``None``.

    실물 WiZ는 색상(RGB) 모드와 색온도(CCT) 모드 중 **하나로만** 켜져 있고,
    getPilot도 그 모드의 필드만 돌려준다 — RGB 모드면 ``r/g/b``, CCT 모드면
    ``temp``. 그래서 어느 필드가 왔는지가 곧 모드다. 둘 다 없으면(장면 모드 등)
    화면이 마지막으로 알던 모드를 유지하는 대신 기본값을 쓴다 — 어차피 다음
    명령 한 번이면 실제 모드가 확정된다.

    읽히지 않은 필드는 기본값으로 남는다. 이 함수는 "읽은 만큼만" 반영하고,
    읽지 못한 것을 추측하지 않는다.
    """
    power = result.get("state")
    if not isinstance(power, bool):
        return None
    bulb = VirtualBulbState(power=power)

    dimming = _number(result, "dimming")
    if dimming is not None:
        bulb.brightness = _clamp(dimming, BRIGHTNESS_MIN, BRIGHTNESS_MAX)

    red, green, blue = (_number(result, param) for param in ("r", "g", "b"))
    temperature = _number(result, "temp")
    if any(component for component in (red, green, blue)):
        # 하나라도 0이 아닌 성분이 오면 RGB 모드다. 전부 0이거나 없으면 색이 아니라
        # 흰색/꺼짐이므로 CCT 쪽으로 넘긴다(rgb_to_hue도 무채색을 0도로 본다).
        bulb.hue = int(round(rgb_to_hue(red or 0.0, green or 0.0, blue or 0.0)))
        bulb.color_mode = True
    elif temperature is not None:
        bulb.color_temperature = _clamp(
            temperature, COLOR_TEMPERATURE_MIN, COLOR_TEMPERATURE_MAX
        )
        bulb.color_mode = False
    return bulb


# 색상각 → 사람이 읽는 이름. 60도마다 이름이 바뀌므로 시연에서 한 스텝의 변화가
# 이름 수준에서 드러난다("빨강 → 노랑"), 숫자만 볼 때보다 무대에서 확인하기 쉽다.
_HUE_NAMES: tuple[tuple[int, str], ...] = (
    (15, "빨강"),
    (45, "주황"),
    (75, "노랑"),
    (165, "초록"),
    (195, "청록"),
    (255, "파랑"),
    (285, "보라"),
    (345, "자홍"),
)


def hue_name(hue_deg: int) -> str:
    for upper, name in _HUE_NAMES:
        if hue_deg % HUE_DEGREES < upper:
            return name
    return "빨강"  # 345도 이상은 다시 빨강으로 감긴다

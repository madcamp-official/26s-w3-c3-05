"""화면상 가상 전구 — 시연에서 "명령이 무엇이었는가"를 눈에 보이게 한다.

실물 전구(Philips WiZ, 로컬 UDP)는 같은 네트워크·전원·설정이 모두 갖춰져야
반응한다. 그 중 하나라도 빠지면 전구 시나리오(같은 slide_down이 노트북에선
스크롤, 전구에선 밝기 감소)가 통째로 빈다. 이 상태 객체는 **dispatch된 Intent를
그대로 누적**해 그 명령이 무엇이었는지를 보여준다.

정직성 경계가 중요하다: 이 값은 **명령 기준**이지 실물의 응답이 아니다. 실물이
성공했는지는 `ExecutionOutcome`이 따로 말해 주며, UI는 그 둘을 분리해 표시해야
한다(가상 상태 + 실물 결과 배지). 실물이 실패했는데 가상 전구가 밝아지는 것을
"전구가 켜졌다"로 읽히게 두면 성공을 지어내는 것이다.

값의 범위·step은 `jarvis.runtime.devices._bulb_profile()`의 capability model과
같다 — 거기서 검증을 통과한 명령만 여기 도달하므로 두 정의가 어긋나면 화면과
실물이 갈린다.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.contracts.messages import Intent
from jarvis.monitoring.demo_bridge import BULB_DEVICE_ID
from jarvis.runtime_protocol.protocol.capability import Operation

# 하한이 0이 아니라 10인 것은 실측 WiZ 모델의 `minDimLevel`이 10이기 때문이다 —
# 0은 전구가 거부하며, 끄기는 power capability의 몫이다
# (`jarvis.runtime.devices._bulb_profile()`과 같은 값이어야 화면과 실물이 갈리지 않는다).
BRIGHTNESS_MIN = 10
BRIGHTNESS_MAX = 100
COLOR_TEMPERATURE_MIN = 2700
COLOR_TEMPERATURE_MAX = 6500


def _clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, value)))


@dataclass
class VirtualBulbState:
    """전구의 **명령 기준** 상태. 실물 응답이 아니다.

    초기값은 SmartThings 기본 프리셋과 무관한 임의의 중간값이다 — 실물 상태를
    읽어오지 않으므로 "지금 실물이 이렇다"고 주장하지 않는다. 시연에서 의미 있는
    것은 절대값이 아니라 제스처에 따른 **변화**다(밝기 capability 자체가 상대
    연산 전용이다).
    """

    power: bool = True
    brightness: int = 60
    color_temperature: int = 4000

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
            return self._apply_number(
                intent, "color_temperature", COLOR_TEMPERATURE_MIN, COLOR_TEMPERATURE_MAX
            )
        return False

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
        """한 줄 요약(로그·툴팁용)."""
        if not self.power:
            return "전원 꺼짐"
        return f"밝기 {self.brightness}% · 색온도 {self.color_temperature}K"

"""가상 전구 상태 — 명령 누적·클램프·무시 규칙을 네트워크 없이 검증한다.

이 상태는 **보낸 명령 기준**이지 실물의 응답이 아니다(virtual_bulb.py docstring).
따라서 여기서 검증하는 것은 "실물이 이렇게 됐다"가 아니라 "이런 명령을 보냈다"가
화면에 어떻게 누적되는가다.
"""

from __future__ import annotations

from jarvis.contracts.messages import Intent
from jarvis.monitoring.virtual_bulb import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    COLOR_TEMPERATURE_MAX,
    COLOR_TEMPERATURE_MIN,
    VirtualBulbState,
)


def _intent(
    *,
    target: str = "room.bulb",
    capability: str = "brightness",
    operation: str = "decrement",
    value: int | float | bool = 10,
) -> Intent:
    return Intent(
        intent_id="intent-1",
        target=target,
        gesture="slide_two_fingers_down",
        capability=capability,
        operation=operation,
        value=value,
        target_confidence=0.9,
        gesture_confidence=0.9,
        expires_in_ms=1000,
    )


# --- 상대 연산 누적 ---


def test_brightness_decrement_accumulates() -> None:
    bulb = VirtualBulbState(brightness=60)
    assert bulb.apply(_intent(operation="decrement", value=10))
    assert bulb.brightness == 50
    assert bulb.apply(_intent(operation="decrement", value=10))
    assert bulb.brightness == 40


def test_brightness_increment_accumulates() -> None:
    bulb = VirtualBulbState(brightness=60)
    assert bulb.apply(_intent(operation="increment", value=10))
    assert bulb.brightness == 70


def test_color_temperature_moves_by_intent_value() -> None:
    bulb = VirtualBulbState(color_temperature=4000)
    assert bulb.apply(_intent(capability="color_temperature", operation="increment", value=100))
    assert bulb.color_temperature == 4100


# --- 클램프: capability model의 범위를 넘지 않는다 ---


def test_brightness_clamps_at_device_minimum() -> None:
    """하한은 0이 아니라 10 — 실측 WiZ의 minDimLevel이다(끄기는 power의 몫)."""
    bulb = VirtualBulbState(brightness=BRIGHTNESS_MIN)
    bulb.apply(_intent(operation="decrement", value=10))
    assert bulb.brightness == BRIGHTNESS_MIN


def test_brightness_clamps_at_maximum() -> None:
    bulb = VirtualBulbState(brightness=BRIGHTNESS_MAX)
    bulb.apply(_intent(operation="increment", value=10))
    assert bulb.brightness == BRIGHTNESS_MAX


def test_color_temperature_clamps_both_ends() -> None:
    bulb = VirtualBulbState(color_temperature=COLOR_TEMPERATURE_MIN)
    bulb.apply(_intent(capability="color_temperature", operation="decrement", value=100))
    assert bulb.color_temperature == COLOR_TEMPERATURE_MIN

    bulb = VirtualBulbState(color_temperature=COLOR_TEMPERATURE_MAX)
    bulb.apply(_intent(capability="color_temperature", operation="increment", value=100))
    assert bulb.color_temperature == COLOR_TEMPERATURE_MAX


# --- power ---


def test_power_toggle_flips() -> None:
    bulb = VirtualBulbState(power=True)
    assert bulb.apply(_intent(capability="power", operation="toggle", value=True))
    assert bulb.power is False
    assert bulb.apply(_intent(capability="power", operation="toggle", value=True))
    assert bulb.power is True


def test_power_set_uses_boolean_value() -> None:
    bulb = VirtualBulbState(power=True)
    assert bulb.apply(_intent(capability="power", operation="set", value=False))
    assert bulb.power is False


# --- 모르는 것은 추측하지 않는다 ---


def test_other_device_intent_is_ignored() -> None:
    """노트북 명령이 전구 그림을 바꾸면 안 된다."""
    bulb = VirtualBulbState(brightness=60)
    assert bulb.apply(_intent(target="laptop", capability="scroll")) is False
    assert bulb.brightness == 60


def test_unknown_capability_is_ignored() -> None:
    bulb = VirtualBulbState(brightness=60)
    assert bulb.apply(_intent(capability="mystery")) is False
    assert bulb.brightness == 60


def test_unknown_operation_is_ignored() -> None:
    bulb = VirtualBulbState(brightness=60)
    assert bulb.apply(_intent(operation="rotate")) is False
    assert bulb.brightness == 60


def test_boolean_value_rejected_for_number_capability() -> None:
    """bool은 int의 하위 타입이라 그냥 두면 True가 1로 더해진다 — 명시적으로 막는다."""
    bulb = VirtualBulbState(brightness=60)
    assert bulb.apply(_intent(capability="brightness", operation="increment", value=True)) is False
    assert bulb.brightness == 60


# --- 표시 ---


def test_describe_distinguishes_power_off() -> None:
    assert VirtualBulbState(power=False).describe() == "전원 꺼짐"
    assert "밝기" in VirtualBulbState(power=True, brightness=40).describe()

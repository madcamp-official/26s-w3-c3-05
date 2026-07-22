"""전구 표시 상태 — 실물 readback 반영과 명령 누적·클램프·무시 규칙을 검증한다.

두 입력이 있다: 실물 `getPilot` 응답(`state_from_pilot`)과 보낸 명령(`apply`).
전자는 "실물이 지금 이렇다", 후자는 "이런 명령을 보냈다"이며 UI는 둘을 구분해
표시한다(virtual_bulb.py docstring). 네트워크 없이 둘 다 검증한다.
"""

from __future__ import annotations

from jarvis.contracts.messages import Intent
from jarvis.monitoring.virtual_bulb import (
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    COLOR_TEMPERATURE_MAX,
    COLOR_TEMPERATURE_MIN,
    VirtualBulbState,
    state_from_pilot,
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


# --- 색상(hue): 순환하고, 모드를 전환한다 ---


def test_hue_cycles_forward_through_a_full_turn() -> None:
    """회전을 6번이면 한 바퀴 — 클램프하면 시연에서 색이 양 끝에 붙어 죽는다."""
    bulb = VirtualBulbState()
    seen = []
    for _ in range(6):
        bulb.apply(_intent(capability="color", operation="increment", value=60))
        seen.append(bulb.hue)
    assert seen == [60, 120, 180, 240, 300, 0]


def test_hue_wraps_backward_below_zero() -> None:
    bulb = VirtualBulbState(hue=0)
    bulb.apply(_intent(capability="color", operation="decrement", value=60))
    assert bulb.hue == 300


def test_color_command_switches_to_color_mode() -> None:
    """실물 WiZ는 r/g/b를 받으면 CCT 모드에서 색상 모드로 넘어간다."""
    bulb = VirtualBulbState()
    assert bulb.color_mode is False
    bulb.apply(_intent(capability="color", operation="increment", value=60))
    assert bulb.color_mode is True


def test_color_temperature_command_switches_back_to_cct_mode() -> None:
    bulb = VirtualBulbState(color_mode=True, hue=120)
    bulb.apply(_intent(capability="color_temperature", operation="increment", value=100))
    assert bulb.color_mode is False


def test_brightness_does_not_change_the_mode() -> None:
    bulb = VirtualBulbState(color_mode=True, hue=120)
    bulb.apply(_intent(capability="brightness", operation="decrement", value=30))
    assert bulb.color_mode is True
    assert bulb.hue == 120


def test_hue_rejects_boolean_value() -> None:
    bulb = VirtualBulbState()
    assert bulb.apply(_intent(capability="color", operation="increment", value=True)) is False
    assert bulb.hue == 0


def test_describe_follows_the_active_mode() -> None:
    assert "색온도" in VirtualBulbState(color_mode=False).describe()
    described = VirtualBulbState(color_mode=True, hue=120).describe()
    assert "색상" in described and "초록" in described


def test_hue_name_covers_the_whole_circle() -> None:
    """이름 없는 각도가 없어야 한다 — 시연에서 '색상 210°'만 뜨면 읽히지 않는다."""
    from jarvis.monitoring.virtual_bulb import hue_name

    assert all(hue_name(deg) for deg in range(0, 360, 5))
    assert hue_name(0) == hue_name(359) == "빨강"  # 원형으로 이어진다


# --- 실물 상태 읽기(state_from_pilot) --------------------------------------
#
# 화면이 실물과 어긋난 근본 원인은 "실물을 한 번도 읽지 않는다"였다. 아래는 그
# 읽기가 WiZ의 실제 응답 모양을 정확히 옮기는지 고정한다.


def test_pilot_cct_mode_is_read_as_color_temperature() -> None:
    """CCT 모드 응답에는 temp만 온다 — 그대로 색온도 모드가 돼야 한다."""
    state = state_from_pilot(
        {"mac": "9877d5cffaf8", "state": True, "dimming": 40, "temp": 4200, "sceneId": 0}
    )
    assert state is not None
    assert state.power is True
    assert state.brightness == 40
    assert state.color_temperature == 4200
    assert state.color_mode is False


def test_pilot_rgb_mode_is_read_as_hue() -> None:
    """RGB 모드 응답에는 r/g/b가 온다 — 각도로 환산하고 색상 모드가 돼야 한다."""
    state = state_from_pilot(
        {"state": True, "dimming": 100, "r": 0, "g": 255, "b": 0, "c": 0, "w": 0}
    )
    assert state is not None
    assert state.color_mode is True
    assert state.hue == 120  # 초록
    assert "초록" in state.describe()


def test_pilot_off_state_is_read_as_off() -> None:
    state = state_from_pilot({"state": False, "dimming": 60, "temp": 2700})
    assert state is not None
    assert state.power is False
    assert state.describe() == "전원 꺼짐"


def test_pilot_without_power_field_is_not_guessed() -> None:
    """전원조차 못 읽으면 화면을 갱신하지 않는다 — 값을 지어내지 않는다."""
    assert state_from_pilot({"mac": "9877d5cffaf8", "rssi": -60}) is None


def test_pilot_all_zero_rgb_falls_back_to_color_temperature() -> None:
    """r=g=b=0은 색이 아니라 무채색이다 — 빨강(0도)으로 오해하면 안 된다."""
    state = state_from_pilot({"state": True, "r": 0, "g": 0, "b": 0, "temp": 6000})
    assert state is not None
    assert state.color_mode is False
    assert state.color_temperature == 6000


def test_pilot_values_are_clamped_to_the_capability_range() -> None:
    """기기가 프로파일 밖 값을 보고해도 화면 보간이 깨지지 않게 경계로 접는다."""
    state = state_from_pilot({"state": True, "dimming": 255, "temp": 9000})
    assert state is not None
    assert state.brightness == BRIGHTNESS_MAX
    assert state.color_temperature == COLOR_TEMPERATURE_MAX


def test_pilot_missing_fields_keep_defaults_without_crashing() -> None:
    state = state_from_pilot({"state": True})
    assert state is not None
    assert state.brightness == VirtualBulbState().brightness

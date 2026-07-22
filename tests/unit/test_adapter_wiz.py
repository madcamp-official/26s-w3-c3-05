"""WiZ 전구 adapter — 명령 매핑·클램프·정직한 상태 보고를 네트워크 없이 검증한다."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.adapters.base import AdapterStatus
from jarvis.runtime_protocol.adapters.wiz import (
    UdpWizTransport,
    hue_to_rgb,
    rgb_to_hue,
    WizAdapter,
    WizConfig,
    WizTimeout,
    _is_mac,
    parse_target,
)
from jarvis.runtime_protocol.protocol.capability import (
    BooleanCapability,
    DeviceProfile,
    NumberCapability,
)

TARGET = "9877d5cffaf8"


def _profile() -> DeviceProfile:
    return DeviceProfile(
        device_id="room.bulb",
        adapter="wiz",
        capabilities={
            "power": BooleanCapability(),
            "brightness": NumberCapability(minimum=10, maximum=100, step=10),
            "color_temperature": NumberCapability(minimum=2700, maximum=6500, step=100),
            "color": NumberCapability(
                minimum=0, maximum=360, step=30,
                operations=frozenset({"increment", "decrement"}),
            ),
        },
    )


def _command(capability: str, operation: str, value: object) -> Command:
    return Command(
        command_id="cmd-1",
        intent_id="intent-1",
        device_id="room.bulb",
        capability=capability,
        operation=operation,
        value=value,  # type: ignore[arg-type]
        expires_at_ms=10_000,
    )


class FakeTransport:
    """getPilot에 돌려줄 상태를 들고 있다가 setPilot을 그 상태에 반영한다."""

    def __init__(self, state: dict[str, object] | None = None, fail: Exception | None = None) -> None:
        self.state: dict[str, object] = state if state is not None else {"state": True, "dimming": 50, "temp": 4000}
        self.sent: list[dict[str, object]] = []
        self._fail = fail

    def send(
        self, target: str, payload: Mapping[str, object], timeout_s: float
    ) -> Mapping[str, object]:
        if self._fail is not None:
            raise self._fail
        method = payload.get("method")
        params = dict(payload.get("params") or {})  # type: ignore[arg-type]
        if method == "getPilot":
            return {"result": dict(self.state)}
        if method == "setPilot":
            self.sent.append(params)
            self.state.update(params)
            return {"result": {"success": True}}
        raise AssertionError(f"unexpected method {method!r}")


def _adapter(transport: FakeTransport) -> WizAdapter:
    return WizAdapter(WizConfig({"room.bulb": TARGET}), transport)


# --- configuration ---------------------------------------------------------


def test_unconfigured_when_no_config() -> None:
    result = WizAdapter(None, FakeTransport()).execute(_command("power", "set", True), _profile())
    assert result.status is AdapterStatus.UNCONFIGURED


def test_unmapped_device_fails() -> None:
    adapter = WizAdapter(WizConfig({"other.bulb": TARGET}), FakeTransport())
    result = adapter.execute(_command("power", "set", True), _profile())
    assert result.status is AdapterStatus.FAILED
    assert "no WiZ device mapped" in result.detail


def test_read_state_returns_none_when_unconfigured() -> None:
    assert WizAdapter(None, FakeTransport()).read_state("room.bulb") is None


def test_read_state_returns_none_for_unmapped_device() -> None:
    adapter = WizAdapter(WizConfig({"other.bulb": TARGET}), FakeTransport())
    assert adapter.read_state("room.bulb") is None


def test_read_state_returns_none_on_transport_failure() -> None:
    adapter = _adapter(FakeTransport(fail=WizTimeout("no response")))
    assert adapter.read_state("room.bulb") is None


def test_read_state_returns_pilot_without_sending_a_command() -> None:
    transport = FakeTransport({"state": True, "dimming": 42, "temp": 3000})
    adapter = _adapter(transport)
    state = adapter.read_state("room.bulb")
    assert state == {"state": True, "dimming": 42, "temp": 3000}
    assert transport.sent == []  # 읽기만 했다 — setPilot은 한 번도 안 나갔다


def test_from_env_returns_none_without_targets() -> None:
    assert WizConfig.from_env({}) is None
    assert WizConfig.from_env({"WIZ_DEVICE_TARGETS": "{}"}) is None
    assert WizConfig.from_env({"WIZ_DEVICE_TARGETS": "not json"}) is None


def test_from_env_parses_targets_and_timeout() -> None:
    config = WizConfig.from_env(
        {"WIZ_DEVICE_TARGETS": '{"room.bulb": "10.0.0.5"}', "WIZ_TIMEOUT_S": "1.5"}
    )
    assert config is not None
    assert config.device_targets == {"room.bulb": "10.0.0.5"}
    assert config.timeout_s == 1.5


# --- power -----------------------------------------------------------------


def test_power_set_on_verifies() -> None:
    transport = FakeTransport({"state": False})
    result = _adapter(transport).execute(_command("power", "set", True), _profile())
    assert result.status is AdapterStatus.VERIFIED
    assert transport.sent == [{"state": True}]


def test_power_toggle_inverts_current_state() -> None:
    transport = FakeTransport({"state": True})
    result = _adapter(transport).execute(_command("power", "toggle", True), _profile())
    assert result.status is AdapterStatus.VERIFIED
    assert transport.sent == [{"state": False}]


def test_power_rejects_unsupported_operation() -> None:
    result = _adapter(FakeTransport()).execute(_command("power", "increment", 1), _profile())
    assert result.status is AdapterStatus.FAILED


# --- brightness / color temperature ----------------------------------------


def test_brightness_set_sends_dimming() -> None:
    transport = FakeTransport({"dimming": 20})
    result = _adapter(transport).execute(_command("brightness", "set", 70), _profile())
    assert result.status is AdapterStatus.VERIFIED
    assert transport.sent == [{"dimming": 70}]


def test_brightness_increment_adds_to_current() -> None:
    transport = FakeTransport({"dimming": 50})
    result = _adapter(transport).execute(_command("brightness", "increment", 10), _profile())
    assert result.status is AdapterStatus.VERIFIED
    assert transport.sent == [{"dimming": 60}]


def test_brightness_clamps_to_profile_maximum() -> None:
    transport = FakeTransport({"dimming": 95})
    _adapter(transport).execute(_command("brightness", "increment", 10), _profile())
    assert transport.sent == [{"dimming": 100}]


def test_brightness_clamps_to_profile_minimum_not_zero() -> None:
    # WiZ의 minDimLevel=10 — 0으로 내려가면 기기가 거부한다. 프로필 하한으로 막는다.
    transport = FakeTransport({"dimming": 15})
    _adapter(transport).execute(_command("brightness", "decrement", 10), _profile())
    assert transport.sent == [{"dimming": 10}]


def test_color_temperature_decrement_clamps_to_minimum() -> None:
    transport = FakeTransport({"temp": 2750})
    _adapter(transport).execute(_command("color_temperature", "decrement", 100), _profile())
    assert transport.sent == [{"temp": 2700}]


def test_unknown_capability_fails() -> None:
    result = _adapter(FakeTransport()).execute(_command("volume", "increment", 1), _profile())
    assert result.status is AdapterStatus.FAILED
    assert "does not handle capability" in result.detail


# --- honest status ---------------------------------------------------------


def test_timeout_is_failed_not_success() -> None:
    transport = FakeTransport(fail=WizTimeout("no response"))
    result = _adapter(transport).execute(_command("power", "set", True), _profile())
    assert result.status is AdapterStatus.FAILED
    assert "timed out" in result.detail


def test_network_error_is_failed() -> None:
    transport = FakeTransport(fail=OSError("network down"))
    result = _adapter(transport).execute(_command("power", "set", True), _profile())
    assert result.status is AdapterStatus.FAILED
    assert "network error" in result.detail


def test_state_mismatch_reports_unverified_never_verified() -> None:
    class StubbornTransport(FakeTransport):
        def send(self, target, payload, timeout_s):  # type: ignore[no-untyped-def]
            if payload.get("method") == "setPilot":
                self.sent.append(dict(payload.get("params") or {}))
                return {"result": {"success": True}}  # 상태를 실제로 바꾸지 않는다
            return {"result": dict(self.state)}

    transport = StubbornTransport({"dimming": 20})
    result = _adapter(transport).execute(_command("brightness", "set", 70), _profile())
    assert result.status is AdapterStatus.UNVERIFIED
    assert "!= expected" in result.detail


def test_device_rejecting_setpilot_is_failed() -> None:
    class RejectingTransport(FakeTransport):
        def send(self, target, payload, timeout_s):  # type: ignore[no-untyped-def]
            if payload.get("method") == "setPilot":
                return {"result": {"success": False}}
            return {"result": dict(self.state)}

    result = _adapter(RejectingTransport()).execute(_command("power", "set", True), _profile())
    assert result.status is AdapterStatus.FAILED
    assert "rejected" in result.detail


# --- addressing ------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("9877d5cffaf8", True),
        ("98:77:D5:CF:FA:F8", True),
        ("98-77-d5-cf-fa-f8", True),
        ("10.26.34.165", False),
        ("bulb.local", False),
    ],
)
def test_mac_detection(target: str, expected: bool) -> None:
    assert _is_mac(target) is expected


@pytest.mark.parametrize(
    ("target", "mac", "host"),
    [
        ("10.26.34.165", None, "10.26.34.165"),
        ("bulb.local", None, "bulb.local"),
        ("9877d5cffaf8", "9877d5cffaf8", None),
        ("98:77:D5:CF:FA:F8", "9877d5cffaf8", None),
        ("9877d5cffaf8@10.26.34.165", "9877d5cffaf8", "10.26.34.165"),
        ("98:77:D5:CF:FA:F8@10.0.0.9", "9877d5cffaf8", "10.0.0.9"),
    ],
)
def test_parse_target(target: str, mac: str | None, host: str | None) -> None:
    parsed = parse_target(target)
    assert parsed.mac == mac
    assert parsed.host == host


def test_resolution_prefers_known_ip_and_skips_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAC@IP는 평소에 탐색을 하지 않는다 — 브로드캐스트 차단 네트워크에서도 동작해야 한다."""
    calls: list[int] = []

    def _never(*_args: object, **_kwargs: object) -> dict[str, str]:
        calls.append(1)
        return {}

    monkeypatch.setattr("jarvis.runtime_protocol.adapters.wiz.discover", _never)
    transport = UdpWizTransport()
    host = transport._resolve(parse_target("9877d5cffaf8@10.26.34.165"), rediscover=False)
    assert host == "10.26.34.165"
    assert calls == []  # 탐색을 부르지 않았다


def test_resolution_falls_back_to_discovery_when_only_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "jarvis.runtime_protocol.adapters.wiz.discover",
        lambda *_a, **_k: {"9877d5cffaf8": "10.0.0.42"},
    )
    transport = UdpWizTransport()
    host = transport._resolve(parse_target("9877d5cffaf8"), rediscover=False)
    assert host == "10.0.0.42"


def test_resolution_raises_when_mac_undiscoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.runtime_protocol.adapters.wiz.discover", lambda *_a, **_k: {})
    transport = UdpWizTransport()
    with pytest.raises(WizTimeout, match="MAC@IP"):
        transport._resolve(parse_target("9877d5cffaf8"), rediscover=False)


# --- color(색상) ---------------------------------------------------------------
#
# 색상은 WiZ에 대응 파라미터가 없어(기기는 r/g/b를 받는다) 다른 수치 capability와
# 다른 경로를 탄다. 그리고 유일하게 **클램프가 아니라 순환**한다 — 클램프하면 회전
# 제스처가 양 끝 색에서 죽어 시연이 멈춘 것처럼 보인다.


def _state_at_hue(hue: float) -> dict[str, object]:
    red, green, blue = hue_to_rgb(hue)
    return {"state": True, "dimming": 50, "r": red, "g": green, "b": blue}


def test_hue_to_rgb_hits_the_primaries() -> None:
    assert hue_to_rgb(0) == (255, 0, 0)
    assert hue_to_rgb(120) == (0, 255, 0)
    assert hue_to_rgb(240) == (0, 0, 255)


def test_hue_to_rgb_wraps_at_a_full_turn() -> None:
    assert hue_to_rgb(360) == hue_to_rgb(0)
    assert hue_to_rgb(420) == hue_to_rgb(60)


def test_rgb_to_hue_round_trips() -> None:
    for hue in (0, 60, 120, 180, 240, 300):
        assert abs(rgb_to_hue(*hue_to_rgb(hue)) - hue) < 1.0


def test_rgb_to_hue_treats_greyscale_as_zero() -> None:
    """CCT 모드에서는 r/g/b가 없거나 같다 — 색상을 지어내지 않고 0도에서 시작한다."""
    assert rgb_to_hue(0, 0, 0) == 0.0
    assert rgb_to_hue(200, 200, 200) == 0.0


def test_color_increment_sends_rgb_not_temp() -> None:
    transport = FakeTransport(_state_at_hue(0))
    adapter = WizAdapter(WizConfig({"room.bulb": TARGET}), transport)
    result = adapter.execute(_command("color", "increment", 60), _profile())

    assert result.status is AdapterStatus.VERIFIED
    sent = transport.sent[0]
    assert set(sent) == {"r", "g", "b"}  # temp를 건드리지 않는다(모드 전환)
    assert sent == dict(zip("rgb", hue_to_rgb(60)))


def test_color_wraps_forward_past_a_full_turn() -> None:
    """330도에서 +60도는 30도다 — 360도 벽에 부딪혀 멈추지 않는다."""
    transport = FakeTransport(_state_at_hue(330))
    adapter = WizAdapter(WizConfig({"room.bulb": TARGET}), transport)
    result = adapter.execute(_command("color", "increment", 60), _profile())

    assert result.status is AdapterStatus.VERIFIED
    assert abs(rgb_to_hue(*(transport.sent[0][k] for k in "rgb")) - 30) < 1.0


def test_color_wraps_backward_below_zero() -> None:
    transport = FakeTransport(_state_at_hue(30))
    adapter = WizAdapter(WizConfig({"room.bulb": TARGET}), transport)
    result = adapter.execute(_command("color", "decrement", 60), _profile())

    assert result.status is AdapterStatus.VERIFIED
    assert abs(rgb_to_hue(*(transport.sent[0][k] for k in "rgb")) - 330) < 1.0


def test_color_starts_from_red_when_bulb_is_in_cct_mode() -> None:
    """r/g/b가 아예 없는 상태(CCT 모드)에서도 실패하지 않고 0도 기준으로 움직인다."""
    transport = FakeTransport({"state": True, "dimming": 50, "temp": 4000})
    adapter = WizAdapter(WizConfig({"room.bulb": TARGET}), transport)
    result = adapter.execute(_command("color", "increment", 60), _profile())

    assert result.status is AdapterStatus.VERIFIED
    assert transport.sent[0] == dict(zip("rgb", hue_to_rgb(60)))


def test_color_reports_unverified_when_device_disagrees() -> None:
    """되읽은 색이 보낸 색과 다르면 성공을 지어내지 않는다."""
    transport = FakeTransport(_state_at_hue(0))

    def _send(target: str, payload: Mapping[str, object], timeout_s: float) -> Mapping[str, object]:
        method = payload.get("method")
        if method == "setPilot":
            transport.sent.append(dict(payload.get("params") or {}))  # type: ignore[arg-type]
            return {"result": {"success": True}}
        return {"result": _state_at_hue(200)}  # 엉뚱한 색을 보고한다

    transport.send = _send  # type: ignore[assignment]
    adapter = WizAdapter(WizConfig({"room.bulb": TARGET}), transport)
    result = adapter.execute(_command("color", "increment", 60), _profile())

    assert result.status is AdapterStatus.UNVERIFIED
    assert "device reports" in result.detail


def test_color_rejects_non_numeric_value() -> None:
    transport = FakeTransport(_state_at_hue(0))
    adapter = WizAdapter(WizConfig({"room.bulb": TARGET}), transport)
    result = adapter.execute(_command("color", "increment", True), _profile())

    assert result.status is AdapterStatus.FAILED
    assert transport.sent == []  # 아무것도 보내지 않았다


def test_color_rejects_unsupported_operation() -> None:
    transport = FakeTransport(_state_at_hue(0))
    adapter = WizAdapter(WizConfig({"room.bulb": TARGET}), transport)
    result = adapter.execute(_command("color", "toggle", 60), _profile())

    assert result.status is AdapterStatus.FAILED
    assert transport.sent == []

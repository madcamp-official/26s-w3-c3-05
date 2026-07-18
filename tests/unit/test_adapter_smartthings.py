"""Unit tests for the SmartThings adapter (fake HTTP transport, no network)."""

from __future__ import annotations

import json

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.adapters.base import AdapterStatus
from jarvis.runtime_protocol.adapters.http import (
    HttpRequest,
    HttpResponse,
    TransportNetworkError,
    TransportTimeout,
)
from jarvis.runtime_protocol.adapters.smartthings import (
    SmartThingsAdapter,
    SmartThingsConfig,
)
from jarvis.runtime_protocol.protocol.capability import (
    BooleanCapability,
    DeviceProfile,
    NumberCapability,
)

_UUID = "st-uuid-1"

_PROFILE = DeviceProfile(
    device_id="room.bulb",
    adapter="smartthings",
    capabilities={
        "power": BooleanCapability(),
        "brightness": NumberCapability(minimum=0, maximum=100, step=10),
        "color_temperature": NumberCapability(minimum=2700, maximum=6500, step=100),
    },
)


def _config() -> SmartThingsConfig:
    return SmartThingsConfig(token="secret-token", device_targets={"room.bulb": _UUID})


def _command(capability: str, operation: str, value: int | float | bool) -> Command:
    return Command(
        command_id="cmd-1",
        intent_id="intent-1",
        device_id="room.bulb",
        capability=capability,
        operation=operation,
        value=value,
        expires_at_ms=10_000,
    )


class FakeTransport:
    """Scripted transport: matches (method, path-suffix) to canned responses.

    A status body is a plain dict describing the main-component status; command
    POSTs default to HTTP 200. ``raise_with`` forces a transport exception.
    """

    def __init__(self) -> None:
        self.requests: list[HttpRequest] = []
        self.status_body: dict[str, object] = {}
        self.command_status = 200
        self.raise_with: Exception | None = None

    def send(self, request: HttpRequest, timeout_s: float) -> HttpResponse:
        self.requests.append(request)
        if self.raise_with is not None:
            raise self.raise_with
        if request.method == "GET":
            return HttpResponse(200, json.dumps(self.status_body).encode())
        return HttpResponse(self.command_status, b"{}")

    def commands_sent(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for req in self.requests:
            if req.method == "POST" and req.body is not None:
                out.extend(json.loads(req.body)["commands"])
        return out


def test_unconfigured_without_token_touches_no_network() -> None:
    transport = FakeTransport()
    adapter = SmartThingsAdapter(None, transport)
    result = adapter.execute(_command("power", "set", True), _PROFILE)
    assert result.status == AdapterStatus.UNCONFIGURED
    assert transport.requests == []


def test_unmapped_device_fails() -> None:
    transport = FakeTransport()
    config = SmartThingsConfig(token="t", device_targets={})  # no mapping
    result = SmartThingsAdapter(config, transport).execute(
        _command("power", "set", True), _PROFILE
    )
    assert result.status == AdapterStatus.FAILED
    assert transport.requests == []


def test_power_set_on_sends_switch_on_and_verifies() -> None:
    transport = FakeTransport()
    transport.status_body = {"switch": {"switch": {"value": "on"}}}
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("power", "set", True), _PROFILE
    )
    assert result.status == AdapterStatus.VERIFIED
    commands = transport.commands_sent()
    assert commands[0]["capability"] == "switch"
    assert commands[0]["command"] == "on"


def test_power_toggle_reads_state_then_sends_opposite() -> None:
    transport = FakeTransport()
    transport.status_body = {"switch": {"switch": {"value": "on"}}}
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("power", "toggle", True), _PROFILE
    )
    # currently on → toggle sends off; readback still says on → UNVERIFIED
    assert transport.commands_sent()[0]["command"] == "off"
    assert result.status == AdapterStatus.UNVERIFIED


def test_brightness_set_sends_setlevel() -> None:
    transport = FakeTransport()
    transport.status_body = {"switchLevel": {"level": {"value": 40}}}
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "set", 40), _PROFILE
    )
    command = transport.commands_sent()[0]
    assert command["command"] == "setLevel"
    assert command["arguments"] == [40]
    assert result.status == AdapterStatus.VERIFIED


def test_brightness_decrement_reads_clamps_and_sets_absolute() -> None:
    transport = FakeTransport()
    # current 5 → decrement 10 → clamp to floor 0
    transport.status_body = {"switchLevel": {"level": {"value": 5}}}
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "decrement", 10), _PROFILE
    )
    command = transport.commands_sent()[0]
    assert command["command"] == "setLevel"
    assert command["arguments"] == [0]  # clamped to minimum, not -5
    assert result.status == AdapterStatus.UNVERIFIED  # readback still 5 != 0


def test_brightness_increment_clamps_to_maximum() -> None:
    transport = FakeTransport()
    transport.status_body = {"switchLevel": {"level": {"value": 95}}}
    SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "increment", 20), _PROFILE
    )
    assert transport.commands_sent()[0]["arguments"] == [100]  # clamped to max


def test_auth_error_classified() -> None:
    transport = FakeTransport()
    transport.command_status = 401
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "set", 40), _PROFILE
    )
    assert result.status == AdapterStatus.FAILED
    assert "authentication" in result.detail


def test_rate_limit_classified() -> None:
    transport = FakeTransport()
    transport.command_status = 429
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "set", 40), _PROFILE
    )
    assert result.status == AdapterStatus.FAILED
    assert "rate limited" in result.detail


def test_timeout_classified() -> None:
    transport = FakeTransport()
    transport.raise_with = TransportTimeout("slow")
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "set", 40), _PROFILE
    )
    assert result.status == AdapterStatus.FAILED
    assert "timed out" in result.detail


def test_network_error_classified() -> None:
    transport = FakeTransport()
    transport.raise_with = TransportNetworkError("no route")
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "set", 40), _PROFILE
    )
    assert result.status == AdapterStatus.FAILED
    assert "network error" in result.detail


def test_token_never_appears_in_result_detail() -> None:
    transport = FakeTransport()
    transport.command_status = 500
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "set", 40), _PROFILE
    )
    assert "secret-token" not in result.detail


def test_verify_unreadable_state_is_unverified() -> None:
    transport = FakeTransport()
    transport.status_body = {}  # command 200, but no switchLevel in status
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("brightness", "set", 40), _PROFILE
    )
    assert result.status == AdapterStatus.UNVERIFIED


def test_unsupported_capability_fails() -> None:
    transport = FakeTransport()
    result = SmartThingsAdapter(_config(), transport).execute(
        _command("scroll", "increment", 1), _PROFILE
    )
    assert result.status == AdapterStatus.FAILED
    assert transport.requests == []


def test_from_env_returns_none_without_token() -> None:
    assert SmartThingsConfig.from_env({}) is None


def test_from_env_parses_token_and_targets() -> None:
    config = SmartThingsConfig.from_env(
        {
            "SMARTTHINGS_TOKEN": "abc",
            "SMARTTHINGS_DEVICE_TARGETS": '{"room.bulb": "uuid-9"}',
        }
    )
    assert config is not None
    assert config.token == "abc"
    assert config.device_targets == {"room.bulb": "uuid-9"}

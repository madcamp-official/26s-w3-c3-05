"""시작 전구 프로브 — 실패 사유를 구분해 돌려주는지 검증(네트워크 없이).

"설정이 없다"와 "설정은 있는데 안 닿는다"는 사용자가 할 일이 전혀 다르므로
(전자는 .env, 후자는 전원·네트워크) 두 경우를 뭉뚱그리면 안 된다.
"""

from __future__ import annotations

from collections.abc import Mapping

from jarvis.monitoring.bulb_probe import probe_bulb
from jarvis.runtime_protocol.adapters.wiz import WizConfig, WizTimeout

TARGETS = {"room.bulb": "9877d5cffaf8@10.26.34.165"}


class FakeTransport:
    """요청을 기록하고 미리 정한 응답(또는 예외)을 돌려준다."""

    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[tuple[str, Mapping[str, object], float]] = []

    def send(
        self, target: str, payload: Mapping[str, object], timeout_s: float
    ) -> Mapping[str, object]:
        self.calls.append((target, payload, timeout_s))
        if self._error is not None:
            raise self._error
        assert isinstance(self._response, dict)
        return self._response


def _ok_response(state: bool = True, dimming: int = 40) -> dict[str, object]:
    return {"method": "getPilot", "result": {"mac": "9877d5cffaf8", "state": state, "dimming": dimming}}


def test_reports_success_with_current_state() -> None:
    transport = FakeTransport(_ok_response(state=True, dimming=40))
    result = probe_bulb(WizConfig(TARGETS), "room.bulb", transport)
    assert result.ok
    assert "연결됨" in result.detail
    assert "켜짐" in result.detail and "40" in result.detail


def test_probe_sends_getpilot_to_configured_target() -> None:
    transport = FakeTransport(_ok_response())
    probe_bulb(WizConfig(TARGETS), "room.bulb", transport, timeout_s=1.5)
    assert len(transport.calls) == 1
    target, payload, timeout_s = transport.calls[0]
    assert target == TARGETS["room.bulb"]
    assert payload["method"] == "getPilot"  # 상태를 바꾸지 않는 읽기 전용 호출
    assert timeout_s == 1.5


def test_unconfigured_is_distinct_from_unreachable() -> None:
    result = probe_bulb(None, "room.bulb", FakeTransport(_ok_response()))
    assert not result.ok
    assert "WIZ_DEVICE_TARGETS" in result.detail


def test_device_not_mapped_names_the_device() -> None:
    result = probe_bulb(WizConfig({"other.bulb": "10.0.0.1"}), "room.bulb", FakeTransport())
    assert not result.ok
    assert "room.bulb" in result.detail


def test_timeout_is_reported_as_unreachable_with_address() -> None:
    transport = FakeTransport(error=WizTimeout("no response"))
    result = probe_bulb(WizConfig(TARGETS), "room.bulb", transport)
    assert not result.ok
    assert "연결 실패" in result.detail
    assert "10.26.34.165" in result.detail  # 어디로 시도했는지 보여준다


def test_malformed_response_is_not_treated_as_success() -> None:
    result = probe_bulb(WizConfig(TARGETS), "room.bulb", FakeTransport({"no_result": 1}))
    assert not result.ok


def test_probe_never_raises_on_transport_error() -> None:
    """어떤 예외도 프로브 밖으로 나가면 안 된다 — 시작이 실패로 끝나면 안 되므로."""
    result = probe_bulb(WizConfig(TARGETS), "room.bulb", FakeTransport(error=OSError("boom")))
    assert not result.ok
    assert "OSError" in result.detail


def test_probe_returns_the_state_it_read() -> None:
    """프로브는 도달 여부만이 아니라 **읽은 상태**를 돌려준다 — 화면이 그 값에서 시작한다."""
    response = {"result": {"state": True, "dimming": 40, "temp": 4200}}
    result = probe_bulb(WizConfig(TARGETS), "room.bulb", FakeTransport(response))
    assert result.state is not None
    assert result.state.brightness == 40
    assert result.state.color_temperature == 4200


def test_failed_probe_carries_no_state() -> None:
    """안 닿았으면 상태도 없다 — 화면은 '확인 전'을 유지해야 한다."""
    result = probe_bulb(WizConfig(TARGETS), "room.bulb", FakeTransport(error=WizTimeout("no")))
    assert result.state is None

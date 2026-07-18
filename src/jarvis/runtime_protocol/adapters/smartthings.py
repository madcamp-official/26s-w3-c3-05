"""SmartThings cloud adapter for the demo bulb.

Translates a validated :class:`Command` into SmartThings device commands over the
injected :class:`HttpTransport`, then reads device state back to verify the
result. Design notes:

- **Honest status** (development-principles 1.1): a command is ``VERIFIED`` only
  when the device state is read back and matches; if it was sent but state can't
  be confirmed the result is ``UNVERIFIED``, never a faked success.
- **Relative-op clamp** (decisions.md): increment/decrement need the device's
  current level, so the adapter reads state, applies the delta, and clamps to the
  capability's ``[min,max]`` before sending an absolute ``setLevel``.
- **Config / secrets** (6.1/6.3): the token and device-UUID map come from the
  environment. With no token the adapter returns ``UNCONFIGURED`` and touches no
  network, so a Windows-only run is not blocked.
- **Error classification** (6.4): timeout, network error, auth (401/403), rate
  limit (429), and other HTTP errors are reported distinctly. The token is never
  put in a result detail or log (6.5).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.adapters.base import AdapterResult, AdapterStatus
from jarvis.runtime_protocol.adapters.http import (
    HttpRequest,
    HttpResponse,
    HttpTransport,
    TransportTimeout,
)
from jarvis.runtime_protocol.protocol.capability import DeviceProfile, NumberCapability

_DEFAULT_BASE_URL = "https://api.smartthings.com/v1"
_DEFAULT_TIMEOUT_S = 5.0

# internal capability -> (SmartThings capability, status attribute, set command)
_NUMBER_CAPS: dict[str, tuple[str, str, str]] = {
    "brightness": ("switchLevel", "level", "setLevel"),
    "color_temperature": ("colorTemperature", "colorTemperature", "setColorTemperature"),
}
_SWITCH_CAP = ("switch", "switch")


@dataclass(frozen=True, slots=True)
class SmartThingsConfig:
    token: str
    device_targets: Mapping[str, str]
    base_url: str = _DEFAULT_BASE_URL
    timeout_s: float = _DEFAULT_TIMEOUT_S

    @staticmethod
    def from_env(env: Mapping[str, str]) -> SmartThingsConfig | None:
        """Build config from an env mapping, or ``None`` if no token is present."""
        token = env.get("SMARTTHINGS_TOKEN", "").strip()
        if not token:
            return None
        raw_targets = env.get("SMARTTHINGS_DEVICE_TARGETS", "").strip() or "{}"
        try:
            parsed = json.loads(raw_targets)
            targets = {str(k): str(v) for k, v in dict(parsed).items()}
        except (ValueError, TypeError):
            targets = {}
        base_url = env.get("SMARTTHINGS_BASE_URL", "").strip() or _DEFAULT_BASE_URL
        try:
            timeout_s = float(env.get("SMARTTHINGS_TIMEOUT_S", "") or _DEFAULT_TIMEOUT_S)
        except ValueError:
            timeout_s = _DEFAULT_TIMEOUT_S
        return SmartThingsConfig(token, targets, base_url.rstrip("/"), timeout_s)


class _Failure(Exception):
    """Internal: carries an AdapterResult to unwind out of a helper."""

    def __init__(self, result: AdapterResult) -> None:
        super().__init__(result.detail)
        self.result = result


class SmartThingsAdapter:
    """Executes bulb commands against the SmartThings cloud API."""

    name = "smartthings"

    def __init__(self, config: SmartThingsConfig | None, transport: HttpTransport) -> None:
        self._config = config
        self._transport = transport

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult:
        if self._config is None:
            return AdapterResult(
                AdapterStatus.UNCONFIGURED, "SMARTTHINGS_TOKEN is not set"
            )
        uuid = self._config.device_targets.get(command.device_id)
        if not uuid:
            return AdapterResult(
                AdapterStatus.FAILED,
                f"no SmartThings device mapped for {command.device_id!r}",
            )
        try:
            if command.capability == "power":
                return self._apply_power(uuid, command)
            if command.capability in _NUMBER_CAPS:
                return self._apply_number(uuid, command, profile)
            return AdapterResult(
                AdapterStatus.FAILED,
                f"smartthings adapter does not handle capability {command.capability!r}",
            )
        except _Failure as failure:
            return failure.result

    # -- power ---------------------------------------------------------------

    def _apply_power(self, uuid: str, command: Command) -> AdapterResult:
        if command.operation == "set":
            target_on = bool(command.value)
        elif command.operation == "toggle":
            current = self._read_value(uuid, *_SWITCH_CAP)
            target_on = str(current) != "on"
        else:
            return AdapterResult(
                AdapterStatus.FAILED, f"power does not support {command.operation!r}"
            )
        st_command = "on" if target_on else "off"
        self._send_command(uuid, "switch", st_command, [])
        return self._verify(uuid, *_SWITCH_CAP, expected="on" if target_on else "off")

    # -- numeric (brightness, color_temperature) -----------------------------

    def _apply_number(
        self, uuid: str, command: Command, profile: DeviceProfile
    ) -> AdapterResult:
        spec = profile.capabilities.get(command.capability)
        if not isinstance(spec, NumberCapability):
            return AdapterResult(
                AdapterStatus.FAILED,
                f"capability {command.capability!r} is not numeric in the device profile",
            )
        st_capability, st_attribute, set_command = _NUMBER_CAPS[command.capability]
        delta = float(command.value)

        if command.operation == "set":
            target = delta
        elif command.operation in ("increment", "decrement"):
            current = self._read_number(uuid, st_capability, st_attribute)
            signed = delta if command.operation == "increment" else -delta
            target = min(spec.maximum, max(spec.minimum, current + signed))
        else:
            return AdapterResult(
                AdapterStatus.FAILED,
                f"{command.capability} does not support {command.operation!r}",
            )

        target_int = int(round(target))
        self._send_command(uuid, st_capability, set_command, [target_int])
        return self._verify(
            uuid, st_capability, st_attribute, expected=target_int, tolerance=1.0
        )

    # -- HTTP helpers --------------------------------------------------------

    def _send_command(
        self, uuid: str, st_capability: str, st_command: str, arguments: list[object]
    ) -> None:
        body = json.dumps(
            {
                "commands": [
                    {
                        "component": "main",
                        "capability": st_capability,
                        "command": st_command,
                        "arguments": arguments,
                    }
                ]
            }
        ).encode("utf-8")
        response = self._request("POST", f"/devices/{uuid}/commands", body)
        self._raise_for_status(response)

    def _read_value(self, uuid: str, st_capability: str, st_attribute: str) -> object:
        status = self._read_main_status(uuid)
        value = _extract(status, st_capability, st_attribute)
        if value is None:
            raise _Failure(
                AdapterResult(
                    AdapterStatus.FAILED,
                    f"could not read {st_capability}.{st_attribute} from device",
                )
            )
        return value

    def _read_number(self, uuid: str, st_capability: str, st_attribute: str) -> float:
        value = self._read_value(uuid, st_capability, st_attribute)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise _Failure(
                AdapterResult(
                    AdapterStatus.FAILED,
                    f"{st_capability}.{st_attribute} is not numeric: {value!r}",
                )
            )
        try:
            return float(value)
        except ValueError as exc:
            raise _Failure(
                AdapterResult(
                    AdapterStatus.FAILED,
                    f"{st_capability}.{st_attribute} is not numeric: {value!r}",
                )
            ) from exc

    def _read_main_status(self, uuid: str) -> dict[str, object]:
        response = self._request("GET", f"/devices/{uuid}/components/main/status", None)
        self._raise_for_status(response)
        try:
            parsed = json.loads(response.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise _Failure(
                AdapterResult(AdapterStatus.FAILED, f"invalid status JSON: {exc}")
            ) from exc
        return dict(parsed) if isinstance(parsed, dict) else {}

    def _verify(
        self,
        uuid: str,
        st_capability: str,
        st_attribute: str,
        *,
        expected: object,
        tolerance: float = 0.0,
    ) -> AdapterResult:
        # The command already went out; if we can't confirm state, report
        # UNVERIFIED (sent but unconfirmed) rather than FAILED.
        try:
            status = self._read_main_status(uuid)
        except _Failure as failure:
            return AdapterResult(
                AdapterStatus.UNVERIFIED, f"sent, readback failed: {failure.result.detail}"
            )
        reported = _extract(status, st_capability, st_attribute)
        if reported is None:
            return AdapterResult(AdapterStatus.UNVERIFIED, "sent, state not readable")
        if _matches(reported, expected, tolerance):
            return AdapterResult(AdapterStatus.VERIFIED, f"state confirmed: {reported}")
        return AdapterResult(
            AdapterStatus.UNVERIFIED, f"sent, state {reported} != expected {expected}"
        )

    def _request(self, method: str, path: str, body: bytes | None) -> HttpResponse:
        assert self._config is not None  # execute() guards this
        request = HttpRequest(
            method=method,
            url=f"{self._config.base_url}{path}",
            headers={
                "Authorization": f"Bearer {self._config.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            body=body,
        )
        try:
            return self._transport.send(request, self._config.timeout_s)
        except TransportTimeout as exc:
            raise _Failure(AdapterResult(AdapterStatus.FAILED, f"request timed out: {exc}")) from exc
        except Exception as exc:  # noqa: BLE001 - any transport failure is honest FAILED
            raise _Failure(
                AdapterResult(AdapterStatus.FAILED, f"network error: {type(exc).__name__}")
            ) from exc

    @staticmethod
    def _raise_for_status(response: HttpResponse) -> None:
        if 200 <= response.status < 300:
            return
        if response.status in (401, 403):
            detail = f"authentication failed (HTTP {response.status})"
        elif response.status == 429:
            detail = "rate limited (HTTP 429)"
        else:
            detail = f"unexpected HTTP {response.status}"
        raise _Failure(AdapterResult(AdapterStatus.FAILED, detail))


def _extract(status: dict[str, object], st_capability: str, st_attribute: str) -> object | None:
    """Pull ``status[cap][attr]["value"]`` defensively, or ``None`` if absent."""
    capability = status.get(st_capability)
    if not isinstance(capability, dict):
        return None
    attribute = capability.get(st_attribute)
    if not isinstance(attribute, dict):
        return None
    return attribute.get("value")


def _matches(reported: object, expected: object, tolerance: float) -> bool:
    if isinstance(expected, str):
        return str(reported) == expected
    try:
        return abs(float(reported) - float(expected)) <= tolerance  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False

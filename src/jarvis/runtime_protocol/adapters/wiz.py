"""Philips WiZ 전구 adapter — 클라우드 없이 **로컬 UDP**로 직접 제어한다.

SmartThings adapter와 같은 계약(`DeviceAdapter`)을 구현하지만 경로가 다르다: WiZ 전구는
같은 LAN에서 UDP 38899 포트로 JSON-RPC(`setPilot`/`getPilot`)를 받는다. 토큰도 클라우드
계정도 필요 없고 왕복이 LAN 한 홉이라(실측 15~47ms) 제스처 제어에 유리하다.

설계 노트:

- **정직한 상태**(development-principles 1.1): `setPilot`을 보낸 뒤 `getPilot`으로 상태를
  되읽어 일치할 때만 ``VERIFIED``. 보냈지만 확인 못 하면 ``UNVERIFIED``이며 성공을
  지어내지 않는다.
- **상대 연산 클램프**(decisions.md): increment/decrement는 기기의 현재 값이 필요하므로
  현재 상태를 읽어 delta를 적용하고 capability의 ``[min,max]``로 클램프한 절대값을 보낸다.
- **설정**(6.1/6.3): 기기 대상은 환경변수에서 온다. 설정이 없으면 ``UNCONFIGURED``를
  반환하고 네트워크를 건드리지 않는다.
- **주소 지정**: 대상은 ``IP``, ``MAC``, 또는 ``MAC@IP`` 세 형식을 받는다. 실측 결과 일부
  네트워크(AP 클라이언트 격리 등)는 브로드캐스트를 흘려주지 않아 MAC 단독 탐색이 실패한다.
  그래서 ``MAC@IP``를 권장한다 — 평소엔 IP로 바로 쏘고(탐색 비용 0), 그 IP가 응답하지
  않을 때만 MAC으로 재탐색해 DHCP 주소 변경을 흡수한다.
"""

from __future__ import annotations

import colorsys
import json
import re
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.adapters.base import AdapterResult, AdapterStatus
from jarvis.runtime_protocol.protocol.capability import DeviceProfile, NumberCapability

WIZ_PORT = 38899
_DEFAULT_TIMEOUT_S = 3.0
_DEFAULT_DISCOVERY_TIMEOUT_S = 4.0

# 내부 capability -> setPilot/getPilot 파라미터 이름.
# dimming: 10~100 (모델의 minDimLevel=10), temp: 켈빈(모델 cctRange 2700~6500).
_NUMBER_PARAMS: dict[str, str] = {
    "brightness": "dimming",
    "color_temperature": "temp",
}
_STATE_PARAM = "state"

# 색상(hue)은 WiZ에 대응하는 스칼라 파라미터가 없다 — 기기는 r/g/b 세 값을 받는다.
# 그래서 이 capability만 다른 경로를 탄다: 각도를 완전채도 RGB로 바꿔 보내고, 되읽을
# 때는 RGB를 다시 각도로 환산해 비교한다.
_COLOR_CAPABILITY = "color"
_RGB_PARAMS = ("r", "g", "b")
_HUE_DEGREES = 360

# 색상은 순환량이라 [min,max] 클램프가 의미가 없다. 다른 수치 capability와 달리
# 360도에서 0도로 **감아 돈다** — 시연에서 회전 제스처를 계속하면 색이 한 바퀴 돈다
# (클램프하면 빨강이나 보라에서 멈춰 제스처가 죽은 것처럼 보인다).
#
# 되읽기 허용 오차. hue→RGB→hue는 8bit 양자화를 거쳐 정확히 돌아오지 않고, 기기가
# 자체 보정을 하기도 한다. 실측 왕복 오차를 덮되 인접 색상 단계(60도)와는 확실히
# 구분되는 값으로 둔다.
_HUE_TOLERANCE_DEG = 15.0


def hue_to_rgb(hue_deg: float) -> tuple[int, int, int]:
    """색상각(도) → 완전채도·최대명도 RGB. 밝기는 dimming이 따로 관장한다."""
    red, green, blue = colorsys.hsv_to_rgb((hue_deg % _HUE_DEGREES) / _HUE_DEGREES, 1.0, 1.0)
    return round(red * 255), round(green * 255), round(blue * 255)


def rgb_to_hue(red: float, green: float, blue: float) -> float:
    """RGB → 색상각(도). 무채색(r=g=b, 흰색·꺼짐·CCT 모드)은 0도로 본다.

    전구가 CCT 모드에 있으면 getPilot이 r/g/b를 아예 안 주거나 0으로 준다. 그때는
    "현재 색상 없음"이므로 0도(빨강)에서 시작한다 — 값을 지어내지 않는다.
    """
    if red == green == blue:
        return 0.0
    hue, _, _ = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
    return hue * _HUE_DEGREES


def _hue_distance_deg(left: float, right: float) -> float:
    """두 색상각 사이의 최단 원형 거리 — 359도와 1도는 2도 차이다."""
    delta = abs(left - right) % _HUE_DEGREES
    return min(delta, _HUE_DEGREES - delta)

_MAC_RE = re.compile(r"^[0-9a-fA-F]{12}$")


def _is_mac(target: str) -> bool:
    """대상 문자열이 MAC(구분자 무시 12자리 hex)인지. 아니면 IP/호스트로 본다."""
    return bool(_MAC_RE.match(target.replace(":", "").replace("-", "")))


def _normalize_mac(target: str) -> str:
    return target.replace(":", "").replace("-", "").lower()


@dataclass(frozen=True, slots=True)
class _Target:
    """설정 문자열을 푼 주소. 둘 중 최소 하나는 있다."""

    mac: str | None
    host: str | None


def parse_target(target: str) -> _Target:
    """``IP`` / ``MAC`` / ``MAC@IP``를 (mac, host)로 푼다.

    ``MAC@IP``는 "이 MAC의 기기이고, 마지막으로 알려진 주소는 이 IP"라는 뜻이다 —
    IP로 먼저 시도하고 실패하면 MAC으로 재탐색할 수 있게 둘 다 보관한다.
    """
    head, _, tail = target.partition("@")
    if tail:
        return _Target(_normalize_mac(head) if _is_mac(head) else None, tail or None)
    if _is_mac(head):
        return _Target(_normalize_mac(head), None)
    return _Target(None, head or None)


class WizTimeout(TimeoutError):
    """전구가 제한 시간 안에 응답하지 않았다."""


class WizTransport(Protocol):
    """WiZ 기기와의 요청/응답 경계. 테스트는 fake를 주입해 네트워크 없이 검증한다."""

    def send(
        self, target: str, payload: Mapping[str, object], timeout_s: float
    ) -> Mapping[str, object]:
        """``target``(IP 또는 MAC)에 JSON-RPC를 보내고 파싱된 응답을 돌려준다."""
        ...


@dataclass(frozen=True, slots=True)
class WizConfig:
    """WiZ 대상 설정. ``device_targets``는 내부 device_id → IP 또는 MAC."""

    device_targets: Mapping[str, str]
    timeout_s: float = _DEFAULT_TIMEOUT_S

    @staticmethod
    def from_env(env: Mapping[str, str]) -> WizConfig | None:
        """환경변수로 설정을 만든다. 대상이 하나도 없으면 ``None``(=미설정)."""
        raw_targets = env.get("WIZ_DEVICE_TARGETS", "").strip() or "{}"
        try:
            parsed = json.loads(raw_targets)
            targets = {str(k): str(v) for k, v in dict(parsed).items()}
        except (ValueError, TypeError):
            targets = {}
        if not targets:
            return None
        try:
            timeout_s = float(env.get("WIZ_TIMEOUT_S", "") or _DEFAULT_TIMEOUT_S)
        except ValueError:
            timeout_s = _DEFAULT_TIMEOUT_S
        return WizConfig(targets, timeout_s)


class _Failure(Exception):
    """내부용: 헬퍼에서 AdapterResult를 들고 빠져나온다."""

    def __init__(self, result: AdapterResult) -> None:
        super().__init__(result.detail)
        self.result = result


class WizAdapter:
    """WiZ 전구 명령을 로컬 UDP로 실행한다."""

    name = "wiz"

    def __init__(self, config: WizConfig | None, transport: WizTransport) -> None:
        self._config = config
        self._transport = transport

    def execute(self, command: Command, profile: DeviceProfile) -> AdapterResult:
        if self._config is None:
            return AdapterResult(
                AdapterStatus.UNCONFIGURED, "WIZ_DEVICE_TARGETS is not set"
            )
        target = self._config.device_targets.get(command.device_id)
        if not target:
            return AdapterResult(
                AdapterStatus.FAILED,
                f"no WiZ device mapped for {command.device_id!r}",
            )
        try:
            if command.capability == "power":
                return self._apply_power(target, command)
            if command.capability == _COLOR_CAPABILITY:
                return self._apply_color(target, command)
            if command.capability in _NUMBER_PARAMS:
                return self._apply_number(target, command, profile)
            return AdapterResult(
                AdapterStatus.FAILED,
                f"wiz adapter does not handle capability {command.capability!r}",
            )
        except _Failure as failure:
            return failure.result

    def read_state(self, device_id: str) -> Mapping[str, object] | None:
        """실물 전구의 현재 ``getPilot`` 상태 조회. 명령을 보내지 않는 순수 읽기다.

        `execute()`와 달리 검증된 `Command`가 필요 없다 — 화면이 "실물이 지금 무슨
        색인가"를 직접 물어볼 때 쓴다. 설정 없음·대상 미매핑·통신 실패는 전부
        ``None``(모른다)이며, 마지막으로 보낸 명령을 대신 지어내 보이지 않는다.
        """
        if self._config is None:
            return None
        target = self._config.device_targets.get(device_id)
        if not target:
            return None
        try:
            return self._read_state(target)
        except _Failure:
            return None

    # -- power ---------------------------------------------------------------

    def _apply_power(self, target: str, command: Command) -> AdapterResult:
        if command.operation == "set":
            target_on = bool(command.value)
        elif command.operation == "toggle":
            current = self._read_state(target).get(_STATE_PARAM)
            target_on = not bool(current)
        else:
            return AdapterResult(
                AdapterStatus.FAILED, f"power does not support {command.operation!r}"
            )
        self._set_pilot(target, {_STATE_PARAM: target_on})
        return self._verify(target, _STATE_PARAM, expected=target_on)

    # -- numeric (brightness, color_temperature) -----------------------------

    def _apply_number(
        self, target: str, command: Command, profile: DeviceProfile
    ) -> AdapterResult:
        spec = profile.capabilities.get(command.capability)
        if not isinstance(spec, NumberCapability):
            return AdapterResult(
                AdapterStatus.FAILED,
                f"capability {command.capability!r} is not numeric in the device profile",
            )
        param = _NUMBER_PARAMS[command.capability]
        delta = float(command.value)

        if command.operation == "set":
            value = delta
        elif command.operation in ("increment", "decrement"):
            current = self._read_number(target, param)
            signed = delta if command.operation == "increment" else -delta
            value = min(spec.maximum, max(spec.minimum, current + signed))
        else:
            return AdapterResult(
                AdapterStatus.FAILED,
                f"{command.capability} does not support {command.operation!r}",
            )

        value_int = int(round(value))
        self._set_pilot(target, {param: value_int})
        return self._verify(target, param, expected=value_int, tolerance=1.0)

    # -- color (hue) ---------------------------------------------------------

    def _apply_color(self, target: str, command: Command) -> AdapterResult:
        """색상각을 RGB로 바꿔 보낸다. 다른 수치와 달리 클램프가 아니라 **순환**한다.

        WiZ에는 hue 파라미터가 없어 r/g/b 세 값을 함께 보낸다. 그래서
        `_apply_number`의 단일 파라미터 경로를 쓸 수 없고, 현재값 읽기·되읽기 검증도
        RGB↔각도 환산을 거친다.
        """
        if isinstance(command.value, bool) or not isinstance(command.value, (int, float)):
            return AdapterResult(
                AdapterStatus.FAILED, f"color requires a numeric value, got {command.value!r}"
            )
        delta = float(command.value)

        if command.operation == "set":
            hue = delta
        elif command.operation in ("increment", "decrement"):
            current = self._read_hue(target)
            hue = current + (delta if command.operation == "increment" else -delta)
        else:
            return AdapterResult(
                AdapterStatus.FAILED, f"color does not support {command.operation!r}"
            )

        hue %= _HUE_DEGREES  # 순환: 360도를 넘으면 0도로 돌아온다
        red, green, blue = hue_to_rgb(hue)
        self._set_pilot(target, {"r": red, "g": green, "b": blue})
        return self._verify_color(target, expected_hue=hue)

    def _read_hue(self, target: str) -> float:
        state = self._read_state(target)
        return rgb_to_hue(*(self._rgb_component(state, param) for param in _RGB_PARAMS))

    @staticmethod
    def _rgb_component(state: Mapping[str, object], param: str) -> float:
        """getPilot의 r/g/b 한 성분. 없거나 비수치면 0 — CCT 모드에서는 아예 안 온다."""
        value = state.get(param)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        return float(value)

    def _verify_color(self, target: str, *, expected_hue: float) -> AdapterResult:
        # 명령은 이미 나갔다 — 확인 실패는 FAILED가 아니라 UNVERIFIED다(_verify와 같은 규약).
        try:
            actual = self._read_hue(target)
        except _Failure as failure:
            return AdapterResult(
                AdapterStatus.UNVERIFIED, f"sent, readback failed: {failure.result.detail}"
            )
        if _hue_distance_deg(actual, expected_hue) <= _HUE_TOLERANCE_DEG:
            return AdapterResult(AdapterStatus.VERIFIED, f"color = {expected_hue:.0f}deg")
        return AdapterResult(
            AdapterStatus.UNVERIFIED,
            f"sent color {expected_hue:.0f}deg but device reports {actual:.0f}deg",
        )

    # -- RPC helpers ---------------------------------------------------------

    def _rpc(self, target: str, method: str, params: Mapping[str, object]) -> Mapping[str, object]:
        assert self._config is not None  # execute() guards this
        try:
            response = self._transport.send(
                target, {"method": method, "params": dict(params)}, self._config.timeout_s
            )
        except WizTimeout as exc:
            raise _Failure(
                AdapterResult(AdapterStatus.FAILED, f"request timed out: {exc}")
            ) from exc
        except Exception as exc:  # noqa: BLE001 - 어떤 전송 실패든 정직한 FAILED
            raise _Failure(
                AdapterResult(AdapterStatus.FAILED, f"network error: {type(exc).__name__}")
            ) from exc
        result = response.get("result")
        if not isinstance(result, dict):
            raise _Failure(
                AdapterResult(AdapterStatus.FAILED, f"invalid {method} response")
            )
        return result

    def _set_pilot(self, target: str, params: Mapping[str, object]) -> None:
        result = self._rpc(target, "setPilot", params)
        # WiZ는 {"result":{"success":true}}로 답한다. 명시적으로 false면 실패로 본다.
        if result.get("success") is False:
            raise _Failure(
                AdapterResult(AdapterStatus.FAILED, "device rejected setPilot")
            )

    def _read_state(self, target: str) -> Mapping[str, object]:
        return self._rpc(target, "getPilot", {})

    def _read_number(self, target: str, param: str) -> float:
        value = self._read_state(target).get(param)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise _Failure(
                AdapterResult(AdapterStatus.FAILED, f"{param} is not numeric: {value!r}")
            )
        try:
            return float(value)
        except ValueError as exc:
            raise _Failure(
                AdapterResult(AdapterStatus.FAILED, f"{param} is not numeric: {value!r}")
            ) from exc

    def _verify(
        self, target: str, param: str, *, expected: object, tolerance: float = 0.0
    ) -> AdapterResult:
        # 명령은 이미 나갔다. 확인에 실패하면 FAILED가 아니라 UNVERIFIED(보냄, 미확인)다.
        try:
            state = self._read_state(target)
        except _Failure as failure:
            return AdapterResult(
                AdapterStatus.UNVERIFIED, f"sent, readback failed: {failure.result.detail}"
            )
        reported = state.get(param)
        if reported is None:
            return AdapterResult(AdapterStatus.UNVERIFIED, "sent, state not readable")
        if _matches(reported, expected, tolerance):
            return AdapterResult(AdapterStatus.VERIFIED, f"state confirmed: {reported}")
        return AdapterResult(
            AdapterStatus.UNVERIFIED, f"sent, state {reported} != expected {expected}"
        )


def _matches(reported: object, expected: object, tolerance: float) -> bool:
    if isinstance(expected, bool):
        return bool(reported) is expected
    try:
        return abs(float(reported) - float(expected)) <= tolerance  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


# --- Hardware boundary ------------------------------------------------------


def _broadcast_addresses() -> list[str]:
    """전역 브로드캐스트 + 이 호스트의 각 IPv4 인터페이스의 /24 서브넷 브로드캐스트.

    가상 어댑터(WSL·Hyper-V·VPN)가 있는 머신에서 ``255.255.255.255``는 엉뚱한
    인터페이스로 나가 기기에 닿지 않을 수 있다(실측). 서브넷 지정 주소도 함께 쏴서
    올바른 NIC로 나갈 확률을 높인다.
    """
    targets = ["255.255.255.255"]
    for address in _local_ipv4_addresses():
        octets = address.split(".")
        if len(octets) == 4:
            targets.append(".".join(octets[:3] + ["255"]))
    return list(dict.fromkeys(targets))


def _local_ipv4_addresses() -> list[str]:
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(str(info[4][0]))
    except OSError:
        pass
    # 기본 경로가 쓰는 주소(호스트명으로 안 잡히는 구성 대비). 실제로 보내지는 않는다.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        addresses.add(str(probe.getsockname()[0]))
    except OSError:
        pass
    finally:
        probe.close()
    return sorted(a for a in addresses if not a.startswith("127."))


def discover(timeout_s: float = _DEFAULT_DISCOVERY_TIMEOUT_S, port: int = WIZ_PORT) -> dict[str, str]:
    """브로드캐스트로 WiZ 기기를 찾아 ``{mac: ip}``를 돌려준다.

    DHCP로 IP가 바뀌었을 때 MAC으로 다시 찾기 위한 폴백이다. 브로드캐스트를 흘려주지
    않는 네트워크도 있으므로(실측) 이것만으로 주소를 정하지 말고 ``MAC@IP`` 형식으로
    마지막 IP를 함께 주는 편이 안전하다. 응답이 없으면 빈 dict.
    """
    probe = json.dumps({"method": "getPilot", "params": {}}).encode("utf-8")
    found: dict[str, str] = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(0.5)
    try:
        for address in _broadcast_addresses():
            try:
                sock.sendto(probe, (address, port))
            except OSError:
                continue  # 어떤 인터페이스는 브로드캐스트를 거부한다 — 나머지를 계속 시도
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                result = json.loads(data.decode("utf-8")).get("result", {})
            except (ValueError, UnicodeDecodeError):
                continue
            mac = result.get("mac") if isinstance(result, dict) else None
            if isinstance(mac, str):
                found[_normalize_mac(mac)] = addr[0]
    finally:
        sock.close()
    return found


class UdpWizTransport:
    """실제 UDP 전송(WiZ 로컬 프로토콜). MAC 대상은 탐색으로 IP를 찾아 캐시한다."""

    def __init__(self, port: int = WIZ_PORT, discovery_timeout_s: float = _DEFAULT_DISCOVERY_TIMEOUT_S) -> None:
        self._port = port
        self._discovery_timeout_s = discovery_timeout_s
        self._mac_to_ip: dict[str, str] = {}

    def send(
        self, target: str, payload: Mapping[str, object], timeout_s: float
    ) -> Mapping[str, object]:
        parsed = parse_target(target)
        host = self._resolve(parsed, rediscover=False)
        try:
            return self._exchange(host, payload, timeout_s)
        except WizTimeout:
            # 알려진 IP가 죽었다. MAC을 아는 경우에만 재탐색해 DHCP 주소 변경을 흡수한다.
            if parsed.mac is None:
                raise
            self._mac_to_ip.pop(parsed.mac, None)
            retry_host = self._resolve(parsed, rediscover=True)
            if retry_host == host:
                raise  # 재탐색해도 같은 주소면 되던 대로 실패시킨다
            return self._exchange(retry_host, payload, timeout_s)

    def _resolve(self, target: _Target, *, rediscover: bool) -> str:
        """주소를 정한다: 캐시 → 설정된 IP → 브로드캐스트 탐색 순.

        평소 경로에서 탐색을 건너뛰는 것이 중요하다 — 탐색은 수 초가 걸리고, 브로드캐스트를
        차단하는 네트워크에서는 항상 실패한다. IP를 알면 곧장 쏜다.
        """
        if not rediscover:
            if target.mac is not None:
                cached = self._mac_to_ip.get(target.mac)
                if cached is not None:
                    return cached
            if target.host is not None:
                return target.host
        if target.mac is not None:
            self._mac_to_ip.update(discover(self._discovery_timeout_s, self._port))
            discovered = self._mac_to_ip.get(target.mac)
            if discovered is not None:
                return discovered
        if target.host is not None:
            return target.host
        raise WizTimeout(
            f"no WiZ device with MAC {target.mac} found (broadcast discovery returned nothing); "
            "give the address as MAC@IP so the last known IP can be used directly"
        )

    def _exchange(
        self, host: str, payload: Mapping[str, object], timeout_s: float
    ) -> Mapping[str, object]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout_s)
        try:
            sock.sendto(json.dumps(dict(payload)).encode("utf-8"), (host, self._port))
            data, _ = sock.recvfrom(8192)
        except socket.timeout as exc:
            raise WizTimeout(f"no response from {host} in {timeout_s}s") from exc
        finally:
            sock.close()
        try:
            parsed = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON from {host}") from exc
        return dict(parsed) if isinstance(parsed, dict) else {}

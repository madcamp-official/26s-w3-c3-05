"""시연 시작 시 전구에 실제로 닿는지 확인하는 프로브 — "연결됐다"를 지어내지 않는다.

WiZ는 **연결 없는 UDP**라 열어 둘 세션이 없다. 그래서 지금까지 앱은 부팅 시 전구를
한 번도 건드리지 않았고, 화면의 "설정됨"은 `WIZ_DEVICE_TARGETS` 환경변수가 있다는
뜻일 뿐이었다 — 전구가 꺼져 있거나 다른 네트워크에 있어도 똑같이 "설정됨"으로 보였고,
그 사실은 **첫 제스처를 하고 나서야** 드러났다. 이 모듈이 그 공백을 메운다: 시작할 때
`getPilot`을 한 번 보내 도달 여부를 눈으로 확인시킨다.

부수 효과가 하나 더 있는데 이쪽이 실질적으로 더 중요하다 — 대상이 `MAC@IP`인데 IP가
바뀐 경우 첫 명령이 브로드캐스트 재탐색(수 초)을 떠안는다. 프로브가 **어댑터와 같은
transport 인스턴스**를 쓰면 그 탐색을 시작할 때 미리 치러 MAC→IP 캐시를 채워 두므로,
정작 제스처를 했을 때는 곧바로 나간다.

Qt 의존은 워커 클래스에만 있다. 판정 자체(`probe_bulb`)는 순수 함수라 가짜 transport로
네트워크 없이 테스트한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from jarvis.monitoring.virtual_bulb import VirtualBulbState, state_from_pilot
from jarvis.runtime_protocol.adapters.wiz import WizConfig, WizTransport, parse_target

# 시작 프로브는 사용자를 기다리게 하는 화면이 아니라 배경 확인이라, 명령용 타임아웃보다
# 짧게 잡아 실패를 빨리 알린다. 실패해도 기능이 죽지 않는다(명령 때 다시 시도한다).
PROBE_TIMEOUT_S = 2.0


@dataclass(frozen=True, slots=True)
class BulbProbeResult:
    """프로브 결과. `ok=False`여도 기능은 살아 있다 — 명령 시점에 다시 시도한다."""

    ok: bool
    detail: str
    state: VirtualBulbState | None = None
    """실물에서 읽은 상태. 못 읽었으면 None — 화면은 "확인 전"을 유지한다."""


def probe_bulb(
    config: WizConfig | None,
    device_id: str,
    transport: WizTransport,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> BulbProbeResult:
    """전구에 `getPilot`을 한 번 보내 도달 여부를 확인한다.

    실패 사유를 구분해서 돌려준다 — "설정이 없다"와 "설정은 있는데 안 닿는다"는
    사용자가 할 일이 전혀 다르기 때문이다(전자는 `.env`, 후자는 전원·네트워크).
    """
    if config is None:
        return BulbProbeResult(False, "전구 미설정 — .env의 WIZ_DEVICE_TARGETS가 비어 있습니다")
    target = config.device_targets.get(device_id)
    if not target:
        return BulbProbeResult(False, f"'{device_id}'에 매핑된 전구가 없습니다 (WIZ_DEVICE_TARGETS 확인)")

    parsed = parse_target(target)
    where = parsed.host or parsed.mac or target
    try:
        response = transport.send(target, {"method": "getPilot", "params": {}}, timeout_s)
    except Exception as exc:  # noqa: BLE001 - 타임아웃·네트워크 오류를 모두 사유로 보여준다
        return BulbProbeResult(
            False,
            f"전구 연결 실패 ({where}): {type(exc).__name__} — 전원·같은 네트워크인지 확인하세요",
        )

    result = response.get("result") if isinstance(response, dict) else None
    if not isinstance(result, dict):
        return BulbProbeResult(False, f"전구 응답을 해석할 수 없습니다 ({where})")

    state = "켜짐" if result.get("state") else "꺼짐"
    dimming = result.get("dimming")
    brightness = f" · 밝기 {dimming}" if isinstance(dimming, (int, float)) else ""
    # 읽은 상태를 함께 돌려준다 — 화면의 전구가 실물과 맞으려면 여기서 시작해야 한다.
    return BulbProbeResult(
        True, f"전구 연결됨 ({where}) — 현재 {state}{brightness}", state_from_pilot(result)
    )


class BulbProbeWorker(QThread):
    """`probe_bulb`을 GUI 스레드 밖에서 실행한다.

    프로브는 동기 UDP라 응답이 없으면 타임아웃만큼 붙잡힌다. GUI 스레드에서 부르면
    창이 그만큼 얼어 "부팅이 느리다"가 된다(`ExecuteWorker`와 같은 이유·같은 패턴).
    """

    result_ready = Signal(object)  # BulbProbeResult

    def __init__(
        self,
        config: WizConfig | None,
        device_id: str,
        transport: WizTransport,
        timeout_s: float = PROBE_TIMEOUT_S,
    ) -> None:
        super().__init__()
        self._config = config
        self._device_id = device_id
        self._transport = transport
        self._timeout_s = timeout_s

    def run(self) -> None:
        self.result_ready.emit(
            probe_bulb(self._config, self._device_id, self._transport, self._timeout_s)
        )

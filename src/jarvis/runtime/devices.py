"""기본 device capability model과 adapter 구성.

`configs/gesture_capability_map.json`은 (기기, 제스처) → capability 동작을 정의하고,
`runtime_protocol.protocol.capability`의 device capability model은 각 capability의
타입·연산·범위·step을 정의한다. 두 곳이 어긋나면 — 매핑엔 있는데 capability model엔
없거나, value가 step 격자를 벗어나면 — Intent가 dispatch 직전 검증에서 조용히
거부된다(protocol/engine.py). 이 모듈이 두 정의를 한곳에서 맞춘 기본 기기 집합을
만들어, `tests/unit/runtime/test_devices.py`의 정합성 테스트가 그 일치를 강제한다.

MVP 대상은 `laptop`(Windows/macOS 로컬 입력)과 `room.bulb`(Philips WiZ, 로컬 UDP)다. 스크롤·
볼륨·창 전환은 절대 상태가 없는 상대 연산이라 min/max는 검증용 상한일 뿐이고
increment/decrement delta만 의미가 있다(capability.validate_request 참고).
"""

from __future__ import annotations

from collections.abc import Mapping

from jarvis.gesture_fusion.intent import DEFAULT_INTENT_CONFIG, GestureCapabilityMap, IntentConfig
from jarvis.runtime_protocol.adapters.base import DeviceAdapter
from jarvis.runtime_protocol.adapters.http import HttpTransport, UrllibTransport
from jarvis.runtime_protocol.adapters.smartthings import SmartThingsAdapter, SmartThingsConfig
from jarvis.runtime_protocol.adapters.windows import InputSink, WindowsAdapter, default_input_sink
from jarvis.runtime_protocol.adapters.wiz import (
    UdpWizTransport,
    WizAdapter,
    WizConfig,
    WizTransport,
)
from jarvis.runtime_protocol.capture.clock import RuntimeClock
from jarvis.runtime_protocol.protocol.capability import (
    BooleanCapability,
    Capability,
    DeviceProfile,
    DeviceRegistry,
    NumberCapability,
    Operation,
)
from jarvis.runtime_protocol.protocol.engine import ProtocolEngine
from jarvis.runtime.executor import IntentExecutor

# 상대 연산(스크롤·볼륨·창 전환)은 increment/decrement만 지원한다 — set은 절대
# 상태를 요구하는데 이 기기들엔 그 개념이 없다. WindowsAdapter도 이 둘만 처리한다.
_RELATIVE_OPS = frozenset({Operation.INCREMENT, Operation.DECREMENT})

# adapter 라우팅 키. DeviceProfile.adapter가 이 이름으로 adapter를 찾는다.
LAPTOP_ADAPTER = "windows"
# 실제 데모 전구는 Philips WiZ(로컬 UDP)라 기본 전구 경로는 이쪽이다. SmartThings
# 경로도 배선은 유지한다 — 클라우드 전구를 쓰려면 프로필의 adapter만 바꾸면 된다.
BULB_ADAPTER = "wiz"
SMARTTHINGS_ADAPTER = "smartthings"


def _laptop_profile() -> DeviceProfile:
    """노트북 capability. scroll/volume/desktop_switch 모두 상대 연산 전용."""
    relative: Capability = NumberCapability(
        minimum=0, maximum=100, step=1, operations=_RELATIVE_OPS
    )
    return DeviceProfile(
        device_id="laptop",
        adapter=LAPTOP_ADAPTER,
        capabilities={
            "scroll": relative,
            "volume": relative,
            "desktop_switch": relative,
        },
    )


def _bulb_profile() -> DeviceProfile:
    """전구 capability. 실측 WiZ 모델(ESP25_SHRGB_01)의 범위와 맞춘다.

    brightness 하한이 0이 아니라 **10**인 것은 기기의 `minDimLevel`이 10이기 때문이다 —
    0은 전구가 거부한다(끄기는 power capability의 몫이다). color_temperature 2700~6500은
    기기가 보고한 `cctRange`의 표준 구간과 정확히 일치한다.

    `color`(색상각, 도)는 다른 수치와 성격이 다르다 — **순환량**이라 min/max가 벽이
    아니라 한 바퀴의 경계다. 그래서 WizAdapter는 이 capability만 클램프하지 않고 360도에서
    0도로 감아 돈다(`_apply_color`). 여기 min/max는 `set` 검증용 경계로만 쓰이며,
    상대 연산(increment/decrement)에서는 delta가 step의 배수인지만 검사된다.
    """
    return DeviceProfile(
        device_id="room.bulb",
        adapter=BULB_ADAPTER,
        capabilities={
            "power": BooleanCapability(),
            "brightness": NumberCapability(minimum=10, maximum=100, step=10),
            "color_temperature": NumberCapability(minimum=2700, maximum=6500, step=100),
            "color": NumberCapability(
                minimum=0, maximum=360, step=30, operations=_RELATIVE_OPS
            ),
        },
    )


def build_default_registry() -> DeviceRegistry:
    """MVP 기본 기기(laptop·room.bulb) 레지스트리.

    실 사용에서는 사용자별 calibration·환경에 따라 기기가 늘거나 줄지만, 데모·
    테스트·정합성 검증의 기준점으로 쓰는 고정 구성이다.
    """
    profiles = {p.device_id: p for p in (_laptop_profile(), _bulb_profile())}
    return DeviceRegistry(profiles)


def build_default_adapters(
    *,
    input_sink: InputSink | None = None,
    wiz_config: WizConfig | None = None,
    wiz_transport: WizTransport | None = None,
    smartthings_config: SmartThingsConfig | None = None,
    http_transport: HttpTransport | None = None,
) -> dict[str, DeviceAdapter]:
    """adapter 이름 → adapter 인스턴스.

    `input_sink`가 없으면 현재 OS에 맞는 실제 sink를 고른다(Windows/macOS). 지원하지
    않는 OS에서는 `default_input_sink()`가 정직하게 raise한다. `wiz_config`(또는
    `smartthings_config`)가 없으면 해당 전구 adapter는 UNCONFIGURED를 반환한다 — 설정
    없이도 배선은 완성되고 실행만 안전하게 실패한다(development-principles 6.3).
    테스트는 fake sink·transport를 주입해 실제 입력·네트워크 없이 배선을 검증한다.
    """
    sink = input_sink if input_sink is not None else default_input_sink()
    transport = http_transport if http_transport is not None else UrllibTransport()
    udp = wiz_transport if wiz_transport is not None else UdpWizTransport()
    return {
        LAPTOP_ADAPTER: WindowsAdapter(sink),
        BULB_ADAPTER: WizAdapter(wiz_config, udp),
        SMARTTHINGS_ADAPTER: SmartThingsAdapter(smartthings_config, transport),
    }


def build_default_capability_map() -> GestureCapabilityMap:
    """`configs/gesture_capability_map.json`을 로드한 (기기,제스처)→동작 매핑."""
    return GestureCapabilityMap.from_json()


def build_laptop_only_executor(
    *,
    input_sink: InputSink | None = None,
    clock: RuntimeClock | None = None,
    capability_map: GestureCapabilityMap | None = None,
    intent_config: IntentConfig = DEFAULT_INTENT_CONFIG,
) -> IntentExecutor:
    """전구 없이 노트북 경로만으로 완결되는 실행기.

    네트워크·전구 설정 없이 Fusion→명령까지 실제로 도는 최소 구성이다. 전구 프로파일은
    레지스트리에 남지만 WiZ config가 없어 실행 시 UNCONFIGURED가 된다 — 라우팅·검증
    배선은 그대로 살아 있다.
    """
    registry = build_default_registry()
    engine = ProtocolEngine(registry, clock if clock is not None else RuntimeClock())
    adapters = build_default_adapters(input_sink=input_sink)
    resolved_map = capability_map if capability_map is not None else build_default_capability_map()
    return IntentExecutor(
        engine=engine,
        registry=registry,
        adapters=adapters,
        capability_map=resolved_map,
        intent_config=intent_config,
    )


def registry_capabilities(registry: DeviceRegistry, device_id: str) -> Mapping[str, Capability]:
    """정합성 테스트 편의: 기기의 capability 매핑을 꺼낸다(없으면 KeyError)."""
    profile = registry.get(device_id)
    if profile is None:
        raise KeyError(device_id)
    return profile.capabilities

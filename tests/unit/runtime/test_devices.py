"""configs/gesture_capability_map.json ↔ device capability model 정합성.

이 테스트가 지키는 불변식: gesture 매핑에 있는 모든 (기기, 제스처) 동작은 기본
레지스트리의 실제 capability model을 통과해야 한다. 둘 중 하나만 바뀌어 어긋나면
(매핑엔 있는데 capability가 없거나, value가 step 격자를 벗어나면) 런타임에서 Intent가
dispatch 직전에 조용히 거부되는데, 그 드리프트를 여기서 배포 전에 잡는다.
"""

from __future__ import annotations

import json
from pathlib import Path

from jarvis.runtime.devices import (
    BULB_ADAPTER,
    LAPTOP_ADAPTER,
    build_default_registry,
)
from jarvis.runtime_protocol.protocol.capability import validate_request

_MAP_PATH = Path(__file__).resolve().parents[3] / "configs" / "gesture_capability_map.json"


def _map_entries() -> list[tuple[str, str, str, str, object]]:
    """(device_id, gesture, capability, operation, value) 튜플 목록."""
    raw = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
    entries: list[tuple[str, str, str, str, object]] = []
    for device_id, gestures in raw["devices"].items():
        for gesture, action in gestures.items():
            entries.append(
                (device_id, gesture, action["capability"], action["operation"], action["value"])
            )
    return entries


def test_every_mapped_action_validates_against_capability_model() -> None:
    registry = build_default_registry()
    failures: list[str] = []
    for device_id, gesture, capability_name, operation, value in _map_entries():
        profile = registry.get(device_id)
        if profile is None:
            failures.append(f"{device_id}: 레지스트리에 없음 (제스처 {gesture})")
            continue
        capability = profile.capabilities.get(capability_name)
        if capability is None:
            failures.append(f"{device_id}.{capability_name}: capability model에 없음")
            continue
        failure = validate_request(capability, operation, value)
        if failure is not None:
            failures.append(
                f"{device_id}.{capability_name} {operation} {value!r}: {failure.detail}"
            )
    assert not failures, "gesture 매핑과 capability model 불일치:\n" + "\n".join(failures)


def test_registry_devices_route_to_known_adapters() -> None:
    registry = build_default_registry()
    laptop = registry.get("laptop")
    bulb = registry.get("room.bulb")
    assert laptop is not None and laptop.adapter == LAPTOP_ADAPTER
    assert bulb is not None and bulb.adapter == BULB_ADAPTER


def test_all_mapped_devices_are_registered() -> None:
    registry = build_default_registry()
    mapped_devices = {device_id for device_id, *_ in _map_entries()}
    for device_id in mapped_devices:
        assert device_id in registry, f"매핑된 기기 {device_id!r}가 레지스트리에 없음"

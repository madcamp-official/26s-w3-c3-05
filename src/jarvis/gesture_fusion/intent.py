"""Intent 조립·출력 — README 9장 Intent 예시, Fusion→Protocol 계약(§3).

Task 6·7(`fusion.py`, `dedup.py`)이 만든 `CommitDecision`(committed=True,
`intent_id` 있음)을 받아, capability/operation/value로 어떻게 바꿀지는 여기서
gesture→capability 매핑(`configs/gesture_capability_map.json`)으로 결정해
`jarvis.contracts.Intent`를 최종 조립한다.

제스처→capability 매핑은 코드가 아니라 config 데이터로 관리한다(documents/
gesture-fusion.md 설계 노트, 2026-07-18) — 커스텀 제스처·신규 기기를 코드 수정
없이 추가하기 위함이다. 매핑 키는 (target device_id, gesture) 쌍이다 — 같은
제스처도 기기에 따라 다른 동작을 의미한다(README 9·15장: 노트북 Swipe Down은
스크롤, 전구 Swipe Down은 밝기 감소).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from jarvis.contracts.messages import Intent
from jarvis.gesture_fusion.fusion import CommitDecision

CapabilityValue = int | float | bool

# repo_root/configs/gesture_capability_map.json — src/jarvis/gesture_fusion/intent.py
# 기준 4단계 상위.
_DEFAULT_MAP_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "gesture_capability_map.json"
)


@dataclass(frozen=True, slots=True)
class CapabilityAction:
    """(target, gesture) 한 쌍이 만드는 capability 동작 하나."""

    capability: str
    operation: str
    value: CapabilityValue

    def __post_init__(self) -> None:
        if not self.capability:
            raise ValueError("capability must not be empty")
        if not self.operation:
            raise ValueError("operation must not be empty")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("value must be finite")


class GestureCapabilityMap:
    """(device_id, gesture) → `CapabilityAction` 조회.

    JSON 데이터로 구성한다 — 매핑 추가·변경은 배포(코드 변경)가 아니라 설정
    변경이다(development-principles.md 8절: threshold·매핑 변경은 config로).
    """

    def __init__(self, mapping: Mapping[str, Mapping[str, CapabilityAction]]) -> None:
        self._mapping = {device: dict(actions) for device, actions in mapping.items()}

    @classmethod
    def from_json(cls, path: Path | str = _DEFAULT_MAP_PATH) -> "GestureCapabilityMap":
        """`configs/gesture_capability_map.json` 형식을 읽어 매핑을 만든다.

        최상위의 `_`로 시작하는 키(예: `_comment`)는 무시한다. 실제 매핑은
        `devices` 키 아래에 있다.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        devices = raw.get("devices", {})
        mapping: dict[str, dict[str, CapabilityAction]] = {}
        for device_id, gestures in devices.items():
            mapping[device_id] = {
                gesture: CapabilityAction(
                    capability=entry["capability"],
                    operation=entry["operation"],
                    value=entry["value"],
                )
                for gesture, entry in gestures.items()
            }
        return cls(mapping)

    def lookup(self, device_id: str, gesture: str) -> CapabilityAction | None:
        """등록되지 않은 조합이면 `None`(호출자는 실행이 아니라 거부로 다뤄야 한다)."""
        return self._mapping.get(device_id, {}).get(gesture)

    def registered_gestures(self) -> frozenset[str]:
        """모든 기기에 걸쳐 매핑에 등장하는 gesture 문자열 전체(중복 제거)."""
        return frozenset(
            gesture for actions in self._mapping.values() for gesture in actions
        )


def validate_capability_map(
    capability_map: GestureCapabilityMap,
    gesture_labels: tuple[str, ...],
    background_labels: frozenset[str] = frozenset({"none"}),
) -> None:
    """`capability_map`과 학습 라벨 집합(`gesture_labels`) 간 불변식을 검사한다.

    `training.data.jester_labels.validate_mapping()`과 같은 목적의 검사를
    gesture→capability 매핑에도 적용한다 — 지금까지는 아무 검증이 없어서, 라벨
    집합이 바뀌어도(2026-07-20: swipe 제거·slide_two_fingers 추가) capability
    map이 조용히 안 맞는 상태로 남을 수 있었다(development-principles.md
    "no silent caps").

    (1) `capability_map`이 참조하는 gesture 문자열은 전부 `gesture_labels`에
        있어야 한다(오타·삭제된 라벨 방지 — 예: 라벨에서 뺀 "swipe_up"이 매핑에
        그대로 남는 경우).
    (2) 배경이 아닌 `gesture_labels` 전부가 최소 하나의 (device, gesture) 매핑을
        가져야 한다(신규 라벨 추가 후 매핑을 깜빡하는 것 방지). 배경 label(기본
        "none")은 "동작 없음"이 곧 의도이므로 이 요구에서 제외한다.
    """
    registered = capability_map.registered_gestures()
    unknown = registered - set(gesture_labels)
    if unknown:
        raise ValueError(
            f"gesture_capability_map.json이 존재하지 않는 gesture를 참조한다: {sorted(unknown)}. "
            f"현재 라벨: {sorted(gesture_labels)}. 라벨이 바뀐 뒤 맵을 갱신하지 않은 상태일 수 있다."
        )

    unmapped = {label for label in gesture_labels if label not in background_labels} - registered
    if unmapped:
        raise ValueError(
            f"gesture_capability_map.json에 매핑이 전혀 없는(배경 아닌) 라벨이 있다: "
            f"{sorted(unmapped)}. 의도적으로 아직 기기 동작이 없다면 이 함수 호출부에서 "
            "명시적으로 허용하거나 configs/gesture_capability_map.json에 매핑을 추가하라."
        )


@dataclass(frozen=True, slots=True)
class IntentConfig:
    """Intent 조립에 쓰는 파라미터."""

    default_expires_in_ms: int = 1000
    """`Intent.expires_in_ms` 기본값. README 9장 Intent 예시와 동일한 값."""

    def __post_init__(self) -> None:
        if self.default_expires_in_ms <= 0:
            raise ValueError("default_expires_in_ms must be positive")


DEFAULT_INTENT_CONFIG = IntentConfig()


def assemble_intent(
    decision: CommitDecision,
    capability_map: GestureCapabilityMap,
    config: IntentConfig = DEFAULT_INTENT_CONFIG,
) -> Intent | None:
    """커밋된 `CommitDecision`을 `jarvis.contracts.Intent`(계약 §3)로 조립한다.

    다음 중 하나라도 해당하면 `None`을 반환한다 — 알 수 없는 조합은 실행이 아니라
    거부다(development-principles.md 2.2):

    - `decision.committed`가 `False`이거나 `intent_id`가 없음(커밋되지 않은 이벤트)
    - `(target, gesture)`에 대한 매핑이 `capability_map`에 없음(신규/미등록 조합)
    """
    if not decision.committed or decision.intent_id is None:
        return None
    if decision.target is None or decision.gesture is None or decision.score is None:
        return None

    action = capability_map.lookup(decision.target, decision.gesture)
    if action is None:
        return None

    return Intent(
        intent_id=decision.intent_id,
        target=decision.target,
        gesture=decision.gesture,
        capability=action.capability,
        operation=action.operation,
        value=action.value,
        target_confidence=decision.score.target_confidence,
        gesture_confidence=decision.score.gesture_confidence,
        expires_in_ms=config.default_expires_in_ms,
    )

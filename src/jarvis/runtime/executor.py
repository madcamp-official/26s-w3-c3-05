"""IntentExecutor — Fusion `CommitDecision`을 실제 기기 명령까지 잇는 마지막 배선.

Fusion은 "커밋해도 되는가"만 판정하고(`CommitDecision`), Protocol은 `Intent`를
받아 검증·dispatch한다. 그 사이 네 단계 — Intent 조립 → submit → 라우팅 → dispatch —
를 한 호출로 묶어 각 단계의 결과를 하나의 `ExecutionOutcome`으로 정직하게 보고한다.

안전 기본값은 항상 비실행이다(development-principles 2.7). 매핑이 없거나(신규 조합),
검증에 실패하거나, TTL이 지났거나, adapter가 실패하면 명령은 실행되지 않고 그 사유가
`ExecutionOutcome`에 남는다 — 어느 단계에서 멈췄는지가 trace로 드러난다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from jarvis.contracts.messages import Intent
from jarvis.gesture_fusion.fusion import CommitDecision
from jarvis.gesture_fusion.intent import (
    DEFAULT_INTENT_CONFIG,
    GestureCapabilityMap,
    IntentConfig,
    assemble_intent,
)
from jarvis.runtime_protocol.adapters.base import (
    DeviceAdapter,
    DispatchCoordinator,
    DispatchReport,
)
from jarvis.runtime_protocol.protocol.capability import DeviceRegistry
from jarvis.runtime_protocol.protocol.engine import Accepted, ProtocolEngine, Rejected
from jarvis.runtime_protocol.protocol.lifecycle import CommandState


class ExecutionStage(StrEnum):
    """`CommitDecision`이 실행 파이프라인의 어느 지점에서 결론 났는지."""

    NOT_COMMITTED = "NOT_COMMITTED"
    """Fusion이 커밋하지 않은 이벤트(거부·미완결). Intent를 만들지 않는다."""

    NO_MAPPING = "NO_MAPPING"
    """커밋됐지만 (기기, 제스처)에 대한 capability 매핑이 없다(신규/미등록 조합)."""

    REJECTED = "REJECTED"
    """Intent가 protocol 검증에서 거부됐다(미등록 기기·capability, 범위 밖 값, 만료 등)."""

    DISPATCHED = "DISPATCHED"
    """명령이 adapter까지 전달됐다 — 최종 상태는 `dispatch.final_state`가 정직하게 담는다."""


# adapter까지 전달됐고, 그 결과가 실제 적용으로 인정되는 최종 상태.
_EXECUTED_STATES = frozenset({CommandState.ACKNOWLEDGED, CommandState.VERIFIED})


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """한 `CommitDecision`을 실행 파이프라인에 흘려보낸 결과.

    `executed`는 명령이 adapter에 전달되어 ACKNOWLEDGED/VERIFIED로 끝났을 때만
    True다 — dispatch까지 갔어도 EXPIRED·FAILED·REJECTED면 False다(성공을 지어내지
    않는다).
    """

    stage: ExecutionStage
    detail: str
    executed: bool
    intent: Intent | None
    command_id: str | None
    dispatch: DispatchReport | None
    rejection: Rejected | None


class IntentExecutor:
    """`CommitDecision` → 기기 명령까지의 조립 지점.

    자체 상태를 두지 않고 protocol engine·dispatch coordinator에 위임한다 —
    중복 방지·TTL·라우팅은 이미 그쪽 계층의 책임이다. 이 클래스는 그 계층들을
    Fusion 출력에 연결하기만 한다.
    """

    def __init__(
        self,
        engine: ProtocolEngine,
        registry: DeviceRegistry,
        adapters: dict[str, DeviceAdapter],
        capability_map: GestureCapabilityMap,
        intent_config: IntentConfig = DEFAULT_INTENT_CONFIG,
    ) -> None:
        self._engine = engine
        self._coordinator = DispatchCoordinator(engine, registry, adapters)
        self._capability_map = capability_map
        self._intent_config = intent_config

    def execute(self, decision: CommitDecision) -> ExecutionOutcome:
        """커밋된 결정을 검증·dispatch까지 흘려보낸다.

        커밋되지 않았거나 매핑이 없으면 명령을 만들지 않고 그 사유만 반환한다.
        커밋·매핑이 있으면 submit→dispatch까지 진행하고, 각 단계 결과를 담는다.
        """
        intent = assemble_intent(decision, self._capability_map, self._intent_config)
        if intent is None:
            return self._no_intent_outcome(decision)

        result = self._engine.submit(intent)
        if isinstance(result, Rejected):
            return ExecutionOutcome(
                stage=ExecutionStage.REJECTED,
                detail=f"{result.reason.value}: {result.detail}",
                executed=False,
                intent=intent,
                command_id=None,
                dispatch=None,
                rejection=result,
            )

        assert isinstance(result, Accepted)
        report = self._coordinator.dispatch(result.command.command_id)
        return ExecutionOutcome(
            stage=ExecutionStage.DISPATCHED,
            detail=report.detail,
            executed=report.final_state in _EXECUTED_STATES,
            intent=intent,
            command_id=result.command.command_id,
            dispatch=report,
            rejection=None,
        )

    def _no_intent_outcome(self, decision: CommitDecision) -> ExecutionOutcome:
        if not decision.committed or decision.intent_id is None:
            stage = ExecutionStage.NOT_COMMITTED
            detail = f"not committed: {decision.reason}"
        else:
            stage = ExecutionStage.NO_MAPPING
            detail = (
                f"no capability mapping for (target={decision.target!r}, "
                f"gesture={decision.gesture!r})"
            )
        return ExecutionOutcome(
            stage=stage,
            detail=detail,
            executed=False,
            intent=None,
            command_id=None,
            dispatch=None,
            rejection=None,
        )

"""Composition root: wires Gaze·Gesture·Fusion·Protocol into one runtime.

세 핵심 모듈은 서로의 내부를 직접 import하지 않고 `jarvis.contracts` 타입으로만
연결된다(documents/repository-structure.md 의존 방향). 그 계약들을 실제 실행
흐름으로 잇는 마지막 배선이 이 패키지다:

    Fusion `CommitDecision`
      → assemble_intent (gesture_fusion.intent)          → `Intent`
      → ProtocolEngine.submit (runtime_protocol.protocol) → `Command`
      → DispatchCoordinator.dispatch (runtime_protocol.adapters) → 실제 기기

`IntentExecutor`가 이 4단계를 하나의 호출로 묶고, `devices.py`가 gesture→capability
매핑(`configs/gesture_capability_map.json`)과 device capability model
(`runtime_protocol.protocol.capability`)이 정합하도록 기본 기기·adapter를 구성한다.
"""

from __future__ import annotations

from jarvis.runtime.devices import (
    build_default_adapters,
    build_default_registry,
    build_laptop_only_executor,
)
from jarvis.runtime.executor import ExecutionOutcome, ExecutionStage, IntentExecutor

__all__ = [
    "ExecutionOutcome",
    "ExecutionStage",
    "IntentExecutor",
    "build_default_adapters",
    "build_default_registry",
    "build_laptop_only_executor",
]

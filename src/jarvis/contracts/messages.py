"""Typed module-boundary messages from documents/interface-contract.md.

Changing these types requires updating the interface contract and decisions log
before producer and consumer implementations are changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GesturePhase(StrEnum):
    IDLE = "IDLE"
    ONSET = "ONSET"
    ACTIVE = "ACTIVE"
    ENDING = "ENDING"


@dataclass(frozen=True, slots=True)
class TargetEstimate:
    timestamp_ms: int
    frame_id: int
    target: str
    probability: float
    second_best_probability: float
    stability: float


@dataclass(frozen=True, slots=True)
class GestureEstimate:
    timestamp_ms: int
    frame_id: int
    gesture: str
    gesture_confidence: float
    phase: GesturePhase
    phase_confidence: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class Intent:
    intent_id: str
    target: str
    gesture: str
    capability: str
    operation: str
    value: int | float | bool
    target_confidence: float
    gesture_confidence: float
    expires_in_ms: int


@dataclass(frozen=True, slots=True)
class Command:
    command_id: str
    intent_id: str
    device_id: str
    capability: str
    operation: str
    value: int | float | bool
    expires_at_ms: int


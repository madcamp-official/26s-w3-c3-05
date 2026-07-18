"""Monitor snapshot: a flat, render-ready view of the whole pipeline's state.

The debug UI renders a :class:`MonitorSnapshot`. The snapshot is deliberately
decoupled from module internals — it depends only on the shared contracts and
telemetry types — so it can be populated from a live run later or from mock data
now, without the UI reaching into Gaze/Gesture/Fusion/Protocol internals.

Honesty (development-principles: monitoring must not fabricate state): views
carry the real state names (UNKNOWN, UNVERIFIED, FAILED, ...) verbatim; the
renderer colors them but never upgrades a status.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from jarvis.contracts.messages import Command
from jarvis.runtime_protocol.telemetry.events import TraceEvent
from jarvis.runtime_protocol.telemetry.latency import LatencyStage, LatencySummary


@dataclass(frozen=True, slots=True)
class CaptureView:
    """Input layer health: is the camera producing frames both streams share?"""

    fps: float | None
    latest_frame_id: int | None
    latest_timestamp_ms: int | None
    face_tracked: bool
    hand_tracked: bool
    queue_drops: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeviceProbability:
    device_id: str
    probability: float


@dataclass(frozen=True, slots=True)
class GazeView:
    available: bool
    lock_state: str
    target: str
    probability: float
    second_best_probability: float
    stability: float
    dwell_ms: float
    lock_ttl_remaining_ms: float | None
    device_probabilities: tuple[DeviceProbability, ...] = ()

    @property
    def margin(self) -> float:
        return self.probability - self.second_best_probability


@dataclass(frozen=True, slots=True)
class GestureView:
    available: bool
    phase: str
    gesture: str
    gesture_confidence: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class CommitCondition:
    """One of the fusion commit conditions (README 9장) and whether it holds now."""

    label: str
    passed: bool


@dataclass(frozen=True, slots=True)
class FusionView:
    available: bool
    state: str
    fusion_score: float
    commit_threshold: float
    conditions: tuple[CommitCondition, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandView:
    command_id: str
    device_id: str
    capability: str
    operation: str
    value: int | float | bool
    state: str
    detail: str
    ttl_remaining_ms: float | None

    @classmethod
    def from_command(
        cls,
        command: Command,
        state: str,
        now_ms: int,
        detail: str = "",
    ) -> CommandView:
        return cls(
            command_id=command.command_id,
            device_id=command.device_id,
            capability=command.capability,
            operation=command.operation,
            value=command.value,
            state=state,
            detail=detail,
            ttl_remaining_ms=float(command.expires_at_ms - now_ms),
        )


@dataclass(frozen=True, slots=True)
class AdapterView:
    name: str
    configured: bool
    last_status: str
    last_detail: str = ""


@dataclass(frozen=True, slots=True)
class LatencyView:
    stage: str
    p50: float
    p95: float
    count: int
    target_ms: float | None

    @property
    def over_target(self) -> bool:
        return self.target_ms is not None and self.p95 > self.target_ms

    @classmethod
    def from_summary(
        cls, stage: LatencyStage, summary: LatencySummary, target_ms: float | None = None
    ) -> LatencyView:
        return cls(
            stage=str(stage),
            p50=summary.p50,
            p95=summary.p95,
            count=summary.count,
            target_ms=target_ms,
        )


@dataclass(frozen=True, slots=True)
class TraceRow:
    timestamp_ms: int
    kind: str
    correlation_id: str
    detail: str

    @classmethod
    def from_event(cls, event: TraceEvent) -> TraceRow:
        return cls(
            timestamp_ms=event.timestamp_ms,
            kind=str(event.kind),
            correlation_id=event.correlation_id,
            detail=event.detail,
        )


@dataclass(frozen=True, slots=True)
class MonitorSnapshot:
    """Everything the debug UI needs for one refresh."""

    source_label: str
    generated_at_ms: int
    capture: CaptureView
    gaze: GazeView
    gesture: GestureView
    fusion: FusionView
    commands: tuple[CommandView, ...] = ()
    adapters: tuple[AdapterView, ...] = ()
    latency: tuple[LatencyView, ...] = ()
    trace: tuple[TraceRow, ...] = ()

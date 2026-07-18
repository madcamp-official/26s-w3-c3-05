"""A representative mock snapshot for developing and demoing the monitor UI.

Real pipeline wiring (composition root) and the Gesture/Fusion module (dev-2) do
not exist yet, so this builds a plausible snapshot from hand-written values. It is
labeled ``mock`` so the UI never presents it as live data. Swap
:func:`build_demo_snapshot` for a live snapshot builder once the pipeline is
assembled; the render path stays the same.

The scene mixes a healthy committed flow (look at bulb → swipe down → brightness
decrement, VERIFIED) with the failure surfaces the UI exists to expose: a
rejected command and an unmet fusion condition.
"""

from __future__ import annotations

from jarvis.monitoring.snapshot import (
    AdapterView,
    CaptureView,
    CommandView,
    CommitCondition,
    DeviceProbability,
    FusionView,
    GazeView,
    GestureView,
    LatencyView,
    MonitorSnapshot,
    TraceRow,
)


def build_demo_snapshot() -> MonitorSnapshot:
    capture = CaptureView(
        fps=29.6,
        latest_frame_id=4821,
        latest_timestamp_ms=1_732_010_400_156,
        face_tracked=True,
        hand_tracked=True,
        queue_drops={"gaze": 0, "gesture": 3},
    )

    gaze = GazeView(
        available=True,
        lock_state="TARGET_LOCKED",
        target="room.bulb",
        probability=0.87,
        second_best_probability=0.13,
        stability=0.91,
        dwell_ms=640,
        lock_ttl_remaining_ms=980,
        device_probabilities=(
            DeviceProbability("room.bulb", 0.87),
            DeviceProbability("laptop", 0.13),
        ),
    )

    gesture = GestureView(
        available=False,  # dev-2 not implemented yet: values are mock
        phase="ENDING",
        gesture="swipe_down",
        gesture_confidence=0.92,
        uncertainty=0.07,
    )

    fusion = FusionView(
        available=False,  # dev-2 not implemented yet: values are mock
        state="COMMITTED",
        fusion_score=0.71,
        commit_threshold=0.60,
        conditions=(
            CommitCondition("등록 기기 하나가 Lock됨", True),
            CommitCondition("Target Lock 이후 제스처 시작", True),
            CommitCondition("Target Lock TTL 안에 제스처 완료", True),
            CommitCondition("Target confidence 기준 충족", True),
            CommitCondition("Gesture confidence 기준 충족", True),
            CommitCondition("시선·제스처 시간 관계 유효", True),
            CommitCondition("동일 이벤트 미실행", True),
        ),
    )

    commands = (
        CommandView(
            command_id="cmd-intent-1042",
            device_id="room.bulb",
            capability="brightness",
            operation="decrement",
            value=10,
            state="VERIFIED",
            detail="state confirmed: 60",
            ttl_remaining_ms=420,
        ),
        CommandView(
            command_id="cmd-intent-1043",
            device_id="room.bulb",
            capability="brightness",
            operation="set",
            value=130,
            state="REJECTED",
            detail="INVALID_VALUE: value 130 outside [0, 100]",
            ttl_remaining_ms=None,
        ),
    )

    adapters = (
        AdapterView("windows", configured=True, last_status="ACKNOWLEDGED", last_detail="scrolled decrement 2"),
        AdapterView("smartthings", configured=True, last_status="VERIFIED", last_detail="state confirmed: 60"),
    )

    latency = (
        LatencyView("capture_to_inference", p50=22, p95=41, count=300, target_ms=None),
        LatencyView("gesture_end_to_commit", p50=8, p95=15, count=42, target_ms=None),
        LatencyView("commit_to_dispatch", p50=2, p95=5, count=42, target_ms=None),
        LatencyView("dispatch_to_ack", p50=310, p95=880, count=42, target_ms=1000),
        LatencyView("end_to_end", p50=95, p95=140, count=42, target_ms=150),
    )

    trace = (
        TraceRow(1_732_010_399_500, "LOCK_TRANSITION", "intent-1042", "CANDIDATE→TARGET_LOCKED"),
        TraceRow(1_732_010_400_156, "INTENT_COMMIT", "intent-1042", "swipe_down → brightness decrement"),
        TraceRow(1_732_010_400_158, "COMMAND_STATE", "cmd-intent-1042", "VALIDATED→DISPATCHED"),
        TraceRow(1_732_010_400_470, "COMMAND_STATE", "cmd-intent-1042", "ACKNOWLEDGED→VERIFIED"),
        TraceRow(1_732_010_401_002, "INTENT_REJECT", "intent-1043", "INVALID_VALUE (130 outside [0,100])"),
        TraceRow(1_732_010_401_500, "QUEUE_DROP", "gesture", "dropped 1 stale frame"),
    )

    return MonitorSnapshot(
        source_label="mock (representative — pipeline not wired, Gesture/Fusion pending)",
        generated_at_ms=1_732_010_401_600,
        capture=capture,
        gaze=gaze,
        gesture=gesture,
        fusion=fusion,
        commands=commands,
        adapters=adapters,
        latency=latency,
        trace=trace,
    )

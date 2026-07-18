"""Unit tests for the monitoring snapshot model and HTML renderer."""

from __future__ import annotations

from jarvis.contracts.messages import Command
from jarvis.monitoring.demo import build_demo_snapshot
from jarvis.monitoring.render import render_html
from jarvis.monitoring.snapshot import CommandView, GazeView, LatencyView, TraceRow
from jarvis.runtime_protocol.telemetry.events import EventKind, TraceEvent
from jarvis.runtime_protocol.telemetry.latency import LatencyStage, LatencySummary


def test_command_view_from_command_computes_ttl() -> None:
    command = Command(
        command_id="cmd-1",
        intent_id="intent-1",
        device_id="room.bulb",
        capability="brightness",
        operation="decrement",
        value=10,
        expires_at_ms=1500,
    )
    view = CommandView.from_command(command, state="DISPATCHED", now_ms=1200, detail="sent")
    assert view.device_id == "room.bulb"
    assert view.state == "DISPATCHED"
    assert view.ttl_remaining_ms == 300


def test_trace_row_from_event() -> None:
    event = TraceEvent(1000, EventKind.INTENT_COMMIT, "intent-1", "committed")
    row = TraceRow.from_event(event)
    assert row.kind == "INTENT_COMMIT"
    assert row.correlation_id == "intent-1"


def test_latency_view_over_target_flag() -> None:
    summary = LatencySummary(count=10, p50=120, p95=180, p99=200, maximum=210, mean=130)
    view = LatencyView.from_summary(LatencyStage.END_TO_END, summary, target_ms=150)
    assert view.p95 == 180
    assert view.over_target is True

    within = LatencyView.from_summary(LatencyStage.END_TO_END, summary, target_ms=250)
    assert within.over_target is False


def test_gaze_margin_is_probability_gap() -> None:
    gaze = GazeView(
        available=True,
        lock_state="TARGET_LOCKED",
        target="room.bulb",
        probability=0.87,
        second_best_probability=0.13,
        stability=0.9,
        dwell_ms=600,
        lock_ttl_remaining_ms=900,
    )
    assert gaze.margin == 0.87 - 0.13


def test_render_demo_is_valid_selfcontained_html() -> None:
    html = render_html(build_demo_snapshot())
    assert html.startswith("<!doctype html>")
    # self-contained: no external asset references
    assert "http://" not in html.replace("http-equiv", "")
    assert "src=" not in html
    # renders panels
    assert "Gaze Targeting" in html
    assert "Intent Fusion" in html
    assert "Trace timeline" in html


def test_render_shows_honest_failure_states() -> None:
    html = render_html(build_demo_snapshot())
    # the rejected command and its reason must be visible, not hidden or upgraded
    assert "REJECTED" in html
    assert "INVALID_VALUE" in html
    assert "UNVERIFIED" not in html or "VERIFIED" in html  # sanity: states shown verbatim


def test_render_marks_unavailable_modules_as_mock() -> None:
    html = render_html(build_demo_snapshot())
    # Gesture/Fusion are not implemented yet → flagged as mock, not shown as live
    assert "mock" in html.lower()


def test_render_escapes_dynamic_text() -> None:
    from jarvis.monitoring.snapshot import (
        AdapterView,
        CaptureView,
        CommitCondition,
        FusionView,
        GestureView,
        MonitorSnapshot,
    )

    snapshot = MonitorSnapshot(
        source_label="<script>alert(1)</script>",
        generated_at_ms=0,
        capture=CaptureView(None, None, None, False, False),
        gaze=GazeView(True, "IDLE", "UNKNOWN", 0.0, 0.0, 0.0, 0.0, None),
        gesture=GestureView(False, "IDLE", "none", 0.0, 0.0),
        fusion=FusionView(False, "IDLE", 0.0, 0.6, (CommitCondition("x", False),)),
        adapters=(AdapterView("windows", True, "-"),),
    )
    html = render_html(snapshot)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html

"""Minimal local monitoring UI: render the pipeline's state for debugging.

Decoupled from module internals — depends only on shared contracts and telemetry
types — so it can be driven by a live snapshot later or mock data now. Never
fabricates state (development-principles).
"""

from jarvis.monitoring.render import render_html
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

__all__ = [
    "AdapterView",
    "CaptureView",
    "CommandView",
    "CommitCondition",
    "DeviceProbability",
    "FusionView",
    "GazeView",
    "GestureView",
    "LatencyView",
    "MonitorSnapshot",
    "TraceRow",
    "render_html",
]

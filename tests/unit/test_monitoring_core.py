"""Unit tests for the monitor's Qt-free core logic."""

from __future__ import annotations

import pytest

from jarvis.monitoring.gesture_source import NullGestureSource
from jarvis.monitoring.messages import MessageLevel, MessageLog
from jarvis.monitoring.pipeline_status import (
    StageState,
    detect_pipeline_status,
)


# --- MessageLog ---------------------------------------------------------------

def test_message_log_records_levels() -> None:
    log = MessageLog()
    log.info("started")
    log.warn("no gesture module")
    log.error("camera failed")
    recent = log.recent()
    assert [m.level for m in recent] == [
        MessageLevel.INFO,
        MessageLevel.WARN,
        MessageLevel.ERROR,
    ]


def test_message_log_recent_limit() -> None:
    log = MessageLog()
    for i in range(5):
        log.info(f"m{i}")
    assert [m.text for m in log.recent(2)] == ["m3", "m4"]


def test_message_log_is_bounded() -> None:
    log = MessageLog(capacity=3)
    for i in range(10):
        log.info(f"m{i}")
    texts = [m.text for m in log.recent()]
    assert texts == ["m7", "m8", "m9"]


def test_message_log_rejects_bad_capacity() -> None:
    with pytest.raises(ValueError):
        MessageLog(capacity=0)


# --- GestureSource ------------------------------------------------------------

def test_null_gesture_source_is_honest() -> None:
    source = NullGestureSource()
    assert source.available is False
    assert source.poll() == []
    assert "미구현" in source.status_text


# --- pipeline status ----------------------------------------------------------

def test_pipeline_status_lists_all_stages() -> None:
    statuses = detect_pipeline_status(env={})
    names = [s.name for s in statuses]
    assert names == [
        "Capture",
        "Gaze Targeting",
        "Gesture Spotter",
        "Intent Fusion",
        "Protocol / Command",
        "Adapters",
    ]


def test_gesture_and_fusion_reported_unavailable() -> None:
    by_name = {s.name: s for s in detect_pipeline_status(env={})}
    assert by_name["Gesture Spotter"].state == StageState.UNAVAILABLE
    assert by_name["Intent Fusion"].state == StageState.UNAVAILABLE


def test_protocol_stage_is_live() -> None:
    by_name = {s.name: s for s in detect_pipeline_status(env={})}
    assert by_name["Protocol / Command"].state == StageState.LIVE


def test_adapters_reflect_smartthings_token() -> None:
    without = {s.name: s for s in detect_pipeline_status(env={})}["Adapters"]
    assert "UNCONFIGURED" in without.detail

    with_token = {
        s.name: s
        for s in detect_pipeline_status(env={"SMARTTHINGS_TOKEN": "abc"})
    }["Adapters"]
    assert "토큰 있음" in with_token.detail


def test_every_stage_has_a_valid_state() -> None:
    for status in detect_pipeline_status(env={}):
        assert isinstance(status.state, StageState)
        assert status.detail

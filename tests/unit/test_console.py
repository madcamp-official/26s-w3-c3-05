"""ConsoleLog 반복 접기·용량, StderrCapture fd 리다이렉트 검증."""

from __future__ import annotations

import os
import time

from jarvis.monitoring.console import ConsoleLog, StderrCapture, _dedup_key


def test_distinct_lines_stay_separate() -> None:
    log = ConsoleLog()
    log.add("first")
    log.add("second")
    lines = log.recent()
    assert [line.text for line in lines] == ["first", "second"]
    assert all(line.count == 1 for line in lines)


def test_repeated_identical_line_collapses_to_count() -> None:
    log = ConsoleLog()
    for _ in range(5):
        log.add("boom")
    lines = log.recent()
    assert len(lines) == 1
    assert lines[0].text == "boom"
    assert lines[0].count == 5


def test_digit_masked_lines_collapse() -> None:
    # clearcut 로그처럼 숫자(타임스탬프·스레드 id)만 다른 줄은 한 줄로 접힌다.
    log = ConsoleLog()
    log.add("E0000 00:00:1784645317.822331   39476 uploader.cc:90] Failed until 2026-07-21")
    log.add("E0000 00:00:1784645377.789700   44860 uploader.cc:90] Failed until 2026-07-22")
    lines = log.recent()
    assert len(lines) == 1
    assert lines[0].count == 2
    # 최신 텍스트를 보관한다(타임스탬프가 최근 것으로 보이게).
    assert "2026-07-22" in lines[0].text


def test_interleaved_repeats_within_window_collapse() -> None:
    # 세 줄짜리 이벤트가 ABCABC로 반복돼도 최근 창 안에서 각각 카운트로 접힌다.
    log = ConsoleLog(dedup_window=6)
    for _ in range(4):
        log.add("=== trace ===")
        log.add("path/to/uploader.cc")
        log.add("Failed to send")
    lines = log.recent()
    assert len(lines) == 3
    assert {line.count for line in lines} == {4}


def test_capacity_is_bounded() -> None:
    # 숫자만 다르면 접히므로, 서로 다른 키가 되도록 알파벳으로 구분한다.
    log = ConsoleLog(capacity=3, dedup_window=1)
    for letter in "abcdefghij":
        log.add(f"line-{letter}")
    lines = log.recent()
    assert len(lines) == 3
    assert [line.text for line in lines] == ["line-h", "line-i", "line-j"]


def test_recent_limit() -> None:
    log = ConsoleLog(dedup_window=1)
    for letter in "abcde":
        log.add(f"n-{letter}")
    assert [line.text for line in log.recent(2)] == ["n-d", "n-e"]


def test_dedup_key_masks_digit_runs() -> None:
    assert _dedup_key("thread 39476 at 12.5") == _dedup_key("thread 44860 at 99.1")
    assert _dedup_key("alpha") != _dedup_key("beta")


def test_stderr_capture_redirects_and_restores() -> None:
    received: list[str] = []
    capture = StderrCapture(received.append)
    capture.start()
    try:
        os.write(2, b"captured-line\n")
        deadline = time.monotonic() + 2.0
        while "captured-line" not in received and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        capture.stop()
    assert "captured-line" in received
    # 복원 후 fd 2는 다시 정상적으로 쓸 수 있어야 한다(파이프가 닫힌 채 남지 않는다).
    os.write(2, b"")

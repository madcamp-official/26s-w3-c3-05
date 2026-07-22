"""실행 워커 — GUI 스레드를 막지 않고 명령을 직렬 실행한다는 계약을 고정한다.

전구 adapter는 동기 네트워크 I/O라 Qt 슬롯에서 직접 부르면 창이 타임아웃만큼
얼어붙는다. 여기서는 가짜 실행기를 주입해 네트워크·기기 없이 큐·시그널·종료
동작만 검증한다.
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from jarvis.gesture_fusion.fusion import CommitDecision  # noqa: E402
from jarvis.monitoring.execute_worker import ExecuteWorker  # noqa: E402
from jarvis.runtime.executor import ExecutionOutcome, ExecutionStage  # noqa: E402


def _decision(frame_id: int = 0, gesture: str = "slide_two_fingers_down") -> CommitDecision:
    return CommitDecision(
        committed=True,
        reason="committed",
        target="laptop",
        gesture=gesture,
        score=None,
        timestamp_ms=frame_id * 10,
        frame_id=frame_id,
        intent_id=f"intent-{frame_id}",
    )


def _outcome(detail: str = "ok", *, executed: bool = True) -> ExecutionOutcome:
    return ExecutionOutcome(
        stage=ExecutionStage.DISPATCHED,
        detail=detail,
        executed=executed,
        intent=None,
        command_id="cmd-1",
        dispatch=None,
        rejection=None,
    )


class _FakeExecutor:
    """호출 순서를 기록하는 가짜 실행기. 실제 기기·네트워크를 건드리지 않는다."""

    def __init__(self, *, delay_s: float = 0.0, raises: bool = False) -> None:
        self.calls: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._delay_s = delay_s
        self._raises = raises
        self._lock = threading.Lock()

    def execute(self, decision: CommitDecision) -> ExecutionOutcome:
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self._raises:
                raise RuntimeError("전구가 응답하지 않습니다")
            if self._delay_s:
                time.sleep(self._delay_s)
            self.calls.append(decision.frame_id)
            return _outcome(f"dispatched #{decision.frame_id}")
        finally:
            with self._lock:
                self.concurrent -= 1


@pytest.fixture(name="qt_app")
def _qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _drain(worker: ExecuteWorker, app: QApplication, timeout_s: float = 3.0) -> None:
    """워커가 큐를 비울 때까지 이벤트를 돌린다(queued 시그널 수신용)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if worker._queue.empty():
            time.sleep(0.05)
            app.processEvents()
            return
        time.sleep(0.01)


# --- 기본 동작 ---


def test_outcome_is_emitted_for_each_submission(qt_app: QApplication) -> None:
    executor = _FakeExecutor()
    worker = ExecuteWorker(executor)
    received: list[ExecutionOutcome] = []
    worker.outcome_ready.connect(received.append)
    worker.start()
    try:
        assert worker.submit(_decision(1)) is True
        _drain(worker, qt_app)
    finally:
        worker.stop()
    qt_app.processEvents()
    assert executor.calls == [1]
    assert len(received) == 1
    assert received[0].detail == "dispatched #1"


def test_commands_run_one_at_a_time(qt_app: QApplication) -> None:
    """ProtocolEngine의 ledger가 단일 소유자 아래 남으려면 직렬이어야 한다."""
    executor = _FakeExecutor(delay_s=0.02)
    worker = ExecuteWorker(executor)
    worker.start()
    try:
        for frame_id in range(5):
            worker.submit(_decision(frame_id))
        _drain(worker, qt_app)
    finally:
        worker.stop()
    assert executor.max_concurrent == 1
    assert executor.calls == [0, 1, 2, 3, 4]


# --- 실패를 삼키지 않는다 ---


def test_executor_exception_reports_and_keeps_thread_alive(qt_app: QApplication) -> None:
    executor = _FakeExecutor(raises=True)
    worker = ExecuteWorker(executor)
    failures: list[str] = []
    worker.failed.connect(failures.append)
    worker.start()
    try:
        worker.submit(_decision(1))
        _drain(worker, qt_app)
        assert worker.isRunning()  # 예외가 워커를 죽이지 않았다
    finally:
        worker.stop()
    qt_app.processEvents()
    assert failures
    assert "전구가 응답하지 않습니다" in failures[0]


def test_full_queue_reports_instead_of_silently_dropping(qt_app: QApplication) -> None:
    """대기열이 넘칠 때 조용히 버리지 않는다(no silent caps)."""
    executor = _FakeExecutor()
    worker = ExecuteWorker(executor)  # 시작하지 않아 큐가 비워지지 않는다
    dropped: list[str] = []
    worker.dropped.connect(dropped.append)
    accepted = [worker.submit(_decision(i)) for i in range(40)]
    assert accepted.count(False) > 0
    assert dropped
    assert "버렸습니다" in dropped[0]


# --- 종료 ---


def test_stop_joins_the_thread(qt_app: QApplication) -> None:
    worker = ExecuteWorker(_FakeExecutor())
    worker.start()
    worker.stop()
    assert not worker.isRunning()


def test_stop_is_safe_when_never_started(qt_app: QApplication) -> None:
    ExecuteWorker(_FakeExecutor()).stop()  # raise하지 않으면 통과

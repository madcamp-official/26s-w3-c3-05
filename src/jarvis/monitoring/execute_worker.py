"""커밋된 Intent를 GUI 스레드 밖에서 실행하는 워커.

`_on_gaze`/`_on_hand`는 Qt 슬롯이라 GUI 스레드에서 돈다. 거기서 곧장
`IntentExecutor.execute()`를 부르면 전구 명령마다 창이 멈춘다 — WiZ adapter는
동기 UDP round trip이고, 전구가 응답하지 않으면 타임아웃(기본 3초)만큼 붙잡힌다.
`CameraWorker`가 무거운 추론을 UI 밖으로 뺀 것과 같은 이유·같은 패턴이다.

**직렬 처리**다. 워커 스레드는 하나뿐이고 큐에서 하나씩 꺼내 실행한다 —
`ProtocolEngine`의 command ledger가 단일 소유자 아래 남아야 상태 전이(VALIDATED→
DISPATCHED)가 경합하지 않는다.
"""

from __future__ import annotations

import queue
from typing import Protocol

from PySide6.QtCore import QThread, Signal

from jarvis.gesture_fusion.fusion import CommitDecision
from jarvis.runtime.executor import ExecutionOutcome

# 대기열 상한. 전구가 죽어 매 명령이 타임아웃까지 가면 큐가 무한히 자란다. 넘치면
# 조용히 버리지 않고 `dropped` 신호로 알린다(development-principles: no silent caps).
_MAX_PENDING = 16


class SupportsExecute(Protocol):
    """`IntentExecutor.execute`만 요구한다 — 테스트가 가짜 실행기를 주입할 수 있게."""

    def execute(self, decision: CommitDecision) -> ExecutionOutcome: ...


class ExecuteWorker(QThread):
    """커밋 판정을 받아 기기 명령까지 실행하고 결과를 시그널로 돌려준다."""

    outcome_ready = Signal(object)  # ExecutionOutcome
    failed = Signal(str)  # 실행기 자체가 예외를 던진 경우(어댑터 실패는 outcome에 담긴다)
    dropped = Signal(str)  # 대기열이 가득 차 버려진 명령

    def __init__(self, executor: SupportsExecute) -> None:
        super().__init__()
        self._executor = executor
        self._queue: queue.Queue[CommitDecision | None] = queue.Queue(maxsize=_MAX_PENDING)

    def submit(self, decision: CommitDecision) -> bool:
        """실행 대기열에 넣는다(GUI 스레드에서 호출). 큐가 가득 차면 False."""
        try:
            self._queue.put_nowait(decision)
        except queue.Full:
            self.dropped.emit(
                f"실행 대기열이 가득 차 명령을 버렸습니다: {decision.gesture} → {decision.target}"
            )
            return False
        return True

    def run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # stop() 이 넣은 종료 신호
                return
            self._process(item)

    def _process(self, decision: CommitDecision) -> None:
        """한 건 실행. 예외가 스레드를 죽이지 않게 삼키되, 조용히 넘기지는 않는다."""
        try:
            outcome = self._executor.execute(decision)
        except Exception as exc:  # noqa: BLE001 - 어떤 실행 오류도 워커를 죽이면 안 된다
            self.failed.emit(f"명령 실행 오류: {exc}")
            return
        self.outcome_ready.emit(outcome)

    def stop(self, timeout_ms: int = 5000) -> None:
        """종료 신호를 넣고 스레드가 끝날 때까지 기다린다(창 닫힐 때 반드시 호출).

        진행 중인 명령 하나는 끝까지 실행된다 — 전구에 절반만 전달된 상태로
        끊는 것보다 낫고, 타임아웃이 있어 무한정 걸리지 않는다.
        """
        if not self.isRunning():
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # 큐가 가득 차 종료 신호조차 못 넣는 경우 — 블로킹 put으로 자리를 기다린다.
            self._queue.put(None)
        self.wait(timeout_ms)

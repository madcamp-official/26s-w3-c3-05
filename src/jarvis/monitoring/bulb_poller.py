"""실물 전구 색을 화면에 그대로 반영하기 위한 백그라운드 조회 스레드.

WiZ 로컬 UDP는 동기 I/O다 — GUI 스레드에서 직접 `getPilot`을 부르면 전구가
응답하지 않을 때 타임아웃(기본 3초)만큼 창이 멎는다. `ExecuteWorker`가 명령
실행을 스레드로 뺀 것과 같은 이유로, 여기서는 주기적 조회를 뺀다.
"""

from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import QThread, Signal

from jarvis.runtime_protocol.adapters.wiz import WizAdapter

_DEFAULT_INTERVAL_MS = 2000


class BulbPoller(QThread):
    """`device_id`의 실물 상태를 일정 주기로 읽어 신호로 알린다.

    조회 실패(설정 없음·통신 오류)도 `None`으로 그대로 신호를 보낸다 — 조용히
    건너뛰면 화면이 마지막으로 성공한 색에 멈춰 실물이 죽었는데 살아있는 것처럼
    보일 수 있다.
    """

    state_ready = Signal(object)  # Mapping[str, object] | None

    def __init__(
        self, adapter: WizAdapter, device_id: str, interval_ms: int = _DEFAULT_INTERVAL_MS
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._device_id = device_id
        self._interval_ms = interval_ms

    def run(self) -> None:
        while not self.isInterruptionRequested():
            state: Mapping[str, object] | None = self._adapter.read_state(self._device_id)
            self.state_ready.emit(state)
            self.msleep(self._interval_ms)

    def stop(self, timeout_ms: int = 5000) -> None:
        if not self.isRunning():
            return
        self.requestInterruption()
        self.wait(timeout_ms)

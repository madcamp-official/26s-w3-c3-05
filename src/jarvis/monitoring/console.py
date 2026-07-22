"""프로세스 stderr(fd 2)를 앱 하단 콘솔 독으로 가로채는 캡처 + 로그 버퍼.

MediaPipe 같은 네이티브 의존성은 로그를 **C++에서 프로세스 stderr로 직접** 쓴다
(예: portable_clearcut_uploader의 clearcut 텔레메트리 실패). Python의 ``sys.stderr``
교체로는 못 잡으므로, OS 레벨에서 ``os.dup2``로 fd 2를 파이프로 돌려 리더 스레드가
읽는다. 읽은 줄은 스레드 안전한 :class:`ConsoleLog`에 넣고, Qt 패널이 타이머로 폴링해
그린다(카메라/리더 스레드가 Qt 위젯을 직접 건드리지 않게 — MessageLog↔MessagePanel과
같은 분리). 이 모듈은 Qt를 import하지 않아 단독 테스트가 가능하다.

반복 노이즈(clearcut가 60초마다 같은 실패를 토함)는 숫자를 마스킹한 키로 최근 창에서
합쳐 ``(xN)`` 카운트로 접는다 — 실제 메시지가 스팸에 묻히지 않게.
"""

from __future__ import annotations

import os
import re
import threading
from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Callable

# 숫자 런(타임스탬프·스레드 id·카운터)을 지워 "같은 메시지의 반복"을 판정하는 키를 만든다.
_DIGITS = re.compile(r"\d+")


def _dedup_key(text: str) -> str:
    return _DIGITS.sub("#", text)


@dataclass(slots=True)
class ConsoleLine:
    """콘솔에 보이는 한 줄. 반복되면 새 줄을 쌓지 않고 ``count``만 올린다."""

    text: str
    key: str
    count: int
    timestamp_ms: int


class ConsoleLog:
    """stderr 줄의 스레드 안전 링 버퍼. 최근 창에서 반복 줄을 카운트로 접는다."""

    def __init__(self, capacity: int = 300, dedup_window: int = 12) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if dedup_window < 1:
            raise ValueError(f"dedup_window must be >= 1, got {dedup_window}")
        self._lines: deque[ConsoleLine] = deque(maxlen=capacity)
        self._dedup_window = dedup_window
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        """한 줄을 추가한다. 최근 ``dedup_window`` 안에 같은(숫자 무시) 줄이 있으면 접는다."""
        key = _dedup_key(text)
        now = int(monotonic() * 1000)
        with self._lock:
            # 뒤에서부터 창 범위만 훑어 같은 키를 찾으면 그 줄을 갱신(최신 텍스트+카운트).
            for line in self._recent_window():
                if line.key == key:
                    line.count += 1
                    line.text = text
                    line.timestamp_ms = now
                    return
            self._lines.append(ConsoleLine(text=text, key=key, count=1, timestamp_ms=now))

    def _recent_window(self) -> list[ConsoleLine]:
        # deque는 슬라이싱이 안 되므로 뒤에서 dedup_window개만 리스트로 뽑는다(락 안에서 호출).
        n = len(self._lines)
        start = max(0, n - self._dedup_window)
        return [self._lines[i] for i in range(n - 1, start - 1, -1)]

    def recent(self, limit: int | None = None) -> list[ConsoleLine]:
        """오래된→최신 순 줄. ``limit``이면 마지막 그만큼만."""
        with self._lock:
            lines = list(self._lines)
        if limit is not None:
            return lines[-limit:]
        return lines


class StderrCapture:
    """fd 2를 파이프로 돌려 리더 스레드로 읽어 ``on_line`` 콜백에 넘긴다.

    ``on_line``은 리더 스레드에서 호출되므로 스레드 안전해야 한다(:meth:`ConsoleLog.add`가
    그렇다). :meth:`stop`은 원래 fd 2를 복원해 프로세스 종료 후에도 stderr가 깨지지 않게 한다.
    """

    def __init__(self, on_line: Callable[[str], None]) -> None:
        self._on_line = on_line
        self._orig_fd: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._orig_fd = os.dup(2)  # 복원용으로 원본 stderr를 보관한다.
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)  # 이제 fd 2로 가는 모든 쓰기(C++ 포함)가 파이프로 들어온다.
        os.close(write_fd)
        self._read_fd = read_fd
        self._running = True
        self._thread = threading.Thread(target=self._pump, name="stderr-capture", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._read_fd is not None
        buffer = b""
        while self._running:
            try:
                chunk = os.read(self._read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break  # 쓰기 끝이 닫힘(정상 종료)
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").rstrip("\r")
                if line:
                    self._on_line(line)

    def stop(self) -> None:
        """캡처를 멈추고 원래 stderr를 복원한다(idempotent)."""
        if not self._running:
            return
        self._running = False
        if self._orig_fd is not None:
            os.dup2(self._orig_fd, 2)  # fd 2를 원래 콘솔로 되돌린다.
            os.close(self._orig_fd)
            self._orig_fd = None
        if self._read_fd is not None:
            os.close(self._read_fd)  # 리더의 os.read가 풀리며 스레드가 빠져나온다.
            self._read_fd = None

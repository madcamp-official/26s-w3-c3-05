"""메인 프로세스에서 레거시 엔진 워커를 자식 프로세스로 부리는 클라이언트.

`LegacyEngineClient`는 격리 venv의 파이썬으로 `legacy_worker.py`를 띄우고, 프레임을
보내면 그 프레임의 랜드마크를 돌려주는 **동기 요청/응답** 인터페이스를 제공한다.
동기로 두는 이유는 A/B 비교에서 두 엔진이 **같은 프레임**을 본 결과끼리만 비교돼야
하기 때문이다 — 비동기 큐를 쓰면 프레임 정렬이 어긋나 편차 수치가 의미를 잃는다.

레거시 엔진은 프레임당 수~수십 ms라 30fps 캡처에서 병목이 되지 않는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import TracebackType

import numpy as np
import numpy.typing as npt

from jarvis.engine_port.protocol import (
    LandmarkResult,
    ProtocolError,
    decode_result,
    encode_frame_header,
)

#: 격리 venv 인터프리터 경로를 덮어쓰는 환경변수.
LEGACY_PYTHON_ENV = "JARVIS_LEGACY_PYTHON"

#: `setup_legacy_env`가 만드는 기본 격리 venv 위치(저장소 루트 기준).
DEFAULT_VENV_DIRNAME = ".venv-legacy"

_WORKER_PATH = Path(__file__).with_name("legacy_worker.py")
_REPO_ROOT = Path(__file__).resolve().parents[3]


class LegacyEngineError(RuntimeError):
    """레거시 워커를 띄우지 못했거나 대화 도중 죽었을 때."""


def default_legacy_python() -> Path | None:
    """격리 venv의 파이썬을 찾는다. 환경변수 > 저장소 루트 `.venv-legacy` 순."""
    override = os.environ.get(LEGACY_PYTHON_ENV)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    candidate = _REPO_ROOT / DEFAULT_VENV_DIRNAME / "bin" / "python"
    return candidate if candidate.is_file() else None


class LegacyEngineClient:
    """참고 레포 레거시 엔진(`mp.solutions.hands`)을 자식 프로세스로 구동한다.

    컨텍스트 매니저로 쓰는 것을 권장한다:

        with LegacyEngineClient(python_path) as engine:
            result = engine.detect(frame_bgr, timestamp_ms)
    """

    def __init__(
        self,
        python_path: Path,
        *,
        min_detection: float = 0.7,
        min_tracking: float = 0.5,
        quiet: bool = True,
    ) -> None:
        """`quiet`이면 워커 stderr(mediapipe 로그)를 버린다 — 화면을 어지럽히지 않도록."""
        if not python_path.is_file():
            raise LegacyEngineError(
                f"격리 venv 파이썬을 찾을 수 없습니다: {python_path}\n"
                "  python -m jarvis.engine_port.setup_legacy_env  로 먼저 만드세요."
            )
        self._process = subprocess.Popen(  # noqa: S603 - 인자는 전부 내부에서 구성
            [
                str(python_path),
                str(_WORKER_PATH),
                "--min-detection",
                str(min_detection),
                "--min-tracking",
                str(min_tracking),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if quiet else None,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise LegacyEngineError("워커 파이프를 열지 못했습니다")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout

    def detect(
        self, frame_bgr: npt.NDArray[np.uint8], timestamp_ms: int
    ) -> LandmarkResult:
        """BGR 프레임 한 장을 레거시 엔진에 넘기고 그 프레임의 결과를 받는다."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"BGR 프레임은 (H, W, 3)이어야 합니다: {frame_bgr.shape}")
        height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        contiguous = np.ascontiguousarray(frame_bgr, dtype=np.uint8)

        try:
            self._stdin.write(encode_frame_header(height, width, timestamp_ms))
            self._stdin.write(contiguous.tobytes())
            self._stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise LegacyEngineError(f"워커가 프레임 전송 도중 죽었습니다: {exc}") from exc

        line = self._stdout.readline()
        if not line:
            raise LegacyEngineError(
                "워커가 결과 없이 종료했습니다. "
                "격리 venv에 solutions 포함 mediapipe가 설치돼 있는지 확인하세요 "
                "(quiet=False로 stderr를 보면 원인이 보입니다)."
            )
        try:
            return decode_result(line)
        except ProtocolError as exc:
            raise LegacyEngineError(f"워커 응답을 해석하지 못했습니다: {exc}") from exc

    def close(self) -> None:
        """stdin을 닫아 워커를 정상 종료시키고, 안 죽으면 강제로 정리한다."""
        process = self._process
        if process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()

    def __enter__(self) -> "LegacyEngineClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def resolve_legacy_python(explicit: str | None) -> Path:
    """CLI 인자 또는 기본 위치에서 격리 venv 파이썬을 찾고, 없으면 안내와 함께 실패."""
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_file():
            raise LegacyEngineError(f"지정한 파이썬이 없습니다: {candidate}")
        return candidate
    found = default_legacy_python()
    if found is None:
        raise LegacyEngineError(
            "격리 venv를 찾지 못했습니다. 아래로 먼저 만드세요:\n"
            f"  {sys.executable} -m jarvis.engine_port.setup_legacy_env"
        )
    return found

"""메인 프로세스(Tasks API) ↔ 레거시 워커(`mp.solutions.hands`) 간 전송 규약.

참고 레포의 랜드마크 **엔진 자체**를 돌리려면 solutions가 살아있는 구버전
mediapipe가 필요하지만, 이 프로젝트 런타임은 0.10.35(slim, solutions 없음)를 쓴다.
두 버전은 한 프로세스에 공존할 수 없으므로 레거시 엔진은 **별도 venv의 자식
프로세스**로 띄우고, 같은 카메라 프레임을 파이프로 흘려 결과만 받아온다.

이 모듈은 그 파이프에 흐르는 바이트 규약만 정의한다. mediapipe·cv2에 의존하지
않고 stdlib + numpy만 쓰므로 **메인 venv와 레거시 venv 양쪽에서 모두 import된다**
(워커가 `src/`를 sys.path에 넣고 이 모듈을 그대로 읽는다).

와이어 포맷
-----------
프레임(메인 → 워커): 고정 길이 헤더 + raw BGR 픽셀.

    | magic 'F' (1B) | height u32 | width u32 | timestamp_ms u64 |

    뒤이어 height * width * 3 바이트의 BGR uint8 픽셀이 그대로 붙는다. 압축하면
    화질 손실이 랜드마크 품질 비교를 오염시키므로 **무압축**으로 보낸다.

결과(워커 → 메인): stdout에 JSON 한 줄(`\\n` 종료). stdout은 이 프로토콜 전용이며
mediapipe가 뱉는 로그는 stderr로 분리된다.

    {"ts": 1234, "points": [[x, y], ...] | null, "handedness": "Left" | null,
     "score": 0.98 | null}

`points`는 이미지 정규화 좌표 [0, 1]의 21점이며, 없으면 null(손 미검출)이다.
종료는 메인이 stdin을 닫으면 워커가 EOF를 보고 스스로 빠져나온다.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: 랜드마크 점 개수 — 레거시 Solutions와 Tasks API가 동일한 21점 토폴로지를 쓴다.
LANDMARK_COUNT = 21

_FRAME_MAGIC = b"F"
_FRAME_STRUCT = struct.Struct("!cIIQ")

#: 프레임 헤더 바이트 수(magic 1 + h 4 + w 4 + ts 8).
FRAME_HEADER_SIZE = _FRAME_STRUCT.size


class ProtocolError(RuntimeError):
    """파이프에서 규약에 맞지 않는 바이트를 읽었을 때."""


@dataclass(frozen=True, slots=True)
class FrameHeader:
    """뒤따를 픽셀 블록의 모양과 그 프레임의 타임스탬프."""

    height: int
    width: int
    timestamp_ms: int

    @property
    def payload_size(self) -> int:
        """헤더 뒤에 이어지는 BGR 픽셀 바이트 수."""
        return self.height * self.width * 3


@dataclass(frozen=True, slots=True)
class LandmarkResult:
    """한 프레임에 대한 엔진의 검출 결과."""

    timestamp_ms: int
    points: FloatArray | None  # (21, 2) 정규화 좌표 [0, 1], 미검출이면 None
    handedness: str | None = None
    score: float | None = None

    @property
    def detected(self) -> bool:
        return self.points is not None


def encode_frame_header(height: int, width: int, timestamp_ms: int) -> bytes:
    """프레임 헤더를 바이트로 직렬화한다(픽셀 블록은 호출자가 이어 붙인다)."""
    return _FRAME_STRUCT.pack(_FRAME_MAGIC, height, width, timestamp_ms)


def decode_frame_header(raw: bytes) -> FrameHeader:
    """`FRAME_HEADER_SIZE` 바이트를 헤더로 역직렬화한다."""
    if len(raw) != FRAME_HEADER_SIZE:
        raise ProtocolError(f"헤더는 {FRAME_HEADER_SIZE}바이트여야 하는데 {len(raw)}바이트를 받음")
    magic, height, width, timestamp_ms = _FRAME_STRUCT.unpack(raw)
    if magic != _FRAME_MAGIC:
        raise ProtocolError(f"프레임 magic이 {_FRAME_MAGIC!r}가 아님: {magic!r}")
    if height <= 0 or width <= 0:
        raise ProtocolError(f"프레임 크기가 유효하지 않음: {width}x{height}")
    return FrameHeader(height=height, width=width, timestamp_ms=timestamp_ms)


def encode_result(result: LandmarkResult) -> bytes:
    """결과를 stdout에 실을 JSON 한 줄(개행 포함)로 직렬화한다."""
    points = None if result.points is None else np.asarray(result.points, dtype=np.float64).tolist()
    payload: dict[str, Any] = {
        "ts": int(result.timestamp_ms),
        "points": points,
        "handedness": result.handedness,
        "score": result.score,
    }
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def decode_result(line: bytes | str) -> LandmarkResult:
    """워커가 보낸 JSON 한 줄을 결과로 역직렬화한다."""
    text = line.decode("utf-8") if isinstance(line, bytes) else line
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"워커 stdout이 JSON이 아님: {text[:200]!r}") from exc
    if not isinstance(payload, dict) or "ts" not in payload:
        raise ProtocolError(f"결과 JSON에 필수 필드가 없음: {text[:200]!r}")

    raw_points = payload.get("points")
    points: FloatArray | None = None
    if raw_points is not None:
        points = np.asarray(raw_points, dtype=np.float64)
        if points.shape != (LANDMARK_COUNT, 2):
            raise ProtocolError(f"랜드마크는 ({LANDMARK_COUNT}, 2)여야 하는데 {points.shape}")

    score = payload.get("score")
    handedness = payload.get("handedness")
    return LandmarkResult(
        timestamp_ms=int(payload["ts"]),
        points=points,
        handedness=None if handedness is None else str(handedness),
        score=None if score is None else float(score),
    )


def read_exactly(stream: Any, size: int) -> bytes | None:  # noqa: ANN401 - 파일류 객체(BinaryIO/파이프)
    """정확히 `size` 바이트를 읽는다. 그 전에 EOF면 None(정상 종료 신호).

    파이프 `read()`는 요청보다 적게 돌려줄 수 있어 루프가 필요하다 — 프레임 픽셀
    블록(수백 KB)에서 실제로 자주 쪼개진다.
    """
    if size == 0:
        return b""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = cast("bytes", stream.read(remaining))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

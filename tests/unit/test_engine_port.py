"""engine_port 검증 — 파이프 규약 왕복과 비교 지표 집계.

레거시 엔진 자체는 격리 venv가 있어야 돌아가므로 여기서는 다루지 않는다. 대신
**엔진 없이도 깨질 수 있는 부분**을 덮는다: 프레임/결과 직렬화가 정확히 왕복하는가,
편차·검출 일치율 집계가 맞는가. 이 둘이 틀리면 A/B 수치 전체가 무의미해진다.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from jarvis.engine_port.metrics import ComparisonAccumulator, JitterTracker, landmark_deviation
from jarvis.engine_port.protocol import (
    FRAME_HEADER_SIZE,
    LANDMARK_COUNT,
    LandmarkResult,
    ProtocolError,
    decode_frame_header,
    decode_result,
    encode_frame_header,
    encode_result,
    read_exactly,
)


def _points(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((LANDMARK_COUNT, 2))


def test_frame_header_round_trip() -> None:
    header = decode_frame_header(encode_frame_header(480, 640, 12345))
    assert (header.height, header.width, header.timestamp_ms) == (480, 640, 12345)
    assert header.payload_size == 480 * 640 * 3


def test_frame_header_rejects_wrong_length() -> None:
    with pytest.raises(ProtocolError):
        decode_frame_header(encode_frame_header(480, 640, 1)[:-1])


def test_frame_header_rejects_bad_magic() -> None:
    corrupted = b"X" + encode_frame_header(480, 640, 1)[1:]
    with pytest.raises(ProtocolError):
        decode_frame_header(corrupted)


def test_result_round_trip_preserves_landmarks() -> None:
    original = LandmarkResult(
        timestamp_ms=77, points=_points(0), handedness="Left", score=0.93
    )
    restored = decode_result(encode_result(original))
    assert restored.timestamp_ms == 77
    assert restored.handedness == "Left"
    assert restored.score == pytest.approx(0.93)
    assert restored.points is not None
    np.testing.assert_allclose(restored.points, original.points, atol=1e-12)


def test_result_round_trip_without_detection() -> None:
    restored = decode_result(encode_result(LandmarkResult(timestamp_ms=5, points=None)))
    assert restored.points is None
    assert not restored.detected


def test_result_rejects_wrong_landmark_shape() -> None:
    with pytest.raises(ProtocolError):
        decode_result('{"ts":1,"points":[[0.1,0.2],[0.3,0.4]]}')


def test_result_rejects_non_json() -> None:
    with pytest.raises(ProtocolError):
        decode_result(b"not json at all\n")


def test_read_exactly_reassembles_split_chunks() -> None:
    payload = bytes(range(256)) * 40
    assert read_exactly(io.BytesIO(payload), len(payload)) == payload


def test_read_exactly_returns_none_on_early_eof() -> None:
    # 헤더 크기보다 짧은 스트림 = 워커/메인이 먼저 종료한 상황.
    assert read_exactly(io.BytesIO(b"ab"), FRAME_HEADER_SIZE) is None


def test_landmark_deviation_is_zero_for_identical_points() -> None:
    pts = _points(1)
    np.testing.assert_allclose(landmark_deviation(pts, pts), np.zeros(LANDMARK_COUNT), atol=1e-12)


def test_landmark_deviation_matches_known_offset() -> None:
    pts = _points(2)
    shifted = pts + np.array([0.03, 0.04])  # 각 점이 정확히 0.05만큼 이동
    np.testing.assert_allclose(landmark_deviation(pts, shifted), np.full(LANDMARK_COUNT, 0.05))


def test_accumulator_counts_detection_agreement() -> None:
    acc = ComparisonAccumulator()
    a, b = _points(3), _points(4)
    acc.update(a, b)      # 둘 다 검출
    acc.update(a, None)   # A만
    acc.update(None, b)   # B만
    acc.update(None, None)  # 둘 다 미검출

    summary = acc.summary()
    assert (summary.frames, summary.both_detected) == (4, 1)
    assert (summary.only_a, summary.only_b, summary.neither) == (1, 1, 1)
    assert summary.agreement_rate == pytest.approx(0.5)  # both + neither = 2/4


def test_accumulator_deviation_stats_use_known_offsets() -> None:
    acc = ComparisonAccumulator()
    base = _points(5)
    acc.update(base, base + np.array([0.03, 0.04]))  # 편차 0.05
    acc.update(base, base + np.array([0.06, 0.08]))  # 편차 0.10

    summary = acc.summary()
    assert summary.mean_deviation == pytest.approx(0.075)
    assert summary.max_deviation == pytest.approx(0.10)
    assert summary.per_landmark_mean is not None
    np.testing.assert_allclose(summary.per_landmark_mean, np.full(LANDMARK_COUNT, 0.075))


def test_accumulator_summary_is_safe_when_never_detected() -> None:
    acc = ComparisonAccumulator()
    acc.update(None, None)
    summary = acc.summary()
    assert summary.mean_deviation is None
    assert summary.per_landmark_mean is None
    # 수치가 없어도 리포트는 렌더링돼야 한다(n/a 표기).
    assert "n/a" in summary.format_report(label_a="A", label_b="B")


def test_jitter_tracker_ignores_first_frame_and_detection_gaps() -> None:
    tracker = JitterTracker()
    base = _points(6)
    assert tracker.update(base) is None  # 첫 프레임은 비교 대상 없음
    assert tracker.update(base + np.array([0.03, 0.04])) == pytest.approx(0.05)

    assert tracker.update(None) is None  # 추적 끊김
    # 끊긴 뒤 첫 프레임은 이전 손과 이어 붙이지 않는다(가짜 큰 이동량 방지).
    assert tracker.update(_points(7)) is None
    assert tracker.mean == pytest.approx(0.05)

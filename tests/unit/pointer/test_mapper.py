"""CursorControlMapper 단위 테스트 — fake sink로 실제 커서 없이 매핑을 검증한다."""

from __future__ import annotations

import pytest

from jarvis.pointer.mapper import CursorControlMapper, PointerConfig, PointerSample


class FakeSink:
    def __init__(self) -> None:
        self.moves: list[tuple[int, int]] = []

    def scroll(self, ticks: int) -> None:  # pragma: no cover - unused by pointer
        raise AssertionError("pointer must not scroll")

    def tap_key(self, key: object) -> None:  # pragma: no cover - unused by pointer
        raise AssertionError("pointer must not tap keys")

    def move_cursor(self, dx: int, dy: int) -> None:
        self.moves.append((dx, dy))

    def switch_window(self, forward: bool, repeat: int) -> None:  # pragma: no cover
        raise AssertionError("pointer must not switch windows")


def _config(**overrides: object) -> PointerConfig:
    base: dict[str, object] = {
        "screen_width": 1000,
        "screen_height": 1000,
        "smoothing": 0.0,  # 테스트는 평활 없이 결정적으로
        "deadzone_px": 0.0,
    }
    base.update(overrides)
    return PointerConfig(**base)  # type: ignore[arg-type]


def _hand(x: float, y: float) -> PointerSample:
    return PointerSample(x=x, y=y, hand_detected=True)


def test_first_active_frame_only_acquires_reference() -> None:
    sink = FakeSink()
    mapper = CursorControlMapper(sink, _config())

    update = mapper.update(
        gaze_locked_to_laptop=True, hand=_hand(0.5, 0.5), gesture_active=False
    )

    assert update.active is True
    assert update.moved is False
    assert sink.moves == []  # 기준점만 잡고 이동 없음


def test_relative_move_scales_by_screen() -> None:
    sink = FakeSink()
    mapper = CursorControlMapper(sink, _config())

    mapper.update(gaze_locked_to_laptop=True, hand=_hand(0.5, 0.5), gesture_active=False)
    update = mapper.update(
        gaze_locked_to_laptop=True, hand=_hand(0.6, 0.5), gesture_active=False
    )

    # x가 0.1 증가 × 화면폭 1000 = 100px 오른쪽, y는 그대로.
    assert update.moved is True
    assert sink.moves == [(100, 0)]


def test_gaze_unlock_stops_and_drops_reference() -> None:
    sink = FakeSink()
    mapper = CursorControlMapper(sink, _config())

    mapper.update(gaze_locked_to_laptop=True, hand=_hand(0.5, 0.5), gesture_active=False)
    stopped = mapper.update(
        gaze_locked_to_laptop=False, hand=_hand(0.9, 0.9), gesture_active=False
    )
    assert stopped.active is False
    assert stopped.moved is False

    # 재개 첫 프레임은 새 기준점만 잡는다 — 0.5→0.9 점프가 커서로 새지 않는다.
    resumed = mapper.update(
        gaze_locked_to_laptop=True, hand=_hand(0.9, 0.9), gesture_active=False
    )
    assert resumed.active is True
    assert resumed.moved is False
    assert sink.moves == []  # 전 구간에서 단 한 번도 이동 없음(teleport 방지)


def test_gesture_active_yields_cursor() -> None:
    sink = FakeSink()
    mapper = CursorControlMapper(sink, _config())

    mapper.update(gaze_locked_to_laptop=True, hand=_hand(0.5, 0.5), gesture_active=False)
    yielded = mapper.update(
        gaze_locked_to_laptop=True, hand=_hand(0.7, 0.5), gesture_active=True
    )
    assert yielded.active is False
    assert sink.moves == []  # 제스처 중에는 커서를 움직이지 않는다


def test_hand_lost_stops() -> None:
    sink = FakeSink()
    mapper = CursorControlMapper(sink, _config())

    mapper.update(gaze_locked_to_laptop=True, hand=_hand(0.5, 0.5), gesture_active=False)
    lost = mapper.update(
        gaze_locked_to_laptop=True,
        hand=PointerSample(x=0.0, y=0.0, hand_detected=False),
        gesture_active=False,
    )
    assert lost.active is False
    assert sink.moves == []


def test_deadzone_suppresses_jitter() -> None:
    sink = FakeSink()
    mapper = CursorControlMapper(sink, _config(deadzone_px=5.0))

    mapper.update(gaze_locked_to_laptop=True, hand=_hand(0.5, 0.5), gesture_active=False)
    # 0.002 × 1000 = 2px < deadzone 5px → 무시.
    update = mapper.update(
        gaze_locked_to_laptop=True, hand=_hand(0.502, 0.5), gesture_active=False
    )
    assert update.moved is False
    assert sink.moves == []


def test_max_step_clamps_large_jump() -> None:
    sink = FakeSink()
    mapper = CursorControlMapper(sink, _config(max_step_px=50))

    mapper.update(gaze_locked_to_laptop=True, hand=_hand(0.1, 0.1), gesture_active=False)
    # 0.8 × 1000 = 800px 점프 → 50px로 제한.
    update = mapper.update(
        gaze_locked_to_laptop=True, hand=_hand(0.9, 0.1), gesture_active=False
    )
    assert update.moved is True
    assert sink.moves == [(50, 0)]


def test_invert_x_flips_direction() -> None:
    sink = FakeSink()
    mapper = CursorControlMapper(sink, _config(invert_x=True))

    mapper.update(gaze_locked_to_laptop=True, hand=_hand(0.5, 0.5), gesture_active=False)
    update = mapper.update(
        gaze_locked_to_laptop=True, hand=_hand(0.6, 0.5), gesture_active=False
    )
    assert sink.moves == [(-100, 0)]  # invert로 부호 반전
    assert update.moved is True


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        PointerConfig(screen_width=0, screen_height=100)
    with pytest.raises(ValueError):
        PointerConfig(screen_width=100, screen_height=100, smoothing=1.0)

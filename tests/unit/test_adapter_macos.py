"""macOS InputSink 확장을 검증한다.

`MacOSInputSink`는 `Win32InputSink`와 같은 이유로 실제 CGEvent 호출을 자동
테스트하지 않는다("manually verified" — Win32InputSink도 직접 단위 테스트가
없다). 여기서는 (1) 이 모듈이 macOS가 아닌 호스트에서도 import는 실패하지
않는지, (2) `default_input_sink`가 플랫폼별로 올바른 클래스를 고르는지,
(3) 미디어 키 매핑 테이블이 `InputKey`와 어긋나지 않는지만 검증한다.
"""

from __future__ import annotations

import sys

import pytest

from jarvis.runtime_protocol.adapters.windows import InputKey, default_input_sink


def test_macos_module_imports_without_pyobjc_at_module_level() -> None:
    """PyObjC는 메서드 안에서만 import한다 — 모듈 import 자체는 항상 성공해야 한다."""
    from jarvis.runtime_protocol.adapters import macos

    assert hasattr(macos, "MacOSInputSink")


def test_nx_keytype_table_covers_every_media_key() -> None:
    """미디어 키(시스템 정의 이벤트)만 이 표에 있어야 한다.

    F11(SHOW_DESKTOP)은 표준 키보드 이벤트, Mission Control은 앱 직접 실행(open -a),
    CLOSE_TAB(Cmd+W)은 표준 키보드 조합이라 셋 다 이 표에 없는 것이 정상이다 — 미디어
    키 경로로 보내면 OS가 인식하지 않는다.
    """
    from jarvis.runtime_protocol.adapters.macos import _NX_KEYTYPE

    assert set(_NX_KEYTYPE) == set(InputKey) - {
        InputKey.SHOW_DESKTOP,
        InputKey.MISSION_CONTROL,
        InputKey.TASK_VIEW,
        InputKey.CLOSE_TAB,
    }


def test_nx_keytype_values_are_unique() -> None:
    from jarvis.runtime_protocol.adapters.macos import _NX_KEYTYPE

    assert len(set(_NX_KEYTYPE.values())) == len(_NX_KEYTYPE)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS에서만 의미 있는 분기")
def test_default_input_sink_is_macos_sink_on_darwin() -> None:
    from jarvis.runtime_protocol.adapters.macos import MacOSInputSink

    assert isinstance(default_input_sink(), MacOSInputSink)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows에서만 의미 있는 분기")
def test_default_input_sink_is_win32_sink_on_windows() -> None:
    from jarvis.runtime_protocol.adapters.windows import Win32InputSink

    assert isinstance(default_input_sink(), Win32InputSink)


def test_default_input_sink_rejects_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="no InputSink implementation"):
        default_input_sink()


def test_dock_transition_reveals_at_bottom_only_once() -> None:
    """하단 진입 시 한 번만 '드러내기'(False)를 내고, 계속 하단이면 None(전환 없음).

    매 프레임 osascript를 띄우지 않으려면 전환 시점에만 값을 내야 한다.
    """
    from jarvis.runtime_protocol.adapters.macos import dock_transition

    bottom = 900.0
    assert dock_transition(899.0, bottom, 2.0, revealed=False) is False
    assert dock_transition(899.0, bottom, 2.0, revealed=True) is None


def test_dock_transition_hides_when_leaving_bottom() -> None:
    """하단을 벗어나면 한 번 '숨기기'(True)를 내고, 계속 위쪽이면 None."""
    from jarvis.runtime_protocol.adapters.macos import dock_transition

    bottom = 900.0
    assert dock_transition(500.0, bottom, 2.0, revealed=True) is True
    assert dock_transition(500.0, bottom, 2.0, revealed=False) is None


def test_dock_transition_edge_threshold() -> None:
    """edge_px 이내면 '하단'으로 본다 — 정확히 경계에서의 판정."""
    from jarvis.runtime_protocol.adapters.macos import dock_transition

    bottom = 900.0
    assert dock_transition(898.0, bottom, 2.0, revealed=False) is False
    assert dock_transition(897.9, bottom, 2.0, revealed=False) is None

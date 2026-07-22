"""고개 끄덕임(nod) 검출 — 특정 target으로 "돌아올 때" 확인 게이트에 쓴다.

배경(documents/gaze.md, decisions.md 2026-07-22): 데모에서 카메라를 노트북
근처에 두면 노트북 방향이 곧 "가만히 있을 때의 기본 시선 방향"이 된다. 다른
물체(전구)를 보다가 안 보다가 그냥 정면을 스쳐도 계속 노트북으로 오확정될
위험이 있는데, 전구는 각도가 확실히 갈라져 있어 그런 오탐이 없다. 그래서
"다른 target에서 확정된 뒤 이 target으로 돌아올 때"만 명시적 끄덕임을
요구하는 게이트를 Gaze Lock에 추가한다(lock.py).

이 모듈은 `head_pitch_deg` + `timestamp_ms`만 다루는 순수 클래스라
카메라 없이 단위 테스트한다(blink.py와 동일한 패턴).
"""

from __future__ import annotations

from jarvis.gaze.config import GazeConfig


class NodDetector:
    """턱을 숙였다가(dip) 다시 드는(recovery) 동작 한 번을 검출한다.

    baseline은 평소 쉬는 자세의 head pitch를 느리게 추적한다(눈 뜸 baseline과
    같은 decay 방식). dip 중에는 baseline을 갱신하지 않아 끄덕임 자체가 새
    baseline으로 흡수되지 않는다. dip이 `nod_max_duration_ms`를 넘겨 지속되면
    끄덕임이 아니라 지속적인 자세 변화로 보고 포기하고, 그 시점의 pitch로
    baseline을 재설정한다.
    """

    def __init__(self, config: GazeConfig = GazeConfig()) -> None:
        self._config = config
        self._baseline_pitch_deg: float | None = None
        self._dip_started_at_ms: int | None = None
        self._dip_extremum_deg: float | None = None

    def reset(self) -> None:
        self._baseline_pitch_deg = None
        self._dip_started_at_ms = None
        self._dip_extremum_deg = None

    @property
    def baseline_pitch_deg(self) -> float | None:
        """진단용 — 현재 추적 중인 평소 pitch 기준값."""
        return self._baseline_pitch_deg

    def update(self, head_pitch_deg: float, timestamp_ms: int) -> bool:
        """한 프레임을 반영한다. 끄덕임이 이 프레임에서 완료됐으면 True.

        얼굴을 못 찾은 프레임은 호출하지 말 것 — 정지된/기본값 pitch가
        baseline을 오염시킨다.
        """
        config = self._config
        if self._baseline_pitch_deg is None:
            self._baseline_pitch_deg = head_pitch_deg
            return False

        if self._dip_started_at_ms is None:
            delta = head_pitch_deg - self._baseline_pitch_deg
            if delta <= -config.nod_dip_threshold_deg:
                self._dip_started_at_ms = timestamp_ms
                self._dip_extremum_deg = head_pitch_deg
                return False
            decay = config.nod_baseline_decay
            self._baseline_pitch_deg += (head_pitch_deg - self._baseline_pitch_deg) * decay
            return False

        assert self._dip_extremum_deg is not None
        self._dip_extremum_deg = min(self._dip_extremum_deg, head_pitch_deg)
        elapsed_ms = timestamp_ms - self._dip_started_at_ms
        if elapsed_ms > config.nod_max_duration_ms:
            # 빠른 끄덕임이 아니라 지속적인 고개 숙임 — 새 평소 자세로 재기준.
            self._dip_started_at_ms = None
            self._dip_extremum_deg = None
            self._baseline_pitch_deg = head_pitch_deg
            return False

        recovered = head_pitch_deg - self._dip_extremum_deg >= config.nod_recovery_deg
        if not recovered:
            return False
        self._dip_started_at_ms = None
        self._dip_extremum_deg = None
        self._baseline_pitch_deg = head_pitch_deg
        return True

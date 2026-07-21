"""웹캠 제스처 클립 녹화의 순수 코어 — CLI(`record_webcam_clips`)와 모니터 GUI가 공유한다.

녹화 로직(프레임 누적 → 미검출 게이트 → fps 정규화 → 저장)을 Qt·mediapipe·cv2에
의존하지 않는 한 클래스로 모은다(`gaze_samples.GazeSampleStore`와 같은 취지). 이렇게
하면 CLI와 GUI가 **정확히 같은** 녹화 규칙을 쓰고, 카메라 없이 합성 `HandObservation`
으로 단위 테스트할 수 있다.

한 클립은 `start()` → 프레임마다 `add(observation)` → `stop()` 로 완결된다. 저장은
`observations_to_cached_clip` + `save_clip`(atomic)을 재사용하며, `target_fps`가 있으면
저장 직전 `resample_clip_to_fps`로 fps를 정규화한다(웹캠 30fps → pretrain 12fps 정합).

저장 경로·명명은 CLI와 동일하다: `<cache_dir>/webcam/<person_id>/<person>-<gesture>-<NNNN>.npz`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jarvis.gesture_fusion.landmarks import HandObservation
from training.augment import resample_clip_to_fps
from training.data.clip_cache import observations_to_cached_clip, save_clip


@dataclass(frozen=True, slots=True)
class RecordResult:
    """한 클립 녹화 종료 결과 — 저장됐는지, 사람이 읽을 사유, 클립 메타."""

    saved: bool
    detail: str
    clip_id: str | None = None
    frame_count: int = 0


class ClipRecorder:
    """프레임 스트림을 받아 하나의 제스처 클립으로 저장하는 순수 녹화기.

    카메라·모델을 직접 몰라도 되도록 `HandObservation`만 받는다(정규화·평활화 전 원본).
    저장은 **평활화 전** 관측값을 쓴다 — 캐시는 raw이고 학습이 읽을 때 평활화·feature를
    다시 계산하기 때문이다(`clip_cache` docstring).
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        max_missing_frame_fraction: float,
        target_fps: float | None = None,
    ) -> None:
        if not 0.0 <= max_missing_frame_fraction <= 1.0:
            raise ValueError("max_missing_frame_fraction must be within [0, 1]")
        if target_fps is not None and target_fps <= 0.0:
            raise ValueError("target_fps must be positive when set")
        self._cache_dir = cache_dir
        self._max_missing_frame_fraction = max_missing_frame_fraction
        self._target_fps = target_fps
        self._recording = False
        self._person_id = ""
        self._gesture_label = ""
        self._buffer: list[HandObservation] = []
        self._missing_count = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def person_id(self) -> str:
        return self._person_id

    @property
    def gesture_label(self) -> str:
        return self._gesture_label

    @property
    def frame_count(self) -> int:
        return len(self._buffer)

    def start(self, person_id: str, gesture_label: str) -> None:
        """새 클립 녹화를 시작한다. 진행 중이던 클립은 저장 없이 버려진다."""
        person = person_id.strip()
        gesture = gesture_label.strip()
        if not person:
            raise ValueError("person_id must not be empty")
        if not gesture:
            raise ValueError("gesture_label must not be empty")
        self._person_id = person
        self._gesture_label = gesture
        self._buffer = []
        self._missing_count = 0
        self._recording = True

    def add(self, observation: HandObservation) -> None:
        """녹화 중이면 프레임 하나를 누적한다(녹화 중이 아니면 무시).

        미검출 프레임(`hand_detected=False`)도 버린 게 아니라 담는다 — 클립 내 위치를
        보존해야 학습의 IGNORE_INDEX 마스킹이 맞고, 미검출 비율 게이트를 stop 시점에
        한 번 적용할 수 있다(CLI와 동일).
        """
        if not self._recording:
            return
        self._buffer.append(observation)
        if not observation.hand_detected:
            self._missing_count += 1

    def stop(self) -> RecordResult:
        """녹화를 끝내고 게이트를 적용해 저장하거나 폐기한다."""
        if not self._recording:
            return RecordResult(saved=False, detail="녹화 중이 아님")
        buffer = self._buffer
        missing = self._missing_count
        person, gesture = self._person_id, self._gesture_label
        # 상태를 먼저 비운다 — 저장이 실패하든 성공하든 다음 녹화가 깨끗이 시작되도록.
        self._recording = False
        self._buffer = []
        self._missing_count = 0

        if not buffer:
            return RecordResult(saved=False, detail="빈 클립 — 저장하지 않음")
        missing_fraction = missing / len(buffer)
        if missing_fraction > self._max_missing_frame_fraction:
            return RecordResult(
                saved=False,
                detail=(
                    f"미검출 프레임 과다로 폐기 ({missing}/{len(buffer)}프레임, "
                    f"허용 {self._max_missing_frame_fraction:.0%})"
                ),
                frame_count=len(buffer),
            )

        out_dir = self._cache_dir / "webcam" / person
        counter = len(list(out_dir.glob("*.npz"))) if out_dir.exists() else 0
        clip_id = f"{person}-{gesture}-{counter:04d}"
        clip = observations_to_cached_clip(buffer, gesture, clip_id)
        if self._target_fps is not None:
            clip = resample_clip_to_fps(clip, self._target_fps)
        save_clip(out_dir / f"{clip_id}.npz", clip)
        return RecordResult(
            saved=True,
            detail=f"저장됨: {clip_id} ({len(clip)}프레임)",
            clip_id=clip_id,
            frame_count=len(clip),
        )

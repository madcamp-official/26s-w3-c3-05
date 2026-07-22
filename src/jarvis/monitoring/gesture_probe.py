"""Live gesture pipeline probe for the desktop monitor.

`GazeProbe`(`gaze_probe.py`)와 같은 역할의 제스처판이다: 카메라 BGR 프레임을 받아
손 랜드마크 검출 → feature → Causal TCN → spotting까지 돌려 프레임별
`GestureEstimate`를 낸다. GazeProbe와 동일한 정직성 원칙을 따른다 — mediapipe
(`vision` extra)·torch(`ml` extra)·모델 파일이 없으면 `available=False`로 두고
`status_text`에 이유를 담을 뿐, 가짜 검출을 지어내지 않는다.

모듈 경계: 무거운 파이프라인 조립은 전부 `jarvis.gesture_fusion`의 공개 API로만
하고, 그 내부 구현을 직접 파고들지 않는다. 색상 변환(BGR→RGB)·`Frame` 언팩 같은
배선 계층 책임은 GazeProbe와 똑같이 여기(앱 계층)서 처리한다.

**미학습 주의**: 기본 제스처 모델은 무작위 초기화 가중치다(`ModelMetadata.
trained=False`). 손 랜드마크 검출·오버레이는 실제로 동작하지만, gesture label은
학습된 가중치를 주입하기 전까지 신뢰할 수 없다.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from jarvis.contracts.messages import GestureEstimate, GesturePhase
from jarvis.gesture_fusion import (
    DEFAULT_GESTURE_CONFIG,
    GestureConfig,
    HandFeatureExtractor,
    feature_dimension,
)
from jarvis.gesture_fusion.model_protocol import (
    EXPECTED_INPUT_FPS,
    INFERENCE_WINDOW_FRAMES,
    FrameRateLimiter,
    GestureModel,
    ModelConfig,
    ModelMetadata,
    SlidingFeatureWindow,
)
from jarvis.gesture_fusion.spotting import DEFAULT_SPOTTER_CONFIG, GestureSpotter, SpotterConfig
from jarvis.monitoring.gesture_source import RecognizedGesture

if TYPE_CHECKING:
    from jarvis.gesture_fusion.landmarks import HandObservation

FloatArray = npt.NDArray[np.float64]


class GestureSnapshot:
    """한 프레임의 제스처 파이프라인 결과 — 오버레이·사이드바 표시용."""

    __slots__ = ("timestamp_ms", "frame_id", "hand_detected", "estimate", "landmarks", "latency_ms")

    def __init__(
        self,
        *,
        timestamp_ms: int,
        frame_id: int,
        hand_detected: bool,
        estimate: GestureEstimate,
        landmarks: FloatArray,
        latency_ms: float,
    ) -> None:
        self.timestamp_ms = timestamp_ms
        self.frame_id = frame_id
        self.hand_detected = hand_detected
        self.estimate = estimate
        self.landmarks = landmarks
        self.latency_ms = latency_ms


class GestureProbe:
    """카메라 BGR 프레임 → `GestureSnapshot`. 파이프라인 상태를 프레임 간 보존한다.

    손 랜드마커(mediapipe)는 GazeProbe와 같이 지연 생성한다. 제스처 모델은 교체
    가능하게 주입받으며(기본: 미학습 Causal TCN), 학습된 가중치가 준비되면 그대로
    갈아끼운다 — `GestureModel` Protocol만 바라보므로 이 클래스는 안 바뀐다.
    """

    def __init__(
        self,
        *,
        model_asset_path: Path | None,
        gesture_config: GestureConfig | None = None,
        spotter_config: SpotterConfig = DEFAULT_SPOTTER_CONFIG,
        model: GestureModel | None = None,
        target_fps: float | None = EXPECTED_INPUT_FPS,
        window_size: int | None = INFERENCE_WINDOW_FRAMES,
    ) -> None:
        self._requested_window_size = window_size
        self._config = gesture_config or DEFAULT_GESTURE_CONFIG
        self._model_asset_path = model_asset_path
        self._injected_model = model
        self._extractor = HandFeatureExtractor(self._config)
        self._spotter = GestureSpotter(spotter_config)
        self._landmarker: object | None = None
        self._model: GestureModel | None = None
        self._window: SlidingFeatureWindow | None = None
        self._available = False
        self._status_text = "gesture 프로브 미시작"
        # 인식 feed를 학습 cadence(EXPECTED_INPUT_FPS)로 솎는다 — 웹캠 30fps를 그대로
        # 넣으면 12fps로 학습된 모델과 velocity·receptive field가 어긋난다. None이면
        # 솎지 않는다(모든 프레임 처리 — 순수 `_advance` 테스트는 이 경로를 안 탄다).
        self._rate_limiter = FrameRateLimiter(target_fps) if target_fps else None
        self._last_snapshot: GestureSnapshot | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def status_text(self) -> str:
        return self._status_text

    def start(self) -> bool:
        """손 랜드마커 + 제스처 모델을 만든다. 성공 시 True.

        mediapipe 미설치·모델 파일 부재·torch 미설치는 전부 정직한 `status_text`를
        남기고 `available=False`로 둔다(GazeProbe와 동일 — 성공을 가장하지 않는다).
        """
        if self._model_asset_path is None or not self._model_asset_path.is_file():
            self._status_text = "hand_landmarker.task 모델 없음 (models/README.md 참고)"
            return False
        try:
            from jarvis.gesture_fusion.mediapipe_hands import MediaPipeHandLandmarker
        except ImportError:
            self._status_text = 'mediapipe 미설치 — pip install -e ".[vision]"'
            return False
        try:
            self._landmarker = MediaPipeHandLandmarker(self._model_asset_path, self._config)
        except Exception as exc:  # noqa: BLE001 - surface any init failure honestly
            self._status_text = f"gesture 랜드마커 초기화 실패: {exc}"
            return False

        model = self._injected_model
        if model is None:
            try:
                model = self._build_default_model()
            except ImportError:
                self._status_text = 'torch 미설치 — pip install -e ".[ml]"'
                return False
        self._model = model
        self._window = SlidingFeatureWindow(
            window_size=self._resolve_window_size(model),
            feature_dim=feature_dimension(self._config),
        )
        self._available = True
        trained = "학습됨" if getattr(model, "metadata", None) and model.metadata.trained else "미학습(랜덤 label)"  # type: ignore[attr-defined]
        self._status_text = f"LIVE · {self._model_asset_path.name} · {trained}"
        return True

    def _resolve_window_size(self, model: GestureModel) -> int:
        """모델에 넣을 윈도우 길이. 기본은 `INFERENCE_WINDOW_FRAMES`(receptive field보다 짧음).

        receptive field보다 긴 값은 의미가 없다 — `_pad_to_window`가 어차피 뒤에서
        잘라내므로, 상한을 여기서 명시해 "설정한 값"과 "실제 쓰이는 값"이 갈리지 않게
        한다. None이면 모델이 요구하는 최소 길이(=receptive field)를 그대로 쓴다
        (짧은 윈도우를 끄고 종전 동작으로 되돌리는 경로).
        """
        if self._requested_window_size is None:
            return model.window_size
        if self._requested_window_size < 1:
            raise ValueError("window_size must be at least 1")
        return min(self._requested_window_size, model.window_size)

    def _build_default_model(self) -> GestureModel:
        from jarvis.gesture_fusion.model import CausalTCNGestureModel

        return CausalTCNGestureModel(ModelConfig(feature_dim=feature_dimension(self._config)))

    def activate_headless(self) -> bool:
        """landmarker 없이 모델·윈도우만 준비한다 — 관측값을 외부에서 주입받는 모드.

        HandProbe가 이미 프레임마다 손을 검출하므로, **두 번째 landmarker를 돌리지 않고**
        그 관측값(`HandSnapshot.observation`)을 재사용한다. `process_bgr`(자체 landmarker
        구동) 대신 `advance(observation)`로 구동한다. 주입된 model이 없으면 False.
        """
        model = self._injected_model
        if model is None:
            self._status_text = "gesture 모델 미주입"
            return False
        self._model = model
        self._window = SlidingFeatureWindow(
            window_size=self._resolve_window_size(model),
            feature_dim=feature_dimension(self._config),
        )
        self._available = True
        meta = getattr(model, "metadata", None)
        if meta is not None and meta.trained:
            self._status_text = (
                f"LIVE · 인식 활성 (관측값 재사용 · {int(EXPECTED_INPUT_FPS)}fps) · {meta.version}"
            )
        else:
            self._status_text = "미학습 모델 — 인식 비활성"
        return True

    def advance(self, observation: HandObservation) -> GestureSnapshot | None:
        """외부 관측값(HandProbe)으로 파이프라인을 한 프레임 진행한다.

        학습 cadence(`EXPECTED_INPUT_FPS`)로 솎아 넣는다. 단 손실 프레임은 **항상 처리**해
        파이프라인을 즉시 리셋한다 — 솎으면 velocity가 손실 구간을 가로질러 튀고, 리미터도
        리셋해 복귀 첫 프레임을 바로 채택한다. **솎여 처리하지 않은 프레임은 None**을
        반환한다(호출자가 12fps tick만 골라내기 쉽도록 — process_bgr의 표시 재사용과 다름).
        """
        if self._model is None or self._window is None:
            return None
        if not observation.hand_detected:
            if self._rate_limiter is not None:
                self._rate_limiter.reset()
        elif self._rate_limiter is not None and not self._rate_limiter.should_accept(
            observation.timestamp_ms
        ):
            return None
        snapshot = self._advance(observation, time.monotonic())
        self._last_snapshot = snapshot
        return snapshot

    def process_bgr(
        self, bgr_frame: npt.NDArray[np.uint8], timestamp_ms: int, frame_id: int
    ) -> GestureSnapshot | None:
        """BGR 프레임 하나를 파이프라인에 흘려 `GestureSnapshot`을 만든다.

        프로브가 준비 안 됐으면 `None`. BGR→RGB 변환은 여기(배선 계층)서 한다.
        """
        if self._landmarker is None or self._model is None or self._window is None:
            return None
        # 학습 cadence로 프레임을 솎는다 — 채택 안 된 프레임은 landmarker(비용 큼)조차
        # 돌리지 않고 직전 스냅샷을 그대로 돌려준다. MediaPipe VIDEO 모드는 더 큰 프레임
        # 간격(≈83ms)을 그냥 낮은 fps로 처리하므로 문제없다.
        if self._rate_limiter is not None and not self._rate_limiter.should_accept(timestamp_ms):
            return self._last_snapshot
        import cv2

        from jarvis.gesture_fusion.mediapipe_hands import MediaPipeHandLandmarker

        assert isinstance(self._landmarker, MediaPipeHandLandmarker)
        started = time.monotonic()
        rgb = cast("npt.NDArray[np.uint8]", cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        observation = self._landmarker.process(rgb, timestamp_ms, frame_id)
        snapshot = self._advance(observation, started)
        self._last_snapshot = snapshot
        return snapshot

    def _advance(self, observation: HandObservation, started: float) -> GestureSnapshot:
        """순수 파이프라인 구간(feature→model→spotting) — mediapipe 무관, 테스트 가능."""
        assert self._model is not None and self._window is not None
        features = self._extractor.push(observation)
        if not observation.hand_detected:
            window = self._window.push(None)
            estimate = self._spotter.push(None, observation.timestamp_ms, observation.frame_id)
        else:
            window = self._window.push(features.vector)
            prediction = self._model.predict(window)
            estimate = self._spotter.push(
                prediction, observation.timestamp_ms, observation.frame_id
            )
        latency_ms = (time.monotonic() - started) * 1000.0
        return GestureSnapshot(
            timestamp_ms=observation.timestamp_ms,
            frame_id=observation.frame_id,
            hand_detected=observation.hand_detected,
            estimate=estimate,
            landmarks=observation.landmarks,
            latency_ms=latency_ms,
        )


class ProbeGestureSource:
    """`GestureProbe`를 사이드바용 `GestureSource`로 감싼다.

    `process_bgr`이 낸 스냅샷 중 phase가 바뀌는(=새 이벤트) 것만 골라 버퍼에 쌓고,
    `poll()`이 그 사이 쌓인 것들을 비우며 돌려준다. `NullGestureSource`를 대체한다.
    """

    def __init__(self, probe: GestureProbe, max_buffer: int = 64) -> None:
        self._probe = probe
        self._buffer: deque[RecognizedGesture] = deque(maxlen=max_buffer)
        self._last_phase: GesturePhase | None = None

    @property
    def available(self) -> bool:
        return self._probe.available

    @property
    def status_text(self) -> str:
        return self._probe.status_text

    def submit(self, snapshot: GestureSnapshot | None) -> None:
        """카메라 스레드가 낸 스냅샷을 반영한다. phase 전이 프레임만 사이드바에 남긴다."""
        if snapshot is None:
            return
        phase = snapshot.estimate.phase
        if phase != self._last_phase and phase != GesturePhase.IDLE:
            self._buffer.append(
                RecognizedGesture(
                    timestamp_ms=snapshot.estimate.timestamp_ms,
                    gesture=snapshot.estimate.gesture,
                    confidence=snapshot.estimate.gesture_confidence,
                    phase=str(phase),
                )
            )
        self._last_phase = phase

    def poll(self) -> list[RecognizedGesture]:
        drained = list(self._buffer)
        self._buffer.clear()
        return drained


# 우선순위: 파인튜닝 체크포인트 > Jester 사전학습. 앞의 것이 있으면 그걸 쓴다.
_CHECKPOINT_PREFERENCE: tuple[str, ...] = (
    "gesture_tcn_finetuned.pt",
    "gesture_tcn_jester.pt",
)


def _checkpoint_metadata(checkpoint: Path) -> ModelMetadata | None:
    """사이드카(`<name>.metadata.json`)를 읽어 `ModelMetadata`를 구성 — 없거나 손상이면 None.

    torch를 로드하지 않는 가벼운 검사다(체크포인트 목록 만들기·trained 판정용).
    """
    sidecar = checkpoint.with_name(checkpoint.name + ".metadata.json")
    if not checkpoint.is_file() or not sidecar.is_file():
        return None
    try:
        meta_dict = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 손상 사이드카는 없는 것으로 취급
        return None
    return ModelMetadata(
        version=str(meta_dict.get("version", "")),
        trained=bool(meta_dict.get("trained", False)),
        training_data_source=str(meta_dict.get("training_data_source", "")),
        evaluation_notes=str(meta_dict.get("evaluation_notes", "")),
    )


def _load_checkpoint(checkpoint: Path, config: GestureConfig) -> GestureModel | None:
    """체크포인트 하나를 로드 — `metadata.trained=True`일 때만 GestureModel, 아니면 None.

    다음은 전부 조용히 None: torch 미설치(`ml` extra 부재), `.pt`·사이드카 부재,
    trained=False, 라벨 집합 불일치(head shape)·손상 파일. 성공을 지어내지 않는다.
    """
    metadata = _checkpoint_metadata(checkpoint)
    if metadata is None or not metadata.trained:
        return None
    try:
        from jarvis.gesture_fusion.model import CausalTCNGestureModel
    except ImportError:
        return None  # torch 없음 — 정직하게 off
    try:
        model = CausalTCNGestureModel(ModelConfig(feature_dim=feature_dimension(config)))
        model.load_weights(str(checkpoint), metadata)
        return model
    except Exception:  # noqa: BLE001 - 라벨 불일치·손상 파일은 off로 떨어뜨린다
        return None


def available_gesture_checkpoints(models_dir: Path) -> list[Path]:
    """`models_dir`에서 로드 가능한(사이드카 `trained=True`) gesture 체크포인트 경로 — 선호 순서.

    실제 torch 로드는 하지 않고 사이드카만 확인한다(가중치 선택 UI 목록 채우기용).
    """
    result: list[Path] = []
    for name in _CHECKPOINT_PREFERENCE:
        checkpoint = models_dir / name
        metadata = _checkpoint_metadata(checkpoint)
        if metadata is not None and metadata.trained:
            result.append(checkpoint)
    return result


def load_gesture_model_from(
    checkpoint: Path, gesture_config: GestureConfig | None = None
) -> GestureModel | None:
    """특정 체크포인트를 로드한다(`trained=True`일 때만) — `load_trained_gesture_model`과 같은 게이트."""
    return _load_checkpoint(checkpoint, gesture_config or DEFAULT_GESTURE_CONFIG)


def load_trained_gesture_model(
    models_dir: Path, gesture_config: GestureConfig | None = None
) -> GestureModel | None:
    """models_dir에서 가장 좋은 **학습된** 체크포인트를 로드한다 — 없으면 None.

    우선순위는 finetuned > jester(사전학습). 각 `.pt`는 사이드카 `<name>.metadata.json`을
    함께 읽어 `ModelMetadata`를 구성하고, **`metadata.trained=True`일 때만** 인정한다 —
    미학습·무작위 가중치를 인식 결과로 내보내지 않는다(성공을 지어내지 않는다).

    데모 노트북에서 VM의 `.pt`를 `models/`로 복사하면 다음 실행 때 자동으로 잡힌다.
    """
    config = gesture_config or DEFAULT_GESTURE_CONFIG
    for name in _CHECKPOINT_PREFERENCE:
        model = _load_checkpoint(models_dir / name, config)
        if model is not None:
            return model
    return None

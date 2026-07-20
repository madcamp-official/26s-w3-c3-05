"""Small per-user gaze calibration model.

The baseline gaze vector is intentionally simple: head pose plus iris offset.
For off-monitor targets that simplicity is the main failure mode: the same
object can produce very different raw yaw/pitch values when the user moves
their head.  This module provides a tiny ridge-regression corrector trained
from look-to-register samples.

It is deliberately lightweight and dependency-free.  If no calibration samples
exist, callers simply keep using the baseline gaze direction.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from jarvis.gaze.direction import direction_to_yaw_pitch, yaw_pitch_to_direction
from jarvis.gaze.features import FaceObservation, GazeVector, Vector3

_FEATURE_DIM = 13


@dataclass(frozen=True, slots=True)
class GazeCalibrationSample:
    """One supervised calibration row: raw observation -> known target direction."""

    features: tuple[float, ...]
    target_yaw: float
    target_pitch: float

    def __post_init__(self) -> None:
        if len(self.features) != _FEATURE_DIM:
            raise ValueError(f"features must have {_FEATURE_DIM} values")
        values = (*self.features, self.target_yaw, self.target_pitch)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("calibration sample values must be finite")


def observation_features(observation: FaceObservation, raw_gaze: GazeVector) -> tuple[float, ...]:
    """Build the regression input vector from a frame and its baseline gaze."""
    raw_yaw, raw_pitch = direction_to_yaw_pitch(raw_gaze.direction)
    left_eye = observation.left_eye_center_normalized
    right_eye = observation.right_eye_center_normalized
    if left_eye is not None and right_eye is not None:
        face_center_x = (left_eye[0] + right_eye[0]) * 0.5
        face_center_y = (left_eye[1] + right_eye[1]) * 0.5
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        face_scale = math.hypot(eye_dx, eye_dy)
    else:
        face_center_x = 0.5
        face_center_y = 0.5
        face_scale = 0.0

    return (
        1.0,
        raw_yaw,
        raw_pitch,
        observation.left_iris_relative[0],
        observation.left_iris_relative[1],
        observation.right_iris_relative[0],
        observation.right_iris_relative[1],
        observation.head_yaw_deg,
        observation.head_pitch_deg,
        observation.head_roll_deg,
        face_center_x,
        face_center_y,
        face_scale,
    )


class GazeCalibrationModel:
    """Ridge-regression yaw/pitch corrector trained from registration samples."""

    def __init__(
        self,
        *,
        coefficients: npt.NDArray[np.float64] | None = None,
        sample_count: int = 0,
        target_count: int = 0,
        regularization: float = 1.0,
    ) -> None:
        self._coefficients = coefficients
        self._sample_count = sample_count
        self._target_count = target_count
        self._regularization = regularization

    @property
    def fitted(self) -> bool:
        return self._coefficients is not None and self._target_count >= 2

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def target_count(self) -> int:
        return self._target_count

    @classmethod
    def fit(
        cls,
        samples: Sequence[GazeCalibrationSample],
        *,
        regularization: float = 1.0,
    ) -> "GazeCalibrationModel":
        if not samples:
            return cls(regularization=regularization)
        target_count = len(
            {
                (round(sample.target_yaw, 3), round(sample.target_pitch, 3))
                for sample in samples
            }
        )
        x = np.asarray([sample.features for sample in samples], dtype=np.float64)
        y = np.asarray(
            [[sample.target_yaw, sample.target_pitch] for sample in samples], dtype=np.float64
        )
        penalty = np.eye(x.shape[1], dtype=np.float64) * regularization
        penalty[0, 0] = 0.0  # Do not regularize the intercept.
        coefficients, *_ = np.linalg.lstsq(x.T @ x + penalty, x.T @ y, rcond=None)
        return cls(
            coefficients=coefficients,
            sample_count=len(samples),
            target_count=target_count,
            regularization=regularization,
        )

    def predict_yaw_pitch(self, features: Sequence[float]) -> tuple[float, float] | None:
        if not self.fitted or self._coefficients is None:
            return None
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (_FEATURE_DIM,) or not np.all(np.isfinite(x)):
            return None
        yaw, pitch = x @ self._coefficients
        if not math.isfinite(float(yaw)) or not math.isfinite(float(pitch)):
            return None
        return float(yaw), float(pitch)

    def correct(self, observation: FaceObservation, raw_gaze: GazeVector) -> GazeVector:
        prediction = self.predict_yaw_pitch(observation_features(observation, raw_gaze))
        if prediction is None:
            return raw_gaze
        yaw, pitch = prediction
        direction: Vector3 = yaw_pitch_to_direction(yaw, pitch)
        return GazeVector(
            direction=direction,
            confidence=raw_gaze.confidence,
            timestamp_ms=raw_gaze.timestamp_ms,
            frame_id=raw_gaze.frame_id,
            origin=raw_gaze.origin,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count": self._sample_count,
            "target_count": self._target_count,
            "regularization": self._regularization,
            "coefficients": (
                self._coefficients.tolist() if self._coefficients is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GazeCalibrationModel":
        coefficients_payload = payload.get("coefficients")
        coefficients = (
            np.asarray(coefficients_payload, dtype=np.float64)
            if coefficients_payload is not None
            else None
        )
        if coefficients is not None and coefficients.shape != (_FEATURE_DIM, 2):
            raise ValueError("invalid gaze calibration coefficient shape")
        raw_sample_count = payload.get("sample_count", 0)
        raw_target_count = payload.get("target_count", 0)
        raw_regularization = payload.get("regularization", 1.0)
        sample_count = int(raw_sample_count) if isinstance(raw_sample_count, (int, float)) else 0
        target_count = int(raw_target_count) if isinstance(raw_target_count, (int, float)) else 0
        regularization = (
            float(raw_regularization) if isinstance(raw_regularization, (int, float)) else 1.0
        )
        return cls(
            coefficients=coefficients,
            sample_count=sample_count,
            target_count=target_count,
            regularization=regularization,
        )


class GazeCalibrationStore:
    """Persistent append-only sample store plus fitted model."""

    def __init__(self, path: Path, *, regularization: float = 1.0) -> None:
        self._path = path
        self._regularization = regularization
        self._samples = self._load_samples()
        self._model = GazeCalibrationModel.fit(
            self._samples, regularization=self._regularization
        )

    @property
    def samples(self) -> tuple[GazeCalibrationSample, ...]:
        return tuple(self._samples)

    @property
    def model(self) -> GazeCalibrationModel:
        return self._model

    def add_samples(self, samples: Iterable[GazeCalibrationSample]) -> GazeCalibrationModel:
        self._samples.extend(samples)
        self._model = GazeCalibrationModel.fit(
            self._samples, regularization=self._regularization
        )
        self._save()
        return self._model

    def _load_samples(self) -> list[GazeCalibrationSample]:
        if not self._path.is_file():
            return []
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid gaze calibration model file: {self._path}")
        raw_samples = payload.get("samples", [])
        if not isinstance(raw_samples, list):
            raise ValueError(f"invalid gaze calibration samples: {self._path}")
        samples: list[GazeCalibrationSample] = []
        for item in raw_samples:
            if not isinstance(item, dict):
                continue
            features = item.get("features", [])
            target = item.get("target", {})
            if not isinstance(features, list) or not isinstance(target, dict):
                continue
            samples.append(
                GazeCalibrationSample(
                    features=tuple(float(value) for value in features),
                    target_yaw=float(target.get("yaw", 0.0)),
                    target_pitch=float(target.get("pitch", 0.0)),
                )
            )
        return samples

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model.to_dict(),
            "samples": [
                {
                    "features": list(sample.features),
                    "target": {"yaw": sample.target_yaw, "pitch": sample.target_pitch},
                }
                for sample in self._samples
            ],
        }
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self._path)

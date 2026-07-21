"""Small residual MLP that corrects the geometric gaze vector per user.

The model predicts ``delta_yaw``/``delta_pitch`` rather than an absolute ray.
That keeps the existing geometric composition as a safe baseline and lets a
small amount of personal registration data learn pose-dependent corrections.
Training happens only after registration; runtime is three NumPy matrix
multiplications and does not require torch.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from jarvis.gaze.direction import yaw_pitch_to_direction
from jarvis.gaze.features import FaceObservation, GazeVector, Vector3

_FEATURE_DIM = 13
_INPUT_DIM = _FEATURE_DIM - 1  # the Ridge feature's explicit bias is not needed
_HIDDEN_1 = 24
_HIDDEN_2 = 12
_MIN_TARGETS = 3
_MIN_SAMPLES = 60


class CalibrationSampleLike(Protocol):
    @property
    def features(self) -> tuple[float, ...]: ...

    @property
    def target_yaw(self) -> float: ...

    @property
    def target_pitch(self) -> float: ...


class GazeMLPCalibrationModel:
    """Two-hidden-layer residual yaw/pitch regressor."""

    kind = "mlp"

    def __init__(
        self,
        *,
        input_mean: npt.NDArray[np.float64] | None = None,
        input_scale: npt.NDArray[np.float64] | None = None,
        output_mean: npt.NDArray[np.float64] | None = None,
        output_scale: npt.NDArray[np.float64] | None = None,
        weights: tuple[npt.NDArray[np.float64], ...] = (),
        biases: tuple[npt.NDArray[np.float64], ...] = (),
        sample_count: int = 0,
        target_count: int = 0,
        validation_raw_error_deg: float | None = None,
        validation_mlp_error_deg: float | None = None,
        max_correction_deg: float = 35.0,
    ) -> None:
        self._input_mean = input_mean
        self._input_scale = input_scale
        self._output_mean = output_mean
        self._output_scale = output_scale
        self._weights = weights
        self._biases = biases
        self._sample_count = sample_count
        self._target_count = target_count
        self._validation_raw_error_deg = validation_raw_error_deg
        self._validation_mlp_error_deg = validation_mlp_error_deg
        self._max_correction_deg = max_correction_deg

    @property
    def fitted(self) -> bool:
        return (
            self._target_count >= _MIN_TARGETS
            and self._input_mean is not None
            and self._input_scale is not None
            and self._output_mean is not None
            and self._output_scale is not None
            and len(self._weights) == 3
            and len(self._biases) == 3
        )

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def target_count(self) -> int:
        return self._target_count

    @property
    def validation_raw_error_deg(self) -> float | None:
        return self._validation_raw_error_deg

    @property
    def validation_mlp_error_deg(self) -> float | None:
        return self._validation_mlp_error_deg

    @classmethod
    def fit(
        cls,
        samples: Sequence[CalibrationSampleLike],
        *,
        epochs: int = 160,
        learning_rate: float = 0.01,
        l2: float = 0.0005,
        seed: int = 26,
    ) -> GazeMLPCalibrationModel:
        usable = [sample for sample in samples if cls._valid_sample(sample)]
        target_keys = [
            (round(sample.target_yaw, 3), round(sample.target_pitch, 3)) for sample in usable
        ]
        unique_targets = sorted(set(target_keys))
        if len(usable) < _MIN_SAMPLES or len(unique_targets) < _MIN_TARGETS:
            return cls(sample_count=len(usable), target_count=len(unique_targets))

        features = np.asarray([sample.features[1:] for sample in usable], dtype=np.float64)
        targets = np.asarray(
            [[sample.target_yaw, sample.target_pitch] for sample in usable], dtype=np.float64
        )
        raw_angles = np.asarray(
            [[sample.features[1], sample.features[2]] for sample in usable], dtype=np.float64
        )
        residuals = targets - raw_angles
        train_indices, validation_indices = cls._split_indices(target_keys)
        x_train = features[train_indices]
        y_train = residuals[train_indices]
        x_validation = features[validation_indices]
        y_validation = residuals[validation_indices]
        target_validation = targets[validation_indices]
        raw_validation = raw_angles[validation_indices]

        input_mean = x_train.mean(axis=0)
        input_scale = np.where(x_train.std(axis=0) > 1e-6, x_train.std(axis=0), 1.0)
        output_mean = y_train.mean(axis=0)
        output_scale = np.where(y_train.std(axis=0) > 1e-6, y_train.std(axis=0), 1.0)
        xz = (x_train - input_mean) / input_scale
        yz = (y_train - output_mean) / output_scale

        rng = np.random.default_rng(seed)
        weights = [
            rng.normal(0.0, math.sqrt(2.0 / _INPUT_DIM), (_INPUT_DIM, _HIDDEN_1)),
            rng.normal(0.0, math.sqrt(2.0 / _HIDDEN_1), (_HIDDEN_1, _HIDDEN_2)),
            rng.normal(0.0, math.sqrt(1.0 / _HIDDEN_2), (_HIDDEN_2, 2)),
        ]
        biases = [np.zeros(_HIDDEN_1), np.zeros(_HIDDEN_2), np.zeros(2)]
        first_moments = [np.zeros_like(value) for value in (*weights, *biases)]
        second_moments = [np.zeros_like(value) for value in (*weights, *biases)]

        best_values = tuple(value.copy() for value in (*weights, *biases))
        best_validation_loss = math.inf
        for step in range(1, epochs + 1):
            hidden_1 = np.tanh(xz @ weights[0] + biases[0])
            hidden_2 = np.tanh(hidden_1 @ weights[1] + biases[1])
            prediction = hidden_2 @ weights[2] + biases[2]
            gradient_output = 2.0 * (prediction - yz) / len(xz)
            gradients_w = [
                np.empty_like(weights[0]),
                np.empty_like(weights[1]),
                hidden_2.T @ gradient_output + l2 * weights[2],
            ]
            gradients_b = [
                np.empty_like(biases[0]),
                np.empty_like(biases[1]),
                gradient_output.sum(axis=0),
            ]
            gradient_hidden_2 = (gradient_output @ weights[2].T) * (1.0 - hidden_2**2)
            gradients_w[1] = hidden_1.T @ gradient_hidden_2 + l2 * weights[1]
            gradients_b[1] = gradient_hidden_2.sum(axis=0)
            gradient_hidden_1 = (gradient_hidden_2 @ weights[1].T) * (1.0 - hidden_1**2)
            gradients_w[0] = xz.T @ gradient_hidden_1 + l2 * weights[0]
            gradients_b[0] = gradient_hidden_1.sum(axis=0)

            values = [*weights, *biases]
            gradients = [*gradients_w, *gradients_b]
            cls._adam_step(
                values,
                gradients,
                first_moments,
                second_moments,
                step=step,
                learning_rate=learning_rate,
            )

            if step == 1 or step % 5 == 0 or step == epochs:
                validation_prediction = cls._forward_normalized(
                    (x_validation - input_mean) / input_scale,
                    weights,
                    biases,
                )
                validation_loss = float(
                    np.mean((validation_prediction - (y_validation - output_mean) / output_scale) ** 2)
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_values = tuple(value.copy() for value in values)

        best_weights = tuple(best_values[:3])
        best_biases = tuple(best_values[3:])
        normalized_prediction = cls._forward_normalized(
            (x_validation - input_mean) / input_scale,
            list(best_weights),
            list(best_biases),
        )
        corrected_validation = raw_validation + normalized_prediction * output_scale + output_mean
        raw_error = cls._mean_angular_error(target_validation, raw_validation)
        mlp_error = cls._mean_angular_error(target_validation, corrected_validation)
        if not math.isfinite(mlp_error) or mlp_error >= raw_error:
            return cls(
                sample_count=len(usable),
                target_count=len(unique_targets),
                validation_raw_error_deg=raw_error,
                validation_mlp_error_deg=mlp_error,
            )
        return cls(
            input_mean=input_mean,
            input_scale=input_scale,
            output_mean=output_mean,
            output_scale=output_scale,
            weights=best_weights,
            biases=best_biases,
            sample_count=len(usable),
            target_count=len(unique_targets),
            validation_raw_error_deg=raw_error,
            validation_mlp_error_deg=mlp_error,
        )

    def predict_yaw_pitch(self, features: Sequence[float]) -> tuple[float, float] | None:
        if not self.fitted:
            return None
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (_FEATURE_DIM,) or not np.all(np.isfinite(values)):
            return None
        assert self._input_mean is not None
        assert self._input_scale is not None
        assert self._output_mean is not None
        assert self._output_scale is not None
        normalized = (values[1:] - self._input_mean) / self._input_scale
        prediction = self._forward_normalized(
            normalized[None, :], list(self._weights), list(self._biases)
        )[0]
        correction = prediction * self._output_scale + self._output_mean
        correction = np.clip(
            correction,
            -self._max_correction_deg,
            self._max_correction_deg,
        )
        yaw = float(values[1] + correction[0])
        pitch = float(values[2] + correction[1])
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            return None
        return yaw, pitch

    def correct(self, observation: FaceObservation, raw_gaze: GazeVector) -> GazeVector:
        # Local import avoids a module cycle while keeping one canonical feature builder.
        from jarvis.gaze.calibration_model import observation_features

        prediction = self.predict_yaw_pitch(observation_features(observation, raw_gaze))
        if prediction is None:
            return raw_gaze
        direction: Vector3 = yaw_pitch_to_direction(*prediction)
        return GazeVector(
            direction=direction,
            confidence=raw_gaze.confidence,
            timestamp_ms=raw_gaze.timestamp_ms,
            frame_id=raw_gaze.frame_id,
            origin=raw_gaze.origin,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "architecture": [_INPUT_DIM, _HIDDEN_1, _HIDDEN_2, 2],
            "sample_count": self._sample_count,
            "target_count": self._target_count,
            "validation_raw_error_deg": self._validation_raw_error_deg,
            "validation_mlp_error_deg": self._validation_mlp_error_deg,
            "max_correction_deg": self._max_correction_deg,
            "input_mean": self._input_mean.tolist() if self._input_mean is not None else None,
            "input_scale": self._input_scale.tolist() if self._input_scale is not None else None,
            "output_mean": self._output_mean.tolist() if self._output_mean is not None else None,
            "output_scale": self._output_scale.tolist() if self._output_scale is not None else None,
            "weights": [value.tolist() for value in self._weights],
            "biases": [value.tolist() for value in self._biases],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GazeMLPCalibrationModel | None:
        try:
            architecture = payload.get("architecture")
            if architecture != [_INPUT_DIM, _HIDDEN_1, _HIDDEN_2, 2]:
                return None
            input_mean = np.asarray(payload["input_mean"], dtype=np.float64)
            input_scale = np.asarray(payload["input_scale"], dtype=np.float64)
            output_mean = np.asarray(payload["output_mean"], dtype=np.float64)
            output_scale = np.asarray(payload["output_scale"], dtype=np.float64)
            raw_weights = payload["weights"]
            raw_biases = payload["biases"]
            if not isinstance(raw_weights, list) or not isinstance(raw_biases, list):
                return None
            weights = tuple(np.asarray(value, dtype=np.float64) for value in raw_weights)
            biases = tuple(np.asarray(value, dtype=np.float64) for value in raw_biases)
            raw_sample_count = payload.get("sample_count", 0)
            raw_target_count = payload.get("target_count", 0)
            raw_max_correction = payload.get("max_correction_deg", 35.0)
            if (
                not isinstance(raw_sample_count, int)
                or not isinstance(raw_target_count, int)
                or not isinstance(raw_max_correction, (int, float))
            ):
                return None
            model = cls(
                input_mean=input_mean,
                input_scale=input_scale,
                output_mean=output_mean,
                output_scale=output_scale,
                weights=weights,
                biases=biases,
                sample_count=raw_sample_count,
                target_count=raw_target_count,
                validation_raw_error_deg=cls._optional_float(
                    payload.get("validation_raw_error_deg")
                ),
                validation_mlp_error_deg=cls._optional_float(
                    payload.get("validation_mlp_error_deg")
                ),
                max_correction_deg=float(raw_max_correction),
            )
        except (KeyError, TypeError, ValueError):
            return None
        expected_weights = ((_INPUT_DIM, _HIDDEN_1), (_HIDDEN_1, _HIDDEN_2), (_HIDDEN_2, 2))
        expected_biases = ((_HIDDEN_1,), (_HIDDEN_2,), (2,))
        arrays = (*weights, *biases, input_mean, input_scale, output_mean, output_scale)
        if (
            tuple(value.shape for value in weights) != expected_weights
            or tuple(value.shape for value in biases) != expected_biases
            or input_mean.shape != (_INPUT_DIM,)
            or input_scale.shape != (_INPUT_DIM,)
            or output_mean.shape != (2,)
            or output_scale.shape != (2,)
            or not all(np.all(np.isfinite(value)) for value in arrays)
        ):
            return None
        return model if model.fitted else None

    @staticmethod
    def _valid_sample(sample: CalibrationSampleLike) -> bool:
        values = (*sample.features, sample.target_yaw, sample.target_pitch)
        return len(sample.features) == _FEATURE_DIM and all(math.isfinite(value) for value in values)

    @staticmethod
    def _split_indices(
        target_keys: Sequence[tuple[float, float]],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        groups: dict[tuple[float, float], list[int]] = {}
        for index, key in enumerate(target_keys):
            groups.setdefault(key, []).append(index)
        train: list[int] = []
        validation: list[int] = []
        for indices in groups.values():
            for position, index in enumerate(indices):
                (validation if position % 5 == 0 else train).append(index)
        return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)

    @staticmethod
    def _forward_normalized(
        values: npt.NDArray[np.float64],
        weights: Sequence[npt.NDArray[np.float64]],
        biases: Sequence[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        hidden_1 = np.tanh(values @ weights[0] + biases[0])
        hidden_2 = np.tanh(hidden_1 @ weights[1] + biases[1])
        result = hidden_2 @ weights[2] + biases[2]
        return cast("npt.NDArray[np.float64]", result)

    @staticmethod
    def _adam_step(
        values: Sequence[npt.NDArray[np.float64]],
        gradients: Sequence[npt.NDArray[np.float64]],
        first_moments: Sequence[npt.NDArray[np.float64]],
        second_moments: Sequence[npt.NDArray[np.float64]],
        *,
        step: int,
        learning_rate: float,
    ) -> None:
        beta_1, beta_2 = 0.9, 0.999
        for value, gradient, first, second in zip(
            values, gradients, first_moments, second_moments, strict=True
        ):
            np.clip(gradient, -5.0, 5.0, out=gradient)
            first *= beta_1
            first += (1.0 - beta_1) * gradient
            second *= beta_2
            second += (1.0 - beta_2) * gradient**2
            first_hat = first / (1.0 - beta_1**step)
            second_hat = second / (1.0 - beta_2**step)
            value -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)

    @staticmethod
    def _mean_angular_error(
        target: npt.NDArray[np.float64], prediction: npt.NDArray[np.float64]
    ) -> float:
        delta_yaw = (target[:, 0] - prediction[:, 0]) * np.cos(np.radians(target[:, 1]))
        delta_pitch = target[:, 1] - prediction[:, 1]
        return float(np.mean(np.hypot(delta_yaw, delta_pitch)))

    @staticmethod
    def _optional_float(value: object) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

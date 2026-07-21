"""Tiny per-user target classifier trained from look-to-register samples.

Training happens only when target registration finishes. Runtime inference is a
single standardized linear-softmax forward pass, so it is negligible compared
with MediaPipe.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from jarvis.gaze.feature_profile import FEATURE_NAMES, TargetFeatureSample
from jarvis.gaze.config import GazeConfig

_MIN_SAMPLES_PER_TARGET = 20


def _feature_weight_array(
    values: tuple[float, ...] | npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    raw = GazeConfig().personal_feature_weights if values is None else values
    weights = np.asarray(raw, dtype=np.float64)
    if (
        weights.shape != (len(FEATURE_NAMES),)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("personal classifier feature weights must be finite and positive")
    return weights


@dataclass(frozen=True, slots=True)
class PersonalTargetPrediction:
    target_id: str
    confidence: float
    second_best_confidence: float


@dataclass(frozen=True, slots=True)
class PersonalTrainingSample:
    target_id: str
    features: tuple[float, ...]

    @classmethod
    def from_feature_sample(
        cls, target_id: str, sample: TargetFeatureSample
    ) -> "PersonalTrainingSample":
        return cls(target_id=target_id, features=tuple(float(v) for v in sample.as_array()))


class PersonalTargetClassifier:
    def __init__(
        self,
        *,
        target_ids: list[str],
        mean: npt.NDArray[np.float64],
        scale: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
        feature_weights: npt.NDArray[np.float64],
        sample_count: int,
        gaze_priority_constrained: bool = True,
    ) -> None:
        self.target_ids = target_ids
        self.mean = mean
        self.scale = scale
        self.weights = weights
        self.bias = bias
        self.feature_weights = feature_weights
        self.sample_count = sample_count
        self.gaze_priority_constrained = gaze_priority_constrained

    @property
    def fitted(self) -> bool:
        return len(self.target_ids) >= 2 and self.weights.size > 0

    @classmethod
    def fit(
        cls,
        samples: list[PersonalTrainingSample],
        *,
        learning_rate: float = 0.08,
        epochs: int = 260,
        l2: float = 0.01,
        feature_weights: tuple[float, ...] | npt.NDArray[np.float64] | None = None,
    ) -> "PersonalTargetClassifier | None":
        counts: dict[str, int] = {}
        for sample in samples:
            counts[sample.target_id] = counts.get(sample.target_id, 0) + 1
        target_ids = sorted(
            target_id for target_id, count in counts.items() if count >= _MIN_SAMPLES_PER_TARGET
        )
        if len(target_ids) < 2:
            return None

        index = {target_id: i for i, target_id in enumerate(target_ids)}
        kept = [sample for sample in samples if sample.target_id in index]
        x = np.asarray([sample.features for sample in kept], dtype=np.float64)
        y = np.asarray([index[sample.target_id] for sample in kept], dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(x)):
            return None

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        priority = _feature_weight_array(feature_weights)
        xz = ((x - mean) / scale) * priority
        classes = len(target_ids)
        weights = np.zeros((xz.shape[1], classes), dtype=np.float64)
        bias = np.zeros(classes, dtype=np.float64)
        one_hot = np.eye(classes, dtype=np.float64)[y]
        n = float(len(xz))
        for _ in range(epochs):
            logits = xz @ weights + bias
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            error = probs - one_hot
            weights -= learning_rate * ((xz.T @ error) / n + l2 * weights)
            bias -= learning_rate * error.mean(axis=0)
        cls._constrain_head_contribution(weights, priority)
        return cls(
            target_ids=target_ids,
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
            feature_weights=priority,
            sample_count=len(kept),
            gaze_priority_constrained=True,
        )

    @staticmethod
    def _constrain_head_contribution(
        weights: npt.NDArray[np.float64],
        feature_weights: npt.NDArray[np.float64],
    ) -> None:
        """Guarantee head context cannot outweigh gaze evidence per class.

        Input scaling alone only changes the regularization pressure; a model
        can still compensate by learning larger head coefficients. Clamp each
        class's effective head norm to the configured head:gaze priority ratio.
        If gaze carries no target evidence, head is suppressed and the runtime
        confidence gate sends the frame to the gaze-area fallback.
        """
        gaze_priority = float(np.mean(feature_weights[:2]))
        head_priority = float(np.mean(feature_weights[2:5]))
        maximum_ratio = head_priority / gaze_priority
        for class_index in range(weights.shape[1]):
            gaze_norm = float(
                np.linalg.norm(weights[:2, class_index] * feature_weights[:2])
            )
            head_norm = float(
                np.linalg.norm(weights[2:5, class_index] * feature_weights[2:5])
            )
            maximum_head_norm = gaze_norm * maximum_ratio
            if head_norm > maximum_head_norm and head_norm > 0.0:
                weights[2:5, class_index] *= maximum_head_norm / head_norm

    def predict(
        self,
        sample: TargetFeatureSample,
        *,
        score_multipliers: dict[str, float] | None = None,
    ) -> PersonalTargetPrediction | None:
        if not self.fitted:
            return None
        x = np.asarray(sample.as_array(), dtype=np.float64)
        if x.shape != self.mean.shape or not np.all(np.isfinite(x)):
            return None
        logits = (((x - self.mean) / self.scale) * self.feature_weights) @ self.weights + self.bias
        if score_multipliers is not None:
            for index, target_id in enumerate(self.target_ids):
                multiplier = float(score_multipliers.get(target_id, 1.0))
                if math.isfinite(multiplier) and multiplier > 0.0:
                    logits[index] += math.log(multiplier)
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        confidence = float(probs[best])
        if not math.isfinite(confidence):
            return None
        return PersonalTargetPrediction(
            target_id=self.target_ids[best],
            confidence=confidence,
            second_best_confidence=float(probs[second]) if second != best else 0.0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "target_ids": self.target_ids,
            "sample_count": self.sample_count,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "feature_weights": self.feature_weights.tolist(),
            "gaze_priority_constrained": self.gaze_priority_constrained,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PersonalTargetClassifier | None":
        try:
            raw_target_ids = payload["target_ids"]
            raw_mean = payload["mean"]
            raw_scale = payload["scale"]
            raw_weights = payload["weights"]
            raw_bias = payload["bias"]
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(raw_target_ids, list):
            return None
        try:
            target_ids = [str(v) for v in raw_target_ids]
            mean = np.asarray(raw_mean, dtype=np.float64)
            scale = np.asarray(raw_scale, dtype=np.float64)
            feature_weights = np.asarray(
                payload.get("feature_weights", np.ones(len(FEATURE_NAMES))),
                dtype=np.float64,
            )
            weights = np.asarray(raw_weights, dtype=np.float64)
            bias = np.asarray(raw_bias, dtype=np.float64)
            raw_sample_count = payload.get("sample_count", 0)
            sample_count = int(raw_sample_count) if isinstance(raw_sample_count, int) else 0
        except (TypeError, ValueError):
            return None
        if (
            mean.shape != (len(FEATURE_NAMES),)
            or scale.shape != mean.shape
            or feature_weights.shape != mean.shape
            or not np.all(np.isfinite(feature_weights))
            or np.any(feature_weights <= 0.0)
        ):
            return None
        if weights.shape != (len(FEATURE_NAMES), len(target_ids)) or bias.shape != (len(target_ids),):
            return None
        return cls(
            target_ids=target_ids,
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
            feature_weights=feature_weights,
            sample_count=sample_count,
            gaze_priority_constrained=payload.get("gaze_priority_constrained") is True,
        )


class PersonalTargetStore:
    def __init__(
        self,
        path: Path,
        *,
        confidence_threshold: float = 0.65,
        feature_weights: tuple[float, ...] | npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.path = path
        self.confidence_threshold = confidence_threshold
        self.feature_weights = _feature_weight_array(feature_weights)
        self.samples: list[PersonalTrainingSample] = []
        self.model: PersonalTargetClassifier | None = None
        self._load()

    def add_samples(
        self,
        target_id: str,
        samples: list[TargetFeatureSample],
        *,
        replace_target: bool = True,
    ) -> PersonalTargetClassifier | None:
        if replace_target:
            self.samples = [sample for sample in self.samples if sample.target_id != target_id]
        self.samples.extend(
            PersonalTrainingSample.from_feature_sample(target_id, sample) for sample in samples
        )
        self.model = PersonalTargetClassifier.fit(
            self.samples,
            feature_weights=self.feature_weights,
        )
        self._save()
        return self.model

    def remove_target(self, target_id: str) -> None:
        self.samples = [sample for sample in self.samples if sample.target_id != target_id]
        self.model = PersonalTargetClassifier.fit(
            self.samples,
            feature_weights=self.feature_weights,
        )
        self._save()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        raw_samples = payload.get("samples", [])
        if isinstance(raw_samples, list):
            for item in raw_samples:
                if not isinstance(item, dict):
                    continue
                features = item.get("features")
                target_id = item.get("target_id")
                if isinstance(target_id, str) and isinstance(features, list):
                    try:
                        self.samples.append(
                            PersonalTrainingSample(
                                target_id=target_id,
                                features=tuple(float(v) for v in features),
                            )
                        )
                    except (TypeError, ValueError):
                        pass
        model_payload = payload.get("model")
        if isinstance(model_payload, dict):
            self.model = PersonalTargetClassifier.from_dict(model_payload)
        if self.samples and (
            self.model is None
            or not np.allclose(self.model.feature_weights, self.feature_weights)
            or not self.model.gaze_priority_constrained
        ):
            self.model = PersonalTargetClassifier.fit(
                self.samples,
                feature_weights=self.feature_weights,
            )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "confidence_threshold": self.confidence_threshold,
            "feature_weights": self.feature_weights.tolist(),
            "model": self.model.to_dict() if self.model is not None else None,
            "samples": [
                {"target_id": sample.target_id, "features": list(sample.features)}
                for sample in self.samples
            ],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

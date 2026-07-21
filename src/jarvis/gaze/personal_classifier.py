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

_MIN_SAMPLES_PER_TARGET = 20


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
        sample_count: int,
    ) -> None:
        self.target_ids = target_ids
        self.mean = mean
        self.scale = scale
        self.weights = weights
        self.bias = bias
        self.sample_count = sample_count

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
        xz = (x - mean) / scale
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
        return cls(
            target_ids=target_ids,
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
            sample_count=len(kept),
        )

    def predict(self, sample: TargetFeatureSample) -> PersonalTargetPrediction | None:
        if not self.fitted:
            return None
        x = np.asarray(sample.as_array(), dtype=np.float64)
        if x.shape != self.mean.shape or not np.all(np.isfinite(x)):
            return None
        logits = ((x - self.mean) / self.scale) @ self.weights + self.bias
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
            weights = np.asarray(raw_weights, dtype=np.float64)
            bias = np.asarray(raw_bias, dtype=np.float64)
            raw_sample_count = payload.get("sample_count", 0)
            sample_count = int(raw_sample_count) if isinstance(raw_sample_count, int) else 0
        except (TypeError, ValueError):
            return None
        if mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape:
            return None
        if weights.shape != (len(FEATURE_NAMES), len(target_ids)) or bias.shape != (len(target_ids),):
            return None
        return cls(
            target_ids=target_ids,
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
            sample_count=sample_count,
        )


class PersonalTargetStore:
    def __init__(self, path: Path, *, confidence_threshold: float = 0.65) -> None:
        self.path = path
        self.confidence_threshold = confidence_threshold
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
        self.model = PersonalTargetClassifier.fit(self.samples)
        self._save()
        return self.model

    def remove_target(self, target_id: str) -> None:
        self.samples = [sample for sample in self.samples if sample.target_id != target_id]
        self.model = PersonalTargetClassifier.fit(self.samples)
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

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "confidence_threshold": self.confidence_threshold,
            "model": self.model.to_dict() if self.model is not None else None,
            "samples": [
                {"target_id": sample.target_id, "features": list(sample.features)}
                for sample in self.samples
            ],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

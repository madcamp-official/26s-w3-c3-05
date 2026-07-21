"""Per-user target classifier trained from look-to-register feature samples."""

from __future__ import annotations

import json
from pathlib import Path

from jarvis.gaze.feature_profile import TargetFeatureSample
from jarvis.gaze.personal_classifier import (
    PersonalTargetClassifier,
    PersonalTargetStore,
    PersonalTrainingSample,
)


def _samples(
    target_id: str,
    *,
    gaze_yaw: float,
    gaze_pitch: float,
    head_yaw: float,
) -> list[tuple[str, TargetFeatureSample]]:
    return [
        (
            target_id,
            TargetFeatureSample(
                gaze_yaw=gaze_yaw + (i % 5) * 0.1,
                gaze_pitch=gaze_pitch + (i % 4) * 0.1,
                head_yaw=head_yaw + (i % 3) * 0.2,
                head_pitch=4.0 + (i % 2) * 0.1,
                head_roll=0.5,
                face_scale=0.10 + (i % 2) * 0.002,
            ),
        )
        for i in range(24)
    ]


def test_linear_softmax_predicts_target_from_registered_features() -> None:
    training = [
        *[
            item
            for target_id, feature in _samples(
                "monitor", gaze_yaw=1.0, gaze_pitch=3.0, head_yaw=-5.0
            )
            for item in [(target_id, feature)]
        ],
        *[
            item
            for target_id, feature in _samples(
                "speaker", gaze_yaw=18.0, gaze_pitch=8.0, head_yaw=15.0
            )
            for item in [(target_id, feature)]
        ],
    ]
    fitted = PersonalTargetClassifier.fit(
        [
            PersonalTrainingSample.from_feature_sample(target_id, feature)
            for target_id, feature in training
        ]
    )

    assert fitted is not None
    prediction = fitted.predict(
        TargetFeatureSample(18.1, 8.1, 15.2, 4.0, 0.5, 0.10)
    )
    assert prediction is not None
    assert prediction.target_id == "speaker"
    assert prediction.confidence > 0.80


def test_store_replaces_target_samples_and_persists_model(tmp_path: Path) -> None:
    path = tmp_path / "personal_target_classifier.json"
    store = PersonalTargetStore(path)
    store.add_samples(
        "monitor",
        [feature for _target_id, feature in _samples("monitor", gaze_yaw=1.0, gaze_pitch=3.0, head_yaw=-5.0)],
    )
    model = store.add_samples(
        "speaker",
        [feature for _target_id, feature in _samples("speaker", gaze_yaw=18.0, gaze_pitch=8.0, head_yaw=15.0)],
    )

    assert model is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model"]["target_ids"] == ["monitor", "speaker"]

    replacement = [
        feature
        for _target_id, feature in _samples("monitor", gaze_yaw=-10.0, gaze_pitch=2.0, head_yaw=-20.0)
    ]
    store.add_samples("monitor", replacement, replace_target=True)

    reloaded = PersonalTargetStore(path)
    assert len([sample for sample in reloaded.samples if sample.target_id == "monitor"]) == len(
        replacement
    )
    assert reloaded.model is not None
    assert reloaded.model.predict(replacement[0]).target_id == "monitor"


def test_store_waits_until_two_targets_have_enough_samples(tmp_path: Path) -> None:
    store = PersonalTargetStore(tmp_path / "personal_target_classifier.json")

    model = store.add_samples(
        "monitor",
        [
            feature
            for _target_id, feature in _samples("monitor", gaze_yaw=1.0, gaze_pitch=3.0, head_yaw=-5.0)
        ],
    )

    assert model is None
    assert store.model is None

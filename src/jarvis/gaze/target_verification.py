"""재등록-검증 루프 분석: 등록 직후와 시간 경과 후의 area 판정을 비교한다.

목적(documents/gaze.md 2026-07-22): pose 보정이 실사용에서 어긋나는 원인이
"등록 수집 자체의 버그"인지 "세션이 지나며 생기는 드리프트"인지 가르는 것.
등록 직후 스윕에서도 OUT이면 수집 문제, 직후엔 IN인데 나중 실행에서 OUT이면
드리프트 — 그때는 등록 단위가 아니라 세션 시작 재보정/온라인 편향 추정으로
옮겨야 한다.

이 모듈은 순수 `TargetFeatureSample` 목록만 다루므로 카메라 없이 단위 테스트
한다. 실카메라 캡처는 `jarvis-gaze verify-target`(cli.py)이 맡는다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from jarvis.gaze.classifier import TargetClassifier
from jarvis.gaze.config import GazeConfig
from jarvis.gaze.feature_profile import TargetFeatureSample


@dataclass(frozen=True, slots=True)
class BinVerification:
    """한 head-yaw 구간의 검증 결과."""

    label: str
    median_head_yaw_deg: float
    frame_count: int
    median_raw_distance: float
    """보정 없이 원시 gaze로 잰 area 정규화 거리의 중앙값."""

    median_effective_distance: float
    """rescue 전용(min) 정책이 실제로 채택한 거리의 중앙값 — 런타임 판정값."""

    in_fraction: float
    """effective distance <= tolerance 인 프레임 비율."""

    rescue_fraction: float
    """원시 대신 보정 gaze가 채택된(rescue가 발동한) 프레임 비율."""


@dataclass(frozen=True, slots=True)
class TargetVerificationSummary:
    device_id: str
    total_samples: int
    tolerance: float
    bins: tuple[BinVerification, ...]


def verify_target_samples(
    classifier: TargetClassifier,
    device_id: str,
    samples: Sequence[TargetFeatureSample],
    config: GazeConfig = GazeConfig(),
) -> TargetVerificationSummary:
    """대상 물체 중앙을 응시하며 고개를 돌린 스윕 샘플을 bin별로 판정한다.

    런타임과 동일한 `TargetClassifier.area_distance_and_gaze`(rescue 전용 min
    정책)를 그대로 사용한다 — 검증 전용 판정 경로를 따로 만들지 않는다.
    """
    profile = classifier.area_profiles.get(device_id)
    if profile is None:
        raise ValueError(f"no traced area profile registered for '{device_id}'")

    edges = (-math.inf, *config.pose_correction_bin_edges_deg, math.inf)
    bins: list[list[TargetFeatureSample]] = [[] for _ in range(len(edges) - 1)]
    labels = [_bin_label(lower, upper) for lower, upper in zip(edges, edges[1:], strict=False)]
    for sample in samples:
        for index, (lower, upper) in enumerate(zip(edges, edges[1:], strict=False)):
            if lower <= sample.head_yaw < upper:
                bins[index].append(sample)
                break

    max_radius = config.registration_max_area_radius_deg
    results: list[BinVerification] = []
    for label, members in zip(labels, bins, strict=True):
        if not members:
            continue
        raw_distances = []
        effective_distances = []
        rescued = 0
        for sample in members:
            raw = profile.normalized_distance(sample.gaze_yaw, sample.gaze_pitch, max_radius)
            effective, gaze_yaw, gaze_pitch = classifier.area_distance_and_gaze(
                device_id, profile, sample
            )
            raw_distances.append(raw)
            effective_distances.append(effective)
            if (gaze_yaw, gaze_pitch) != (sample.gaze_yaw, sample.gaze_pitch):
                rescued += 1
        effective_array = np.asarray(effective_distances, dtype=np.float64)
        results.append(
            BinVerification(
                label=label,
                median_head_yaw_deg=float(
                    np.median(np.asarray([s.head_yaw for s in members], dtype=np.float64))
                ),
                frame_count=len(members),
                median_raw_distance=float(np.median(np.asarray(raw_distances))),
                median_effective_distance=float(np.median(effective_array)),
                in_fraction=float(np.mean(effective_array <= config.target_match_tolerance)),
                rescue_fraction=rescued / len(members),
            )
        )
    return TargetVerificationSummary(
        device_id=device_id,
        total_samples=len(samples),
        tolerance=config.target_match_tolerance,
        bins=tuple(results),
    )


def _bin_label(lower: float, upper: float) -> str:
    left = "-inf" if math.isinf(lower) else f"{lower:+.0f}"
    right = "+inf" if math.isinf(upper) else f"{upper:+.0f}"
    return f"[{left},{right})"


_MINIMUM_COMPARE_FRAMES = 8
"""이보다 표본이 적은 bin은 비교 판정에서 제외한다(우연을 결론으로 만들지 않는다)."""

_IN_FRACTION_THRESHOLD = 0.7
"""bin을 IN으로 판정하는 최소 in_fraction — 스윕 가장자리 프레임의 잡음 허용치."""


def compare_verifications(
    earlier_bins: Sequence[dict[str, Any]],
    later_bins: Sequence[dict[str, Any]],
) -> list[str]:
    """등록 직후 실행(earlier)과 나중 실행(later)의 bin 판정을 비교한다.

    JSON으로 저장된 `BinVerification` dict 목록을 그대로 받는다(CLI가 파일에서
    읽은 값). 반환은 사람이 읽는 판정 문장 목록이다.
    """
    earlier_by_label = {b["label"]: b for b in earlier_bins}
    lines: list[str] = []
    drift_bins: list[str] = []
    collection_bins: list[str] = []
    for later in later_bins:
        earlier = earlier_by_label.get(later["label"])
        if earlier is None:
            continue
        if (
            earlier["frame_count"] < _MINIMUM_COMPARE_FRAMES
            or later["frame_count"] < _MINIMUM_COMPARE_FRAMES
        ):
            continue
        earlier_in = earlier["in_fraction"] >= _IN_FRACTION_THRESHOLD
        later_in = later["in_fraction"] >= _IN_FRACTION_THRESHOLD
        lines.append(
            f"{later['label']}: 직후 x{earlier['median_effective_distance']:.2f}"
            f" (IN {earlier['in_fraction']:.0%}) → 현재"
            f" x{later['median_effective_distance']:.2f} (IN {later['in_fraction']:.0%})"
        )
        if not earlier_in:
            collection_bins.append(later["label"])
        elif earlier_in and not later_in:
            drift_bins.append(later["label"])
    if collection_bins:
        lines.append(
            f"판정: {', '.join(collection_bins)} 구간은 등록 직후부터 OUT — "
            "등록 수집 문제(스윕이 그 자세를 안 지났거나 bin이 IQR 게이트에 걸림). "
            "저장된 pose_correction.points와 등록 스윕 범위를 확인하세요."
        )
    if drift_bins:
        lines.append(
            f"판정: {', '.join(drift_bins)} 구간은 직후엔 IN이었는데 지금 OUT — "
            "세션 드리프트. 등록 단위 보정이 아니라 세션 시작 시 짧은 재보정"
            "(또는 온라인 편향 추정)으로 옮겨야 합니다."
        )
    if not collection_bins and not drift_bins and lines:
        lines.append("판정: 비교 가능한 모든 구간이 두 실행 모두 IN — 보정이 유지되고 있습니다.")
    if not lines:
        lines.append("비교 가능한 bin이 없습니다(표본 부족) — 스윕을 더 넓고 길게 다시 실행하세요.")
    return lines

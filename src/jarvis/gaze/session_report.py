"""라벨된 gaze 세션(JSONL)을 집계해 정확도·편향 리포트를 만든다.

`jarvis.monitoring.session_recorder`가 남긴 세션 파일 하나만으로 동작한다 —
헤더에 기록 시점의 config와 target 프로필(보정 테이블 포함)이 들어 있으므로
현재 profiles.json 상태와 무관하게 재현 가능한 분석이 나온다.

집계 축은 두 개다:

- 정답 라벨(사용자가 녹화 중 표시한 "지금 보고 있는 것")별 분류 결과 분포와
  탈락 사유.
- 라벨 × head-yaw 구간별: in-area 비율, 실측 gaze 편향(중앙값)과 저장된
  pose 보정 오프셋의 대조. 등록 시 배운 보정이 현재 세션의 실제 편향과
  어긋나는 구간을 경고로 드러낸다 — 2026-07-22 디버깅에서 손으로 만들던 표.

라벨 규약은 레코더와 같다: ``"none"``은 "아무 target도 보지 않음"이라는 명시적
정답(기대 결과 UNKNOWN)이고, 라벨이 없는 프레임은 정확도 집계에서 제외한다.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.gaze.feature_profile import PoseCorrectionPoint, TargetPoseCorrection

NO_TARGET_LABEL = "none"
UNKNOWN_TARGET = "UNKNOWN"

#: 실측 편향과 저장된 보정 오프셋이 이보다 벌어진 bin은 경고로 표시한다(도).
CORRECTION_MISMATCH_WARNING_DEG = 3.0


@dataclass(frozen=True, slots=True)
class SessionData:
    header: dict[str, Any]
    frames: list[dict[str, Any]]

    @property
    def target_names(self) -> dict[str, str]:
        return {
            str(record["target_id"]): str(record["name"])
            for record in self.header.get("targets", [])
        }

    def target_record(self, target_id: str) -> dict[str, Any] | None:
        for record in self.header.get("targets", []):
            if record["target_id"] == target_id:
                return dict(record)
        return None


def load_session(path: Path) -> SessionData:
    header: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON line") from error
            kind = payload.get("type")
            if kind == "header":
                header = payload
            elif kind == "frame":
                frames.append(payload)
            # footer는 요약 중복이므로 무시한다(프레임에서 다시 센다).
    if header is None:
        raise ValueError(f"{path}: session header line is missing")
    return SessionData(header=header, frames=frames)


def _pose_correction_from_record(record: dict[str, Any]) -> TargetPoseCorrection | None:
    stored = record.get("pose_correction")
    if not stored or not stored.get("points"):
        return None
    return TargetPoseCorrection(
        points=tuple(
            PoseCorrectionPoint(
                head_yaw_deg=float(point["head_yaw_deg"]),
                offset_yaw_deg=float(point["offset_yaw_deg"]),
                offset_pitch_deg=float(point["offset_pitch_deg"]),
                sample_count=int(point["sample_count"]),
            )
            for point in stored["points"]
        ),
        reference_head_yaw_deg=stored.get("reference_head_yaw_deg"),
    )


def _reject_reason_key(reason: str | None) -> str:
    """수치가 섞인 탈락 사유 문자열을 히스토그램 키로 정규화한다."""
    if reason is None:
        return "(none)"
    return reason.split(":", 1)[0].strip()


def _percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


@dataclass(frozen=True, slots=True)
class BinStats:
    lower_deg: float
    upper_deg: float
    frame_count: int
    accuracy_percent: float
    unknown_percent: float
    in_area_percent: float | None
    correction_used_percent: float | None
    measured_bias_yaw: float | None
    measured_bias_pitch: float | None
    stored_offset_yaw: float | None
    stored_offset_pitch: float | None
    in_correction_coverage: bool | None


@dataclass(frozen=True, slots=True)
class SliceStats:
    """한 관측 축의 구간별 판정 성능."""

    lower: float
    upper: float
    frame_count: int
    accuracy_percent: float
    unknown_percent: float
    no_gaze_percent: float
    in_area_percent: float | None


@dataclass(frozen=True, slots=True)
class FailureExample:
    """같은 실패를 수백 줄 나열하지 않고 원인별 한 프레임만 보존한다."""

    frame_id: int
    timestamp_ms: int
    predicted: str
    reason: str
    gaze_source: str
    feature_gaze: tuple[float, float] | None
    head: tuple[float, float, float]
    face_center: tuple[float, float] | None
    face_scale: float | None
    eyes_open: bool | None
    eye_open_ratio: tuple[float | None, float | None] | None
    area_normalized_distance: float | None
    feature_normalized_distance: float | None


@dataclass(frozen=True, slots=True)
class LabelReport:
    label: str
    display_name: str
    frame_count: int
    no_gaze_frames: int
    classified: dict[str, int]
    reject_reasons: dict[str, int]
    bins: list[BinStats]
    lock_completions: int
    max_dwell_ms: int
    warnings: list[str]
    axis_slices: dict[str, list[SliceStats]]
    failure_examples: list[FailureExample]
    source_reasons: dict[str, int]
    closed_eye_frames: int

    @property
    def accuracy_percent(self) -> float:
        expected = UNKNOWN_TARGET if self.label == NO_TARGET_LABEL else self.label
        return _percent(self.classified.get(expected, 0), self.frame_count)


@dataclass(frozen=True, slots=True)
class SessionReport:
    path: str
    duration_ms: int
    total_frames: int
    labeled_frames: int
    gaze_sources: dict[str, int]
    labels: list[LabelReport]


def build_report(
    session: SessionData,
    *,
    path: Path | str = "",
    bin_width_deg: float = 10.0,
) -> SessionReport:
    if bin_width_deg <= 0.0 or not math.isfinite(bin_width_deg):
        raise ValueError("bin_width_deg must be finite and positive")
    frames = session.frames
    names = session.target_names
    tolerance = float(session.header["config"]["target_match_tolerance"])

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gaze_sources: Counter[str] = Counter()
    for frame in frames:
        gaze_sources[str(frame["gaze"]["source"])] += 1
        label = frame.get("label")
        if label is not None:
            by_label[str(label)].append(frame)

    label_reports: list[LabelReport] = []
    for label in sorted(by_label, key=lambda key: (key == NO_TARGET_LABEL, key)):
        label_frames = by_label[label]
        classified: Counter[str] = Counter(
            str(frame["cls"]["target"]) for frame in label_frames
        )
        reject_reasons: Counter[str] = Counter(
            _reject_reason_key(frame["cls"]["reject"])
            for frame in label_frames
            if frame["cls"]["target"] == UNKNOWN_TARGET
        )
        no_gaze = sum(1 for frame in label_frames if frame["gaze"]["feature"] is None)

        record = session.target_record(label)
        correction = _pose_correction_from_record(record) if record else None
        area = record.get("area_profile") if record else None
        center = (
            (float(area["center_yaw"]), float(area["center_pitch"]))
            if area
            else None
        )

        bins = _build_bins(
            label_frames,
            label=label,
            bin_width_deg=bin_width_deg,
            tolerance=tolerance,
            center=center,
            correction=correction,
        )
        warnings = _bin_warnings(label, bins, correction)
        reference_face_scale = (
            float(record["reference_face_scale"])
            if record and record.get("reference_face_scale")
            else None
        )
        axis_slices = _build_axis_slices(
            label_frames,
            label=label,
            tolerance=tolerance,
            reference_face_scale=reference_face_scale,
        )
        warnings.extend(
            _axis_warnings(
                axis_slices,
                overall_accuracy=_percent(
                    classified.get(
                        UNKNOWN_TARGET if label == NO_TARGET_LABEL else label,
                        0,
                    ),
                    len(label_frames),
                ),
            )
        )
        source_reasons = Counter(
            str(frame["gaze"].get("source_reason") or "(none)")
            for frame in label_frames
        )

        lock_completions = 0
        max_dwell_ms = 0
        previously_locked = False
        for frame in label_frames:
            locked = frame["lock"]["locked"] == label
            if locked and not previously_locked:
                lock_completions += 1
            previously_locked = locked
            max_dwell_ms = max(max_dwell_ms, int(frame["lock"]["dwell_ms"]))

        label_reports.append(
            LabelReport(
                label=label,
                display_name=names.get(label, label),
                frame_count=len(label_frames),
                no_gaze_frames=no_gaze,
                classified=dict(classified),
                reject_reasons=dict(reject_reasons),
                bins=bins,
                lock_completions=lock_completions,
                max_dwell_ms=max_dwell_ms,
                warnings=warnings,
                axis_slices=axis_slices,
                failure_examples=_failure_examples(label_frames, label=label),
                source_reasons=dict(source_reasons),
                closed_eye_frames=sum(
                    1 for frame in label_frames if frame["obs"].get("eyes_open") is False
                ),
            )
        )

    timestamps = [int(frame["t"]) for frame in frames]
    return SessionReport(
        path=str(path),
        duration_ms=(max(timestamps) - min(timestamps)) if timestamps else 0,
        total_frames=len(frames),
        labeled_frames=sum(len(items) for items in by_label.values()),
        gaze_sources=dict(gaze_sources),
        labels=label_reports,
    )


def _face_center(frame: dict[str, Any]) -> tuple[float, float] | None:
    stored = frame["obs"].get("face_center")
    if stored is not None:
        return float(stored[0]), float(stored[1])
    centers = [
        center
        for center in (frame["obs"].get("eye_l"), frame["obs"].get("eye_r"))
        if center is not None
    ]
    if not centers:
        return None
    return (
        sum(float(center[0]) for center in centers) / len(centers),
        sum(float(center[1]) for center in centers) / len(centers),
    )


def _axis_value(
    frame: dict[str, Any],
    axis: str,
    reference_face_scale: float | None,
) -> float | None:
    if axis == "head_pitch":
        return float(frame["obs"]["head"][1])
    if axis in {"face_x", "face_y"}:
        center = _face_center(frame)
        return center[0 if axis == "face_x" else 1] if center is not None else None
    if axis == "face_scale_ratio":
        current = frame["obs"].get("face_scale")
        if current is None or reference_face_scale is None or reference_face_scale <= 0.0:
            return None
        return float(current) / reference_face_scale
    raise ValueError(f"unsupported report axis: {axis}")


def _build_axis_slices(
    frames: list[dict[str, Any]],
    *,
    label: str,
    tolerance: float,
    reference_face_scale: float | None,
) -> dict[str, list[SliceStats]]:
    widths = {
        "head_pitch": 10.0,
        "face_scale_ratio": 0.20,
        "face_x": 0.10,
        "face_y": 0.10,
    }
    expected = UNKNOWN_TARGET if label == NO_TARGET_LABEL else label
    result: dict[str, list[SliceStats]] = {}
    for axis, width in widths.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for frame in frames:
            value = _axis_value(frame, axis, reference_face_scale)
            if value is not None and math.isfinite(value):
                grouped[int(math.floor(value / width))].append(frame)
        slices: list[SliceStats] = []
        for index in sorted(grouped):
            members = grouped[index]
            target_rows = [
                frame["targets"].get(label)
                for frame in members
                if label != NO_TARGET_LABEL
            ]
            scored = [row for row in target_rows if row is not None]
            slices.append(
                SliceStats(
                    lower=index * width,
                    upper=(index + 1) * width,
                    frame_count=len(members),
                    accuracy_percent=_percent(
                        sum(1 for frame in members if frame["cls"]["target"] == expected),
                        len(members),
                    ),
                    unknown_percent=_percent(
                        sum(
                            1
                            for frame in members
                            if frame["cls"]["target"] == UNKNOWN_TARGET
                        ),
                        len(members),
                    ),
                    no_gaze_percent=_percent(
                        sum(1 for frame in members if frame["gaze"].get("feature") is None),
                        len(members),
                    ),
                    in_area_percent=(
                        _percent(
                            sum(
                                1
                                for row in scored
                                if float(row["area_nd"]) <= tolerance
                            ),
                            len(scored),
                        )
                        if scored
                        else None
                    ),
                )
            )
        if slices:
            result[axis] = slices
    return result


def _axis_warnings(
    axes: dict[str, list[SliceStats]],
    *,
    overall_accuracy: float,
) -> list[str]:
    labels = {
        "head_pitch": "head pitch",
        "face_scale_ratio": "등록 대비 얼굴 크기",
        "face_x": "화면 내 얼굴 x",
        "face_y": "화면 내 얼굴 y",
    }
    warnings: list[str] = []
    for axis, slices in axes.items():
        for item in slices:
            if item.frame_count < 10:
                continue
            if item.accuracy_percent + 20.0 <= overall_accuracy:
                warnings.append(
                    f"{labels[axis]} {item.lower:.2f}..{item.upper:.2f}: "
                    f"정확도 {item.accuracy_percent:.0f}% (전체 {overall_accuracy:.0f}%)"
                )
            elif item.no_gaze_percent >= 50.0:
                warnings.append(
                    f"{labels[axis]} {item.lower:.2f}..{item.upper:.2f}: "
                    f"시선 산출 실패 {item.no_gaze_percent:.0f}%"
                )
    return warnings


def _failure_examples(
    frames: list[dict[str, Any]],
    *,
    label: str,
    limit: int = 6,
) -> list[FailureExample]:
    expected = UNKNOWN_TARGET if label == NO_TARGET_LABEL else label
    examples: list[FailureExample] = []
    seen_reasons: set[str] = set()
    for frame in frames:
        predicted = str(frame["cls"]["target"])
        if predicted == expected:
            continue
        if frame["gaze"].get("feature") is None:
            reason = "no-gaze: " + str(
                frame["gaze"].get("source_reason") or frame["gaze"].get("source")
            )
        elif predicted == UNKNOWN_TARGET:
            reason = _reject_reason_key(frame["cls"].get("reject"))
        else:
            reason = f"misclassified as {predicted}"
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        target_detail = frame.get("targets", {}).get(label, {})
        feature_gaze = frame["gaze"].get("feature")
        center = _face_center(frame)
        eye_ratio = frame["obs"].get("eye_open_ratio")
        head = frame["obs"]["head"]
        examples.append(
            FailureExample(
                frame_id=int(frame["frame"]),
                timestamp_ms=int(frame["t"]),
                predicted=predicted,
                reason=reason,
                gaze_source=str(frame["gaze"].get("source", "unavailable")),
                feature_gaze=(
                    (float(feature_gaze[0]), float(feature_gaze[1]))
                    if feature_gaze is not None
                    else None
                ),
                head=(float(head[0]), float(head[1]), float(head[2])),
                face_center=center,
                face_scale=(
                    float(frame["obs"]["face_scale"])
                    if frame["obs"].get("face_scale") is not None
                    else None
                ),
                eyes_open=frame["obs"].get("eyes_open"),
                eye_open_ratio=(
                    (eye_ratio[0], eye_ratio[1]) if eye_ratio is not None else None
                ),
                area_normalized_distance=(
                    float(target_detail["area_nd"])
                    if target_detail.get("area_nd") is not None
                    else None
                ),
                feature_normalized_distance=(
                    float(target_detail["feature_nd"])
                    if target_detail.get("feature_nd") is not None
                    else None
                ),
            )
        )
        if len(examples) >= limit:
            break
    return examples


def _build_bins(
    label_frames: list[dict[str, Any]],
    *,
    label: str,
    bin_width_deg: float,
    tolerance: float,
    center: tuple[float, float] | None,
    correction: TargetPoseCorrection | None,
) -> list[BinStats]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in label_frames:
        head_yaw = float(frame["obs"]["head"][0])
        grouped[int(math.floor(head_yaw / bin_width_deg))].append(frame)

    coverage: tuple[float, float] | None = None
    if correction is not None:
        coverage = (
            correction.points[0].head_yaw_deg,
            correction.points[-1].head_yaw_deg,
        )

    bins: list[BinStats] = []
    for index in sorted(grouped):
        members = grouped[index]
        expected = UNKNOWN_TARGET if label == NO_TARGET_LABEL else label
        accurate = sum(1 for frame in members if frame["cls"]["target"] == expected)
        unknown = sum(1 for frame in members if frame["cls"]["target"] == UNKNOWN_TARGET)

        in_area_percent: float | None = None
        correction_used_percent: float | None = None
        measured_bias: tuple[float, float] | None = None
        stored_offset: tuple[float, float] | None = None
        in_coverage: bool | None = None
        if label != NO_TARGET_LABEL:
            target_rows = [
                frame["targets"].get(label) for frame in members
            ]
            scored = [row for row in target_rows if row is not None]
            if scored:
                in_area_percent = _percent(
                    sum(1 for row in scored if float(row["area_nd"]) <= tolerance),
                    len(scored),
                )
                correction_used_percent = _percent(
                    sum(1 for row in scored if row.get("correction_applied")),
                    len(scored),
                )
            if center is not None:
                gazes = [
                    frame["gaze"]["feature"]
                    for frame in members
                    if frame["gaze"]["feature"] is not None
                ]
                if gazes:
                    matrix = np.asarray(gazes, dtype=np.float64)
                    measured_bias = (
                        float(np.median(matrix[:, 0])) - center[0],
                        float(np.median(matrix[:, 1])) - center[1],
                    )
            head_yaws = [float(frame["obs"]["head"][0]) for frame in members]
            median_head_yaw = float(np.median(np.asarray(head_yaws)))
            if correction is not None:
                stored_offset = correction.offset_for(median_head_yaw)
                assert coverage is not None
                in_coverage = coverage[0] <= median_head_yaw <= coverage[1]

        bins.append(
            BinStats(
                lower_deg=index * bin_width_deg,
                upper_deg=(index + 1) * bin_width_deg,
                frame_count=len(members),
                accuracy_percent=_percent(accurate, len(members)),
                unknown_percent=_percent(unknown, len(members)),
                in_area_percent=in_area_percent,
                correction_used_percent=correction_used_percent,
                measured_bias_yaw=measured_bias[0] if measured_bias else None,
                measured_bias_pitch=measured_bias[1] if measured_bias else None,
                stored_offset_yaw=stored_offset[0] if stored_offset else None,
                stored_offset_pitch=stored_offset[1] if stored_offset else None,
                in_correction_coverage=in_coverage,
            )
        )
    return bins


def _bin_warnings(
    label: str,
    bins: list[BinStats],
    correction: TargetPoseCorrection | None,
) -> list[str]:
    warnings: list[str] = []
    if label == NO_TARGET_LABEL:
        return warnings
    if correction is None:
        if any(abs(item.lower_deg) >= 10.0 or abs(item.upper_deg) > 10.0 for item in bins):
            warnings.append(
                "pose 보정 테이블이 없는 target인데 head yaw가 ±10° 밖까지 관측됨 - "
                "재등록(중앙 응시 스윕)으로 보정을 만들어야 함"
            )
        return warnings
    for item in bins:
        if item.frame_count < 5:
            continue
        if item.in_correction_coverage is False:
            warnings.append(
                f"head yaw {item.lower_deg:+.0f}..{item.upper_deg:+.0f}°: "
                "보정 테이블 커버리지 밖(끝점 상수 외삽)"
            )
        if (
            item.measured_bias_yaw is not None
            and item.stored_offset_yaw is not None
        ):
            delta_yaw = abs(item.measured_bias_yaw - item.stored_offset_yaw)
            delta_pitch = abs(
                (item.measured_bias_pitch or 0.0) - (item.stored_offset_pitch or 0.0)
            )
            if max(delta_yaw, delta_pitch) > CORRECTION_MISMATCH_WARNING_DEG:
                warnings.append(
                    f"head yaw {item.lower_deg:+.0f}..{item.upper_deg:+.0f}°: "
                    f"실측 편향 ({item.measured_bias_yaw:+.1f},"
                    f"{item.measured_bias_pitch:+.1f})° vs 저장된 보정 "
                    f"({item.stored_offset_yaw:+.1f},{item.stored_offset_pitch:+.1f})° - "
                    "등록 시점과 편향이 달라짐(세션 드리프트 또는 등록 데이터 오염)"
                )
    return warnings


def format_report(report: SessionReport) -> str:
    lines: list[str] = []
    lines.append(f"session: {report.path}")
    lines.append(
        f"frames: {report.total_frames} (labeled {report.labeled_frames}), "
        f"duration {report.duration_ms / 1000.0:.1f}s"
    )
    sources = ", ".join(
        f"{name} {count}" for name, count in sorted(report.gaze_sources.items())
    )
    lines.append(f"gaze sources: {sources}")

    for label in report.labels:
        lines.append("")
        title = label.display_name if label.display_name != label.label else label.label
        lines.append(f"== label: {title} ({label.label}) - {label.frame_count} frames ==")
        expected = UNKNOWN_TARGET if label.label == NO_TARGET_LABEL else label.label
        classified = ", ".join(
            f"{target} {_percent(count, label.frame_count):.0f}%"
            for target, count in sorted(
                label.classified.items(), key=lambda item: -item[1]
            )
        )
        lines.append(
            f"accuracy(={expected}): {label.accuracy_percent:.0f}% | classified: {classified}"
        )
        if label.no_gaze_frames:
            lines.append(f"no-gaze frames: {label.no_gaze_frames}")
        lines.append(
            f"eyes closed: {label.closed_eye_frames} "
            f"({_percent(label.closed_eye_frames, label.frame_count):.0f}%)"
        )
        source_reasons = ", ".join(
            f"{reason} x{count}"
            for reason, count in sorted(
                label.source_reasons.items(), key=lambda item: -item[1]
            )
        )
        if source_reasons:
            lines.append(f"gaze source reasons: {source_reasons}")
        if label.reject_reasons:
            reasons = ", ".join(
                f"{reason} x{count}"
                for reason, count in sorted(
                    label.reject_reasons.items(), key=lambda item: -item[1]
                )
            )
            lines.append(f"UNKNOWN reasons: {reasons}")
        lines.append(
            f"lock: {label.lock_completions} completions, "
            f"max dwell {label.max_dwell_ms} ms"
        )

        if label.bins:
            lines.append(
                f"{'head yaw bin':>14} {'n':>5} {'acc%':>6} {'UNK%':>6} "
                f"{'in-area%':>9} {'corr%':>6} {'bias(y,p)':>13} {'stored(y,p)':>13}"
            )
            for item in label.bins:
                bias = (
                    f"{item.measured_bias_yaw:+.1f},{item.measured_bias_pitch:+.1f}"
                    if item.measured_bias_yaw is not None
                    else "--"
                )
                stored = (
                    f"{item.stored_offset_yaw:+.1f},{item.stored_offset_pitch:+.1f}"
                    if item.stored_offset_yaw is not None
                    else "--"
                )
                in_area = (
                    f"{item.in_area_percent:.0f}"
                    if item.in_area_percent is not None
                    else "--"
                )
                corr = (
                    f"{item.correction_used_percent:.0f}"
                    if item.correction_used_percent is not None
                    else "--"
                )
                coverage_mark = "" if item.in_correction_coverage is not False else " *extrap"
                lines.append(
                    f"{item.lower_deg:+6.0f}..{item.upper_deg:+4.0f}° "
                    f"{item.frame_count:>5} {item.accuracy_percent:>5.0f} "
                    f"{item.unknown_percent:>6.0f} {in_area:>9} {corr:>6} "
                    f"{bias:>13} {stored:>13}{coverage_mark}"
                )
        axis_titles = {
            "head_pitch": "head pitch (deg)",
            "face_scale_ratio": "face scale / registered scale",
            "face_x": "face center x",
            "face_y": "face center y",
        }
        for axis, slices in label.axis_slices.items():
            lines.append(f"-- {axis_titles[axis]} --")
            lines.append(
                f"{'range':>15} {'n':>5} {'acc%':>6} {'UNK%':>6} "
                f"{'no-gaze%':>9} {'in-area%':>9}"
            )
            for slice_item in slices:
                in_area = (
                    f"{slice_item.in_area_percent:.0f}"
                    if slice_item.in_area_percent is not None
                    else "--"
                )
                lines.append(
                    f"{slice_item.lower:+7.2f}..{slice_item.upper:+6.2f} "
                    f"{slice_item.frame_count:>5} {slice_item.accuracy_percent:>6.0f} "
                    f"{slice_item.unknown_percent:>6.0f} "
                    f"{slice_item.no_gaze_percent:>9.0f} "
                    f"{in_area:>9}"
                )
        if label.failure_examples:
            lines.append("representative failures:")
            for failure in label.failure_examples:
                gaze = (
                    f"({failure.feature_gaze[0]:+.1f},{failure.feature_gaze[1]:+.1f})"
                    if failure.feature_gaze is not None
                    else "--"
                )
                face = (
                    f"({failure.face_center[0]:.2f},{failure.face_center[1]:.2f})"
                    if failure.face_center is not None
                    else "--"
                )
                scale = (
                    f"{failure.face_scale:.3f}"
                    if failure.face_scale is not None
                    else "--"
                )
                area = (
                    f"x{failure.area_normalized_distance:.2f}"
                    if failure.area_normalized_distance is not None
                    else "--"
                )
                lines.append(
                    f"  frame {failure.frame_id} t={failure.timestamp_ms} "
                    f"pred={failure.predicted} reason={failure.reason} "
                    f"source={failure.gaze_source} gaze={gaze} "
                    f"head=({failure.head[0]:+.1f},{failure.head[1]:+.1f},"
                    f"{failure.head[2]:+.1f}) face={face} scale={scale} area={area}"
                )
        for warning in label.warnings:
            lines.append(f"[warn] {warning}")
    return "\n".join(lines)

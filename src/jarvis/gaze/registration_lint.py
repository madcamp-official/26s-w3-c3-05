"""등록 직후 target 프로필의 품질 문제를 즉시 경고하는 린터.

2026-07-22 실측에서 겪은 두 실패는 모두 등록 완료 시점의 저장값만 봐도 잡을 수
있었다: (1) speaker의 1단계 스윕이 head yaw −25~−5°만 커버해 실사용 자세(정면)가
전부 상수 외삽이 됐고, (2) monitor의 +32° bin 오프셋이 clamp(±10°)에 걸린 채
저장돼 그 bin 데이터 자체가 오염돼 있었다. 이 린터는 그런 패턴을 등록 완료
직후(모니터링 앱 로그)와 저장된 profiles.json(`jarvis-gaze lint-profiles`)에서
경고한다.

집계·판정 로직을 복제하지 않는다 — 저장된 값의 형태만 검사하므로 등록 파이프라인이
바뀌어도 여기가 거짓 안심을 만들지 않는다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.gaze.config import GazeConfig

if TYPE_CHECKING:
    from jarvis.calibration.registry import TargetRecord

#: 이 head-yaw 범위(도)가 보정 테이블에 안 들어가면 실사용 자세가 외삽된다.
_NEUTRAL_HEAD_YAW_DEG = 0.0
#: 기준 자세(2단계)가 정면에서 이보다 멀면 hull 자체가 돌린 고개 기준이다.
_REFERENCE_YAW_WARNING_DEG = 15.0
#: 보정 커버리지가 이보다 좁으면 스윕이 부족했다고 본다.
_MINIMUM_COVERAGE_SPAN_DEG = 25.0
#: bin 표본이 이보다 적으면 그 점은 잡음일 가능성이 크다.
_SPARSE_BIN_SAMPLES = 30
#: clamp 판정 여유(부동소수 비교).
_CLAMP_EPSILON_DEG = 1e-6


def lint_target_record(record: "TargetRecord", config: GazeConfig) -> list[str]:
    """저장된 등록 결과에서 정확도를 깎을 소지가 있는 패턴을 경고 문자열로 반환한다."""
    warnings: list[str] = []

    area = record.area_profile
    if area is None:
        warnings.append("area profile 없음(구식 등록) - 재등록 필요")
    else:
        if not area.boundary_polygon:
            warnings.append("traced hull 없음 - 레거시 타원 폴백으로 동작, 재등록 권장")
        cap = config.registration_max_area_radius_deg
        if area.radius_yaw >= cap or area.radius_pitch >= cap:
            warnings.append(
                f"area 반경({area.radius_yaw:.1f}/{area.radius_pitch:.1f}°)이 "
                f"cap({cap:.1f}°)에 걸림 - 2단계 경계 추적이 퍼졌거나 물체가 너무 큼"
            )

    correction = record.pose_correction
    if correction is None:
        warnings.append(
            "pose 보정 없음 - 1단계에서 유효 bin이 2개 미만"
            "(스윕 범위가 좁았거나 bin IQR 초과). 고개를 돌린 자세는 보정 없이 판정됨"
        )
        return warnings

    coverage_low = correction.points[0].head_yaw_deg
    coverage_high = correction.points[-1].head_yaw_deg
    span = coverage_high - coverage_low
    if span < _MINIMUM_COVERAGE_SPAN_DEG:
        warnings.append(
            f"보정 커버리지가 head yaw {coverage_low:+.0f}~{coverage_high:+.0f}°"
            f"(폭 {span:.0f}°)뿐 - 그 밖 자세는 끝점 상수 외삽"
        )
    if not coverage_low <= _NEUTRAL_HEAD_YAW_DEG <= coverage_high:
        warnings.append(
            f"보정 커버리지({coverage_low:+.0f}~{coverage_high:+.0f}°)에 정면(0°)이 없음 - "
            "고개를 정면에 두고 눈짓만 하는 실사용 자세가 외삽됨"
        )

    reference = correction.reference_head_yaw_deg
    if reference is not None:
        if not coverage_low <= reference <= coverage_high:
            warnings.append(
                f"기준 자세(2단계 head yaw {reference:+.0f}°)가 보정 커버리지 밖 - "
                "기준 정규화 자체가 외삽값"
            )
        if abs(reference) > _REFERENCE_YAW_WARNING_DEG:
            warnings.append(
                f"hull을 고개 돌린 자세(head yaw {reference:+.0f}°)에서 추적함 - "
                "다른 자세에서 볼수록 보정 의존이 커짐"
            )

    clamp = config.pose_correction_max_offset_deg
    for point in correction.points:
        if (
            abs(point.offset_yaw_deg) >= clamp - _CLAMP_EPSILON_DEG
            or abs(point.offset_pitch_deg) >= clamp - _CLAMP_EPSILON_DEG
        ):
            warnings.append(
                f"head yaw {point.head_yaw_deg:+.0f}° bin 오프셋"
                f"({point.offset_yaw_deg:+.1f},{point.offset_pitch_deg:+.1f})이 "
                f"clamp(±{clamp:.0f}°)에 걸림 - 그 자세의 1단계 데이터 오염 의심"
            )
        if point.sample_count < _SPARSE_BIN_SAMPLES:
            warnings.append(
                f"head yaw {point.head_yaw_deg:+.0f}° bin 표본 {point.sample_count}개로 희박 - "
                "그 자세 오프셋은 잡음일 수 있음"
            )
    return warnings


def lint_records(
    records: list["TargetRecord"], config: GazeConfig
) -> dict[str, list[str]]:
    """target_id → 경고 목록. 경고 없는 target은 빈 목록으로 포함한다."""
    return {record.target_id: lint_target_record(record, config) for record in records}

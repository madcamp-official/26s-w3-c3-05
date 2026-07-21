"""Jester 클립을 오프라인으로 돌려 정규화된 landmark 시퀀스를 캐싱한다.

각 클립의 JPG 프레임을 순서대로 읽어 `jarvis.gesture_fusion.mediapipe_hands.
MediaPipeHandLandmarker`(런타임과 동일한 프로덕션 어댑터)로 처리한다 — 그래서
학습 데이터 전처리가 실제 추론 경로와 항상 같은 코드를 탄다(development-principles.md
7.3).

**미검출 프레임 처리(2026-07-20)**: 클립 내 손 미검출 프레임 비율이
`TrainingConfig.max_missing_frame_fraction`(기본 0.3)을 넘으면 클립 전체를
제외한다. 그 이하면 미검출 프레임도 그대로 클립에 남긴다 —
`HandFeatureExtractor.push()`가 실시간 추론과 동일하게 그 프레임을 추적 손실로
처리(reset + 0벡터)하고, `training/dataset.py`가 그 프레임의 loss target을
IGNORE_INDEX로 마스킹한다. "프레임 하나라도 실패하면 클립 전체 제외"였던 이전
규칙은 클립당 프레임 수가 많을수록 실패율을 지수적으로 증폭시켜(1-(1-p)^n) 프레임별
실패율이 몇 %만 돼도 대다수 클립이 통째로 버려지는 문제가 있었다. 조용히 버리지
않고 어느 쪽이든 매니페스트 CSV에 사유를 남긴다.

**Monotonic timestamp**: `detect_for_video`는 프레임 간 timestamp 단조 증가를
요구한다(mediapipe_hands.py 참조). 워커(프로세스)당 랜드마커 인스턴스 하나를
재사용하며, 합성 timestamp는 그 워커가 처리하는 모든 클립에 걸쳐 계속 증가한다
(클립 경계에서 리셋하지 않음) — 클립 경계 인식은 feature 조립 단계의
`HandFeatureExtractor.reset()`이 담당하므로 여기서는 문제없다.

**재개 가능**: 이미 캐싱된 clip_id는 건너뛴다 — 수 시간짜리 배치가 중단돼도
처음부터 다시 돌릴 필요가 없다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2

from jarvis.gesture_fusion.landmarks import HandObservation
from jarvis.gesture_fusion.mediapipe_hands import MediaPipeHandLandmarker
from jarvis.gesture_fusion.model_protocol import DEFAULT_GESTURE_LABELS
from training.config import DEFAULT_TRAINING_CONFIG
from training.data.clip_cache import observations_to_cached_clip, save_clip
from training.data.jester_manifest import JesterClipRef, Split, build_manifest

_FPS = 12.0  # Jester는 12fps로 추출된 프레임 시퀀스다(공식 문서).
_FRAME_INTERVAL_MS = 1000.0 / _FPS

# 미검출 비율 제한(`max_missing_frame_fraction`)을 적용하지 않는 라벨들. 둘 다
# "정의된 손동작이 없는" 카테고리라 애초에 손이 화면에 없거나 안 보이는 프레임이
# 구조적으로 많다 — 다른 클래스와 같은 기준을 적용하면 클래스 자체가 거의 사라진다
# (2026-07-20 실측: "none"은 validation 533개 중 17개=3.2%, "doing_other_things"는
# train 9592개 중 4454개=46.4%만 생존). `drumming_fingers`처럼 정의된 정적 손
# 모양은 손이 계속 보여야 정상이므로 포함하지 않는다 — 미검출이 많다면 그건 진짜
# 추출 실패다.
_MISSING_FRAME_LIMIT_EXEMPT_LABELS = frozenset({DEFAULT_GESTURE_LABELS[0], "doing_other_things"})


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """클립 하나의 처리 결과 — 매니페스트 CSV 한 행이 된다(no silent caps 원칙)."""

    clip_id: str
    split: Split
    status: str  # "ok" | "already_cached" | "no_frames" | "unreadable_frame" | "too_many_missed_frames"
    frame_count: int
    missing_frame_count: int = 0


def _list_frame_files(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.jpg"))


def _extract_clip(
    landmarker: MediaPipeHandLandmarker,
    ref: JesterClipRef,
    start_timestamp_ms: float,
    max_missing_frame_fraction: float,
) -> tuple[ExtractionResult, list[HandObservation], float]:
    """클립 하나를 처리한다. 반환값의 세 번째 요소는 다음 클립에 이어 쓸 timestamp다.

    프레임 하나라도 읽기 자체가 실패하면(파일 손상 등, MediaPipe 검출 실패와는
    다른 문제) 그 시점에서 클립 전체를 제외한다 — 검출 실패와 달리 이후 프레임을
    이어 읽을 신뢰할 수 있는 근거가 없다. 손 미검출은 다르다: 클립 끝까지 계속
    처리하고, 미검출 비율이 `max_missing_frame_fraction`을 넘을 때만 제외한다.

    **`_MISSING_FRAME_LIMIT_EXEMPT_LABELS`(`none`·`doing_other_things`) 클립은 이
    미검출 비율 제한을 적용하지 않는다.** 둘 다 정의상 손이 화면에 없거나 정의된
    동작을 안 하는 장면이 많아 MediaPipe 검출 실패율이 다른 클래스보다 구조적으로
    높다(2026-07-20 실측: 이 필터로 `none`은 validation 533개 중 17개=3.2%,
    `doing_other_things`는 train 9592개 중 4454개=46.4%만 생존 — 다른 클래스는
    25~75% 생존). 미검출 프레임은 다른 클립과 동일하게 `hand_detected=False`로
    캐시에 남아, 학습 시 `dataset.py`의 IGNORE_INDEX 마스킹이 그대로 적용된다 —
    배제하지 않고 신호 그대로 남길 뿐이다.
    """
    frame_files = _list_frame_files(ref.frames_dir)
    if not frame_files:
        return ExtractionResult(ref.clip_id, ref.split, "no_frames", 0), [], start_timestamp_ms

    observations: list[HandObservation] = []
    timestamp_ms = start_timestamp_ms
    for frame_id, frame_path in enumerate(frame_files):
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            result = ExtractionResult(ref.clip_id, ref.split, "unreadable_frame", frame_id)
            return result, [], timestamp_ms + _FRAME_INTERVAL_MS
        # mediapipe_hands.py 색상 규약: process()는 RGB를 기대하는데 cv2는 BGR로 읽는다.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        observation = landmarker.process(rgb, int(timestamp_ms), frame_id)
        timestamp_ms += _FRAME_INTERVAL_MS
        observations.append(observation)

    missing = sum(1 for o in observations if not o.hand_detected)
    if (
        ref.our_label not in _MISSING_FRAME_LIMIT_EXEMPT_LABELS
        and missing / len(observations) > max_missing_frame_fraction
    ):
        result = ExtractionResult(
            ref.clip_id, ref.split, "too_many_missed_frames", len(frame_files), missing
        )
        return result, [], timestamp_ms

    result = ExtractionResult(ref.clip_id, ref.split, "ok", len(frame_files), missing)
    return result, observations, timestamp_ms


def _process_shard(
    refs: list[JesterClipRef],
    model_path: Path,
    cache_dir: Path,
    max_missing_frame_fraction: float,
) -> list[ExtractionResult]:
    """워커 프로세스 하나가 맡은 클립들을 순서대로 처리한다(랜드마커 인스턴스 재사용)."""
    results: list[ExtractionResult] = []
    timestamp_ms = 0.0
    with MediaPipeHandLandmarker(model_path) as landmarker:
        for ref in refs:
            out_path = cache_dir / ref.split / f"{ref.clip_id}.npz"
            if out_path.exists():
                results.append(ExtractionResult(ref.clip_id, ref.split, "already_cached", 0))
                continue
            result, observations, timestamp_ms = _extract_clip(
                landmarker, ref, timestamp_ms, max_missing_frame_fraction
            )
            if result.status == "ok":
                clip = observations_to_cached_clip(observations, ref.our_label, ref.clip_id)
                save_clip(out_path, clip)
            results.append(result)
    return results


def _write_manifest(cache_dir: Path, results: list[ExtractionResult]) -> None:
    manifest_path = cache_dir / "jester_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if fh.tell() == 0:
            writer.writerow(["clip_id", "split", "status", "frame_count", "missing_frame_count"])
        for r in results:
            writer.writerow([r.clip_id, r.split, r.status, r.frame_count, r.missing_frame_count])


def run(
    jester_dir: Path,
    model_path: Path,
    cache_dir: Path,
    splits: list[Split],
    workers: int,
    limit: int | None,
    max_missing_frame_fraction: float = DEFAULT_TRAINING_CONFIG.max_missing_frame_fraction,
) -> list[ExtractionResult]:
    all_refs: list[JesterClipRef] = []
    for split in splits:
        refs = build_manifest(jester_dir, split)
        if limit is not None:
            refs = refs[:limit]
        all_refs.extend(refs)

    if not all_refs:
        return []

    shards: list[list[JesterClipRef]] = [all_refs[i::workers] for i in range(workers)]
    shards = [s for s in shards if s]

    all_results: list[ExtractionResult] = []
    if workers <= 1:
        all_results.extend(
            _process_shard(all_refs, model_path, cache_dir, max_missing_frame_fraction)
        )
    else:
        with ProcessPoolExecutor(max_workers=len(shards)) as pool:
            futures = [
                pool.submit(_process_shard, shard, model_path, cache_dir, max_missing_frame_fraction)
                for shard in shards
            ]
            for future in as_completed(futures):
                all_results.extend(future.result())

    _write_manifest(cache_dir, all_results)
    return all_results


def _summarize(results: list[ExtractionResult]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    total = len(results)
    lines = [f"{total}개 클립 처리"]
    for status, count in sorted(counts.items()):
        lines.append(f"  {status}: {count}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jester-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.jester_dir)
    parser.add_argument("--model", type=Path, default=DEFAULT_TRAINING_CONFIG.models_dir / "hand_landmarker.task")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.cache_dir / "jester")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["train", "validation"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="split당 처리할 클립 수 상한(스모크 테스트용)")
    parser.add_argument(
        "--max-missing-frame-fraction",
        type=float,
        default=DEFAULT_TRAINING_CONFIG.max_missing_frame_fraction,
        help="클립 내 미검출 프레임 비율이 이 값을 넘으면 클립 전체를 제외한다(기본 %(default)s)",
    )
    args = parser.parse_args(argv)

    results = run(
        jester_dir=args.jester_dir,
        model_path=args.model,
        cache_dir=args.cache_dir,
        splits=list(args.splits),
        workers=args.workers,
        limit=args.limit,
        max_missing_frame_fraction=args.max_missing_frame_fraction,
    )
    print(_summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

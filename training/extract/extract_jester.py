"""Jester 클립을 오프라인으로 돌려 정규화된 landmark 시퀀스를 캐싱한다.

각 클립의 JPG 프레임을 순서대로 읽어 `jarvis.gesture_fusion.mediapipe_hands.
MediaPipeHandLandmarker`(런타임과 동일한 프로덕션 어댑터)로 처리한다 — 그래서
학습 데이터 전처리가 실제 추론 경로와 항상 같은 코드를 탄다(development-principles.md
7.3). 손 미검출 프레임이 하나라도 있으면 클립 전체를 학습셋에서 제외한다(학습
파이프라인 인터뷰 결정) — 조용히 버리지 않고 매니페스트 CSV에 사유를 남긴다.

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
from training.config import DEFAULT_TRAINING_CONFIG
from training.data.clip_cache import observations_to_cached_clip, save_clip
from training.data.jester_manifest import JesterClipRef, Split, build_manifest

_FPS = 12.0  # Jester는 12fps로 추출된 프레임 시퀀스다(공식 문서).
_FRAME_INTERVAL_MS = 1000.0 / _FPS

# 트리밍 후 유지할 최소 프레임 수. causal 미분(velocity/acceleration)이 성립하려면
# 최소 2프레임이 필요하고, 이보다 훨씬 짧은 클립은 제스처로서 정보가 빈약하므로
# 기본값을 넉넉히 둔다(모델 receptive_field 29의 참고값보다는 관대하게 — 짧은
# 클립도 zero-padding으로 학습 가능하되 극단적으로 짧은 것만 거른다). --min-frames로 조정.
_DEFAULT_MIN_FRAMES = 8


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """클립 하나의 처리 결과 — 매니페스트 CSV 한 행이 된다(no silent caps 원칙)."""

    clip_id: str
    split: Split
    # "ok" | "already_cached" | "no_frames" | "unreadable_frame"
    # | "no_hand"(전 프레임 미검출) | "interior_gap"(앞뒤 트리밍 후에도 내부에 미검출)
    # | "too_short"(트리밍 후 유지 프레임 < min_frames)
    status: str
    frame_count: int  # 원본 프레임 수
    kept_frames: int = 0  # 앞뒤 트리밍 후 실제로 캐싱한 프레임 수(status=="ok"일 때만 의미)


def _list_frame_files(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.jpg"))


def _extract_clip(
    landmarker: MediaPipeHandLandmarker,
    ref: JesterClipRef,
    start_timestamp_ms: float,
    min_frames: int,
) -> tuple[ExtractionResult, list[HandObservation], float]:
    """클립 하나를 처리한다. 반환값의 세 번째 요소는 다음 클립에 이어 쓸 timestamp다.

    **가장자리 트리밍 정책**(2026-07-19 검출 수율 문제로 완화): Jester 클립은 손이
    화면에 들어오고 나가는 시작/끝 프레임에서 미검출이 잦다. 원래는 프레임 하나라도
    미검출이면 클립 전체를 버렸으나(전-프레임 검출률 ~58%에서 클립 생존율이 붕괴),
    이제는 앞뒤의 연속된 미검출 프레임만 잘라내고 **내부가 전부 검출된** 클립만
    채택한다. 이렇게 유지된 클립은 여전히 100% 검출 프레임만 담으므로 downstream
    (`training/dataset.py`·`training/augment.py`의 "전부 검출" 가정)은 손대지 않아도
    된다 — 좌표를 지어내지 않는다는 원칙(development-principles.md 1·2절)도 유지된다.
    내부에 미검출이 남는 클립("interior_gap")은 조용히 채우지 않고 매니페스트에 남긴다.
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

    result, core = _trim_and_classify(observations, ref, min_frames)
    return result, core, timestamp_ms


def _trim_and_classify(
    observations: list[HandObservation],
    ref: JesterClipRef,
    min_frames: int,
) -> tuple[ExtractionResult, list[HandObservation]]:
    """프레임별 관측값 시퀀스를 가장자리 트리밍 정책으로 분류한다(MediaPipe 무의존, 단위 테스트 가능).

    앞뒤의 연속 미검출 프레임만 잘라내고, 내부가 전부 검출된 클립만 "ok"로 채택한다
    — `_extract_clip` docstring의 정책 참조. 채택 시 반환하는 관측값 리스트(core)는
    전부 `hand_detected=True`이므로 downstream의 "전부 검출" 가정을 그대로 만족한다.
    """
    total = len(observations)
    detected = [obs.hand_detected for obs in observations]
    if not any(detected):
        return ExtractionResult(ref.clip_id, ref.split, "no_hand", total), []

    lo = detected.index(True)
    hi = len(detected) - 1 - detected[::-1].index(True)
    core = observations[lo : hi + 1]

    if not all(obs.hand_detected for obs in core):
        # 트리밍 후에도 내부에 미검출이 남는다 — 좌표를 지어내 채우지 않고 제외한다.
        return ExtractionResult(ref.clip_id, ref.split, "interior_gap", total), []
    if len(core) < min_frames:
        return ExtractionResult(ref.clip_id, ref.split, "too_short", total), []

    return ExtractionResult(ref.clip_id, ref.split, "ok", total, kept_frames=len(core)), core


def _process_shard(
    refs: list[JesterClipRef],
    model_path: Path,
    cache_dir: Path,
    min_frames: int,
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
                landmarker, ref, timestamp_ms, min_frames
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
            writer.writerow(["clip_id", "split", "status", "frame_count", "kept_frames"])
        for r in results:
            writer.writerow([r.clip_id, r.split, r.status, r.frame_count, r.kept_frames])


def run(
    jester_dir: Path,
    model_path: Path,
    cache_dir: Path,
    splits: list[Split],
    workers: int,
    limit: int | None,
    min_frames: int = _DEFAULT_MIN_FRAMES,
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
        all_results.extend(_process_shard(all_refs, model_path, cache_dir, min_frames))
    else:
        with ProcessPoolExecutor(max_workers=len(shards)) as pool:
            futures = [
                pool.submit(_process_shard, shard, model_path, cache_dir, min_frames)
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
        "--min-frames",
        type=int,
        default=_DEFAULT_MIN_FRAMES,
        help="앞뒤 미검출 프레임 트리밍 후 유지할 최소 프레임 수(이보다 짧으면 too_short로 제외)",
    )
    args = parser.parse_args(argv)

    results = run(
        jester_dir=args.jester_dir,
        model_path=args.model,
        cache_dir=args.cache_dir,
        splits=list(args.splits),
        workers=args.workers,
        min_frames=args.min_frames,
        limit=args.limit,
    )
    print(_summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

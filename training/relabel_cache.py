"""캐시된 클립의 `gesture_label`을 현재 `JESTER_TO_OUR_LABEL` 매핑으로 갱신한다.

캐시(`clip_cache.CachedClip`)는 추출 시점의 **파생 라벨 문자열**을 그대로 담는다.
그래서 라벨 매핑(`training/data/jester_labels.py`)이나 라벨 집합
(`DEFAULT_GESTURE_LABELS`)을 바꾸면 캐시에 옛 라벨이 남아 `ClipDataset`이
`KeyError`를 낸다. 랜드마크 자체는 매핑과 무관하므로 몇 시간짜리 MediaPipe 재추출은
필요 없고, clip_id로 Jester 원본 CSV를 되짚어 라벨만 다시 쓰면 된다.

원본 CSV가 진실의 출처이므로 이 스크립트는 몇 번을 돌려도 같은 결과가 나오고
(idempotent), 매핑을 되돌린 뒤 다시 돌리면 이전 라벨 구성으로 복구된다.

    python -m training.relabel_cache                     # 기본 캐시 경로
    python -m training.relabel_cache --dry-run           # 바뀔 개수만 확인
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from training.config import DEFAULT_TRAINING_CONFIG
from training.data.clip_cache import load_clip, save_clip
from training.data.jester_labels import JESTER_TO_OUR_LABEL, validate_mapping

_SPLITS = ("train", "validation")


def _read_jester_labels(jester_dir: Path) -> dict[str, str]:
    """clip_id -> Jester 원본 클래스명. train·validation CSV를 합쳐 읽는다."""
    mapping: dict[str, str] = {}
    for split in _SPLITS:
        csv_path = jester_dir / "annotations" / f"jester-v1-{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Jester split CSV not found at {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh, delimiter=";"):
                if len(row) == 2:
                    mapping[row[0].strip()] = row[1].strip()
    return mapping


def _relabel_one(args: tuple[Path, str, bool]) -> int:
    """클립 하나를 갱신한다. 반환값은 실제로 바뀐 개수(0 또는 1)."""
    path, target_label, dry_run = args
    clip = load_clip(path)
    if clip.gesture_label == target_label:
        return 0
    if not dry_run:
        save_clip(path, replace(clip, gesture_label=target_label))
    return 1


def run(cache_dir: Path, jester_dir: Path, workers: int, dry_run: bool) -> tuple[int, int]:
    validate_mapping()
    jester_labels = _read_jester_labels(jester_dir)

    jobs: list[tuple[Path, str, bool]] = []
    missing: list[str] = []
    for split in _SPLITS:
        for path in sorted((cache_dir / split).glob("*.npz")):
            jester_label = jester_labels.get(path.stem)
            if jester_label is None:
                missing.append(path.stem)
                continue
            our_label = JESTER_TO_OUR_LABEL[jester_label]
            if our_label is None:
                # 현재 매핑이 학습에서 제외한 클래스 — 라벨을 지어내지 않고 건너뛴다.
                missing.append(path.stem)
                continue
            jobs.append((path, our_label, dry_run))

    if missing:
        print(f"경고: 원본 CSV에서 라벨을 찾지 못했거나 제외된 클립 {len(missing)}개는 건너뜀")

    if workers <= 1:
        changed = sum(_relabel_one(j) for j in jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            changed = sum(pool.map(_relabel_one, jobs, chunksize=256))
    return changed, len(jobs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.cache_dir / "jester")
    parser.add_argument("--jester-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.jester_dir)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="쓰지 않고 바뀔 개수만 보고")
    args = parser.parse_args(argv)

    changed, total = run(args.cache_dir, args.jester_dir, args.workers, args.dry_run)
    verb = "바뀔 예정" if args.dry_run else "갱신됨"
    print(f"{total}개 클립 중 {changed}개 {verb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

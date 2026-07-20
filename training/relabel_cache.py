"""캐시된 클립의 `gesture_label`을 현재 `JESTER_TO_OUR_LABEL` 매핑으로 갱신한다.

캐시(`clip_cache.CachedClip`)는 추출 시점의 **파생 라벨 문자열**을 그대로 담는다.
그래서 라벨 매핑(`training/data/jester_labels.py`)이나 라벨 집합
(`DEFAULT_GESTURE_LABELS`)을 바꾸면 캐시에 옛 라벨이 남아 `ClipDataset`이
`KeyError`를 낸다. 랜드마크 자체는 매핑과 무관하므로 몇 시간짜리 MediaPipe 재추출은
필요 없고, clip_id로 Jester 원본 CSV를 되짚어 라벨만 다시 쓰면 된다.

원본 CSV가 진실의 출처이므로 이 스크립트는 몇 번을 돌려도 같은 결과가 나오고
(idempotent), 매핑을 되돌린 뒤 다시 돌리면 이전 라벨 구성으로 복구된다.

**제외된 클래스(현재 매핑에서 `None`)의 npz는 지우지 않고 `--cache-dir` 옆의
`jester_excluded/`로 옮긴다** — `ClipDataset`이 `<root>/**/*.npz`로 재귀 수집하므로
`cache_dir` 하위에 두면(하위 폴더라도) 계속 주워진다. 삭제하지 않는 이유는 이미
몇 시간짜리 MediaPipe 추출을 거친 데이터라서다. 나중에 그 클래스를 다시 포함하기로
하면: (1) 매핑에서 다시 non-None으로 바꾸고 (2) `training/cache/jester_excluded/`의
해당 파일들을 `training/cache/jester/{train,validation}/`로 직접 옮긴 뒤 (3) 이
스크립트를 평소대로 돌려 라벨을 갱신한다 — 재추출은 필요 없다.

    python -m training.relabel_cache                     # 기본 캐시 경로
    python -m training.relabel_cache --dry-run           # 바뀔/옮겨질 개수만 확인
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


def _move_one(args: tuple[Path, Path, bool]) -> None:
    """제외된 클립 파일을 `jester_excluded/<split>/`로 옮긴다(삭제하지 않음)."""
    src, dest, dry_run = args
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.replace(dest)


def run(
    cache_dir: Path, jester_dir: Path, workers: int, dry_run: bool
) -> tuple[int, int, int]:
    validate_mapping()
    jester_labels = _read_jester_labels(jester_dir)
    excluded_root = cache_dir.parent / "jester_excluded"

    relabel_jobs: list[tuple[Path, str, bool]] = []
    move_jobs: list[tuple[Path, Path, bool]] = []
    unmapped: list[str] = []
    for split in _SPLITS:
        for path in sorted((cache_dir / split).glob("*.npz")):
            jester_label = jester_labels.get(path.stem)
            if jester_label is None:
                # 원본 CSV에 이 clip_id가 없다 — 매핑 문제가 아니라 데이터 자체의
                # 이상이므로 옮기지 않고 원인 파악이 가능하도록 보고만 한다.
                unmapped.append(path.stem)
                continue
            our_label = JESTER_TO_OUR_LABEL[jester_label]
            if our_label is None:
                # 현재 매핑이 학습에서 제외한 클래스 — 지우지 않고 옆으로 옮긴다
                # (ClipDataset의 재귀 glob이 cache_dir 하위는 계속 주우므로 밖으로 뺀다).
                move_jobs.append((path, excluded_root / split / path.name, dry_run))
                continue
            relabel_jobs.append((path, our_label, dry_run))

    if unmapped:
        print(f"경고: 원본 CSV에서 clip_id를 찾지 못한 클립 {len(unmapped)}개는 그대로 둠")

    if workers <= 1:
        changed = sum(_relabel_one(j) for j in relabel_jobs)
        for j in move_jobs:
            _move_one(j)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            changed = sum(pool.map(_relabel_one, relabel_jobs, chunksize=256))
            list(pool.map(_move_one, move_jobs, chunksize=256))
    return changed, len(relabel_jobs), len(move_jobs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.cache_dir / "jester")
    parser.add_argument("--jester-dir", type=Path, default=DEFAULT_TRAINING_CONFIG.jester_dir)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="쓰지 않고 바뀔/옮겨질 개수만 보고")
    args = parser.parse_args(argv)

    changed, total, moved = run(args.cache_dir, args.jester_dir, args.workers, args.dry_run)
    verb = "바뀔 예정" if args.dry_run else "갱신됨"
    move_verb = "옮겨질 예정" if args.dry_run else "jester_excluded/로 옮겨짐"
    print(f"{total}개 클립 중 {changed}개 {verb}")
    print(f"현재 매핑에서 제외된 클립 {moved}개 {move_verb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

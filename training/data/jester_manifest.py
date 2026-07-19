"""Jester 공식 annotation CSV를 읽어 (클립 경로, 우리 라벨, split) 목록을 만든다.

Jester는 subject/person ID를 공개하지 않으므로(학습 파이프라인 인터뷰 기록,
documents/decisions.md), 사람 단위 split이 불가능하다 — 대신 Jester가 이미 나눠준
공식 `jester-v1-train.csv`/`jester-v1-validation.csv`를 그대로 쓴다(벤치마크 관례와도
일치). 공식 `test` split은 라벨이 공개되지 않아(비공개 리더보드용) 여기서 다루지 않는다.

CSV 포맷: `video_id;label` (세미콜론 구분, 헤더 없음) — `jester-v1-labels.csv`는
27개 라벨을 한 줄씩 담는다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from training.data.jester_labels import JESTER_TO_OUR_LABEL, validate_mapping

Split = Literal["train", "validation"]


@dataclass(frozen=True, slots=True)
class JesterClipRef:
    """학습셋의 클립 하나를 가리키는 참조(프레임을 로드하지 않음, 경로만)."""

    clip_id: str
    frames_dir: Path
    our_label: str
    split: Split


def _read_split_csv(csv_path: Path) -> list[tuple[str, str]]:
    """`video_id;label` 두 열짜리 CSV를 읽는다. 헤더 없음."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Jester split CSV not found at {csv_path}")
    rows: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter=";"):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"malformed row in {csv_path}: {row!r}")
            video_id, label = row
            rows.append((video_id.strip(), label.strip()))
    return rows


def build_manifest(jester_dir: Path, split: Split) -> list[JesterClipRef]:
    """`jester_dir/annotations/jester-v1-{split}.csv`를 읽어 우리 라벨로 매핑된 클립만 남긴다.

    `JESTER_TO_OUR_LABEL`에서 `None`으로 매핑된(아직 결정되지 않은) 클래스의 클립은
    조용히 버리지 않고 이 함수 호출자가 카운트를 볼 수 있게 반환 리스트 자체가 그
    결과다 — 얼마나 빠졌는지는 `extract_jester.py`가 매니페스트 CSV에 남긴다.
    """
    validate_mapping()
    csv_name = f"jester-v1-{split}.csv"
    rows = _read_split_csv(jester_dir / "annotations" / csv_name)

    refs: list[JesterClipRef] = []
    for video_id, jester_label in rows:
        our_label = JESTER_TO_OUR_LABEL.get(jester_label)
        if our_label is None:
            continue
        frames_dir = jester_dir / "20bn-jester-v1" / video_id
        refs.append(
            JesterClipRef(
                clip_id=video_id,
                frames_dir=frames_dir,
                our_label=our_label,
                split=split,
            )
        )
    return refs

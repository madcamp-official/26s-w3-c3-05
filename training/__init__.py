"""Gesture TCN 오프라인 학습 파이프라인.

`src/jarvis`(런타임 패키지) 밖에 의도적으로 둔다 — DataLoader·옵티마이저·체크포인트
저장 등은 런타임 추론 경로에 전혀 필요 없는 무거운 개발 도구다. `pyproject.toml`의
mypy `packages = ["jarvis"]` 범위에 들지 않으므로 strict 게이트 대상이 아니다
(documents/decisions.md 2026-07-19, 학습 파이프라인 코드 위치 결정 참조).

이 패키지는 `jarvis.gesture_fusion`의 순수 함수(`normalize_hand`,
`HandFeatureExtractor`, `GestureConfig`)를 그대로 재사용해, 학습 시 전처리가
추론 시 전처리와 항상 같은 코드 경로를 타도록 한다(development-principles.md
7.3: 학습·추론 전처리 일관성).
"""

from __future__ import annotations

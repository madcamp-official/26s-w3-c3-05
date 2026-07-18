# Plan — 일별 진행 트래킹

Day별 큰 계획은 이미 [README.md](../README.md) 16장 개발 일정에 있다.
이 문서는 그 계획 대비 실제로 무엇을 했는지, 무엇이 지연되고 있는지를 매일 짧게 기록한다.

## Day 1

- Gaze: Day 1~5 범위(얼굴·홍채 추적 어댑터, gaze vector 합성, calibration, target classifier,
  Unknown rejection, Gaze Lock 상태 머신, Target Selection Accuracy 평가)를 한 세션에 구현.
  `src/jarvis/gaze/`, `src/jarvis/calibration/` 전체 + unit/contract 테스트 54개, ruff·mypy
  strict 통과. 실제 카메라로 검증하지 않았고 head yaw/pitch/roll 부호 규약은 Day 1 실기기
  통합 때 확인 필요(landmarks.py의 `rotation_matrix_to_euler_deg` 주석 참고).
- Gesture·Fusion:
- Runtime·Protocol:

## Day 2

- Gaze:
- Gesture·Fusion:
- Runtime·Protocol:

## Day 3

- Gaze:
- Gesture·Fusion:
- Runtime·Protocol:

## Day 4

- Gaze:
- Gesture·Fusion:
- Runtime·Protocol:

## Day 5

- Gaze:
- Gesture·Fusion:
- Runtime·Protocol:

## Day 6

- Gaze:
- Gesture·Fusion:
- Runtime·Protocol:

## Day 7

- Gaze:
- Gesture·Fusion:
- Runtime·Protocol:

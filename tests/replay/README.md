# Trace replay and benchmark

동일한 timestamp와 frame sequence를 재생해 Target Selection Accuracy, Gesture Event Recall,
Wrong Actuation Rate와 p95 latency를 재현 가능하게 측정한다.

`test_gaze_targeting_replay.py`는 calibration부터 target classifier, `UNKNOWN` 거부,
Target Selection Accuracy 계산까지 이어지는 합성 yaw trace다. 실제 카메라 성능 결과가
아니며, 기능 간 계약과 회귀만 검증한다.

# Gaze Targeting

담당 범위는 `documents/gaze.md`를 따른다. 얼굴·홍채 추적, 사용자 calibration,
target classifier, `UNKNOWN` rejection, smoothing과 Gaze Lock을 이 패키지 안에서 구현한다.

외부로 내보내는 값은 `jarvis.contracts.TargetEstimate`만 사용한다. Gesture 또는 adapter의
내부 구현을 직접 import하지 않는다.


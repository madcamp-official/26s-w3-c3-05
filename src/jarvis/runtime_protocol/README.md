# Runtime & Device Protocol

담당 범위는 `documents/runtime-protocol.md`를 따른다. 카메라 캡처, 공통 monotonic timestamp,
bounded queue, capability 검증, timeout·ACK·deduplication과 latency 측정을 담당한다.

- `capture/`: 한 번 캡처한 프레임을 Gaze와 Gesture에 배포
- `protocol/`: Intent 검증, Command 생성, TTL과 deduplication
- `adapters/`: Windows 및 SmartThings의 실제 실행 경계
- `telemetry/`: 상태 전이, 오류, latency 측정


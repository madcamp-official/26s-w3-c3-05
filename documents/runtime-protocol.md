# Runtime & Device Protocol — 담당: 3인

README [10장 핵심 기능 4](../README.md), [11장 전자기기 연결 방법](../README.md)의 구현 설계/진행 상황을 기록하는 문서.
다른 모듈과 주고받는 데이터 포맷은 여기가 아니라 [interface-contract.md](interface-contract.md)에 정의한다.

## 담당 범위 (README 12장)

- 카메라 멀티스트림 pipeline
- timestamp 동기화
- bounded queue
- device capability model
- Windows adapter
- SmartThings adapter
- 명령 timeout·ACK·deduplication
- End-to-End latency 측정

## 개발 환경

- Python 3.12 로컬 venv(`.venv`, gitignore됨) + `pip install -e ".[dev]"`. pyproject는 3.11+ 요구.
- 검증: `.venv/Scripts/python -m pytest -q`, `... -m mypy`(strict), `... -m ruff check`. 세 검사 모두 통과가 DoD.
- `src/jarvis/py.typed` 추가(PEP 561) — mypy가 설치된 패키지를 타입체크하도록.

## 구현 순서 (청크)

1. **capture/** — 클럭·프레임·bounded queue·fan-out (✅ 완료)
2. protocol/ — capability 검증 + Intent→Command + TTL + dedup (다음)
3. adapters/ — Windows(Win32) + SmartThings
4. telemetry/ — 상태 전이·latency 측정

## 설계 노트

### capture/ (청크 1)

- `clock.py` `RuntimeClock`: 단일 monotonic 클럭. `stamp()`이 `FrameStamp(timestamp_ms, frame_id)`를 발급. `frame_id`는 프로세스 내 gapless 증가. `time_source` 주입으로 테스트에서 시간 결정적 제어. 계약 공통 규칙(모든 프레임 메시지가 이 클럭의 stamp를 계승)을 여기서 실현.
- `frame.py` `Frame[ImageT]`: stamp + 불투명 이미지 payload. 코어가 이미지 라이브러리에 의존하지 않도록 제네릭. 계약 밖(runtime_protocol 내부 타입).
- `queue.py` `BoundedLatestQueue`: bounded + drop-oldest(latest-frame) 정책, drop 카운트 노출. 실시간 소비자가 stale 프레임 backlog를 재생하지 않게(원칙 5.2). thread-safe(1 producer/1 consumer).
- `source.py` `FrameSource`(Protocol) + `OpenCVCameraSource`: 카메라는 하드웨어 IO 경계로 분리, `cv2` lazy import(vision extra 없이도 코어·테스트 동작). 성공 위조 없음(원칙 1.1).
- `pipeline.py` `CapturePipeline`: 1회 캡처 → 1회 stamp → 모든 소비자에 **동일 Frame** 배포(원칙 5.1). `run_once()`가 결정적 단위(스레드 없이 테스트), `start()/stop()`은 배경 스레드 wrapper.
- 테스트 17개: 클럭 monotonic/gapless, queue drop-oldest·카운트, 파이프라인 fan-out 동일 stamp·소비자별 독립 backpressure.

## 진행 상황

- [x] 청크 1: capture/ 구현 + 단위 테스트 17개 (pytest/mypy/ruff 통과)
- [ ] 청크 2: protocol/
- [ ] 청크 3: adapters/
- [ ] 청크 4: telemetry/

## 이슈 / 의사결정 필요 사항

- (미정) bounded queue 기본 capacity 값 — 실제 카메라 fps·소비자 처리속도 측정 후 configs로 확정 예정.
- (있으면 [decisions.md](decisions.md)로 옮기기)

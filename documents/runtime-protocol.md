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

### protocol/ (청크 2)

안전 실행 코어. 실제 기기는 건드리지 않고, Intent가 command이 될 자격이 있는지 판정한다(원칙 2·4). 장애·불확실 시 기본은 미실행.

- `capability.py`: Device Capability Model(README 10장). `BooleanCapability`·`NumberCapability`(min/max/step) + `DeviceRegistry`. `validate_request()`가 operation 지원 여부·값 type/range/step을 검증. 상대 연산(increment/decrement)은 delta만 검증하고, 결과 절대값의 [min,max] clamp는 기기 현재 상태가 필요하므로 adapter 몫으로 미룸.
- `lifecycle.py`: `CommandState`(VALIDATED→DISPATCHED→ACKNOWLEDGED→VERIFIED, 실패 REJECTED/EXPIRED/FAILED/UNVERIFIED) + 합법 전이 테이블. terminal 상태는 나가는 edge 없음.
- `ledger.py`: `CommandLedger` — command_id 단위 dedup(중복 register 거부) + 합법 전이만 허용(원칙 3).
- `engine.py`: `ProtocolEngine.submit(intent)` → `Accepted(command)` | `Rejected(reason, detail)`. `dispatch_guard()`가 dispatch 직전 TTL 재검증(원칙 4) → DISPATCHED 또는 EXPIRED. ack/verify/fail/unverify 전이 제공.
- command_id는 `cmd-{intent_id}`로 결정적 생성 → 같은 intent 재시도가 같은 id로 collapse(idempotency, 원칙 3).
- **dedup 경쟁 조건 수정(리뷰 후)**: `submit()`이 `seen()` 선검사 후 `register()`하던 2단계를 없애고, 원자적인 `register()` 하나에만 의존하도록 변경. `DuplicateCommandError`를 잡아 `Rejected(DUPLICATE)`로 반환한다. 동시 submit 시 예외가 새어나가던 버그 해소.
- 테스트 35개: capability 값 검증, lifecycle 전이, ledger dedup, engine 6종 reject·TTL·dedup·전체 성공 경로.

### capture/ (청크 1)

- `clock.py` `RuntimeClock`: 단일 monotonic 클럭. `stamp()`이 `FrameStamp(timestamp_ms, frame_id)`를 발급. `frame_id`는 프로세스 내 gapless 증가. `time_source` 주입으로 테스트에서 시간 결정적 제어. 계약 공통 규칙(모든 프레임 메시지가 이 클럭의 stamp를 계승)을 여기서 실현.
- `frame.py` `Frame[ImageT]`: stamp + 불투명 이미지 payload. 코어가 이미지 라이브러리에 의존하지 않도록 제네릭. 계약 밖(runtime_protocol 내부 타입).
- `queue.py` `BoundedLatestQueue`: bounded + drop-oldest(latest-frame) 정책, drop 카운트 노출. 실시간 소비자가 stale 프레임 backlog를 재생하지 않게(원칙 5.2). thread-safe(1 producer/1 consumer).
- `source.py` `FrameSource`(Protocol) + `OpenCVCameraSource`: 카메라는 하드웨어 IO 경계로 분리, `cv2` lazy import(vision extra 없이도 코어·테스트 동작). 성공 위조 없음(원칙 1.1).
- `pipeline.py` `CapturePipeline`: 1회 캡처 → 1회 stamp → 모든 소비자에 **동일 Frame** 배포(원칙 5.1). `run_once()`가 결정적 단위(스레드 없이 테스트), `start()/stop()`은 배경 스레드 wrapper.
- **transient miss vs end-of-stream 구분(리뷰 후)**: `read()→None`은 "일시적 미스(이번 tick에 프레임 없음, 재시도)"만 의미하고, 유한 소스의 스트림 종료는 `EndOfStream` 예외로 신호한다. `_loop`은 None이면 계속, `EndOfStream`이면 종료. 웹캠 프레임 하나 놓쳐도 파이프라인이 죽지 않는다. 실제 카메라는 자연 종료가 없으므로 `EndOfStream`을 던지지 않고 `close()`로만 멈춘다.
- **소스 lifecycle(리뷰 후)**: `CapturePipeline`이 컨텍스트 매니저(`with`) + `close()`로 소스 디바이스를 결정적으로 해제한다. `stop()`은 루프만 멈추고 디바이스는 유지(재시작 가능). `close()` 없이는 `cv2.VideoCapture` 핸들이 프로세스 종료까지 열린 채 누수됐던 문제 해소.
- 테스트 21개(리뷰 후 +4): 클럭 monotonic/gapless, queue drop-oldest·카운트, 파이프라인 fan-out 동일 stamp·소비자별 독립 backpressure, transient miss 내성, close/컨텍스트 매니저 해제.

## 진행 상황

- [x] 청크 1: capture/ 구현 + 단위 테스트 17개 (pytest/mypy/ruff 통과)
- [x] 청크 2: protocol/ 구현 + 단위 테스트 35개 (누적 52개, pytest/mypy/ruff 통과)
- [ ] 청크 3: adapters/
- [ ] 청크 4: telemetry/

## 이슈 / 의사결정 필요 사항

- **(계약 공백) Intent에 생성 timestamp가 없다.** `expires_in_ms`는 상대값인데 기준 시점이 없어, protocol이 "수신 시점 + expires_in_ms"로 해석한다. Fusion→Protocol 전송 지연이 TTL에 반영되지 않는다. 정밀 TTL이 필요하면 Intent에 `created_at_ms`(공통 clock) 추가를 계약 변경으로 논의. 2인(Fusion)과 협의 대상.
- **(계약 공백) enum capability를 Intent가 표현할 수 없다.** `Intent.value`가 `int|float|bool`이라 enum 멤버 문자열(예: aircon `mode`)을 담지 못한다. MVP 기기(전구·노트북)엔 enum이 없어 boolean/number만 구현. 확장 기기 추가 시 `Intent.value`에 `str` 허용을 계약 변경으로 논의.
- **(결정됨)** 상대 연산의 결과 절대값 clamp를 adapter가 담당 — 아래 decisions.md 참고.
- (미정) bounded queue 기본 capacity 값 — 실제 카메라 fps·소비자 처리속도 측정 후 configs로 확정 예정.

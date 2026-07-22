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
- WiZ 전구 adapter (로컬 UDP, 2026-07-22 추가 — 클라우드/토큰 없이 LAN 직접 제어)
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

### telemetry/ (청크 4)

관찰 가능성 프리미티브. 상태 전이·오류·latency를 공통 correlation id와 shared-clock timestamp로 추적하고, 영상·비밀값은 남기지 않는다(원칙 5.5).

- `events.py`: `TraceEvent`(timestamp_ms·kind·correlation_id·detail) + `EventKind`(TRACKING_LOST/QUEUE_DROP/LOCK_TRANSITION/INTENT_COMMIT/INTENT_REJECT/COMMAND_STATE, 원칙 5.5 목록) + `TraceSink`(Protocol) + `InMemoryTraceSink`(테스트·replay·monitoring용, correlation별 조회) + `Tracer`(shared clock으로 stamp 후 sink에 기록). detail은 호출자가 넘긴 짧은 문자열만 저장 — 토큰·프레임은 담지 않는다(adapter가 이미 비밀 마스킹).
- `latency.py`: `LatencyStage`(capture→inference / gesture_end→commit / commit→dispatch / dispatch→ACK / end_to_end, 원칙 5.4·README 13) + `span_ms`(음수 span은 out-of-order 버그라 raise) + `percentile`(nearest-rank) + `LatencyAggregator`(stage별 샘플 수집, p50/p95/p99/max/mean). p95 목표(노트북 ≤150ms, 전구 ≤1000ms)를 재현 가능한 실제 샘플로 평가(원칙 1.4, 손으로 만든 숫자 금지).
- **통합(emit 지점 배선)은 별도 후속 작업**: capture/protocol/adapter가 실제로 이 tracer/aggregator를 호출하도록 composition root에서 배선하는 것은 이번 범위 밖. telemetry는 monitoring(공동 소유)이 소비할 안정적 프리미티브만 제공.
- 테스트 14개: tracer stamp·순서·correlation 조회·snapshot 격리, span 음수 거부, percentile nearest-rank·빈 입력·비정렬, aggregator 음수 거부·stage 독립·집계.

### adapters/ (청크 3b — SmartThings)

전구 실제 실행 경계. 실제 상태를 되읽어 확인하고 성공을 위조하지 않는다(원칙 1.1).

- `http.py`: `HttpTransport`(Protocol) + `HttpRequest`/`HttpResponse` + 타입 오류(`TransportTimeout`/`TransportNetworkError`) + `UrllibTransport`(stdlib, 추가 의존성 없음). adapter는 구체 HTTP 라이브러리가 아니라 이 경계에 의존 → fake transport로 네트워크 없이 테스트.
- `config.py`: `read_env_file()` — `.env`를 dict로 파싱(주석·빈 줄 무시, 없으면 빈 dict → "미설정"으로 degrade). 비밀·환경 종속 값은 env/`.env`로만 주입(원칙 6.1/6.2).
- `smartthings.py`: `SmartThingsConfig.from_env()`(토큰 없으면 `None`) + `SmartThingsAdapter`.
  - 토큰 없음 → `UNCONFIGURED`(네트워크 미접촉, Windows 전용 실행 차단 안 함, 원칙 6.3). device_id 미매핑 → `FAILED`.
  - power: set(on/off), toggle(현재 상태 읽어 반대로). brightness/color_temperature: set은 절대값, increment/decrement는 **현재 상태 GET → delta 적용 → [min,max] clamp → 절대 setLevel**(clamp가 adapter 몫이라는 decisions.md 결정 구현). README 데모의 "Swipe Down 밝기 감소"가 이 경로.
  - 명령 후 상태 GET으로 verify: 일치 `VERIFIED`, 되읽기 실패·불일치 `UNVERIFIED`(보냈으나 미확인). 위조된 성공 없음.
  - 오류 분류(원칙 6.4): timeout / network / auth(401·403) / rate limit(429) / 기타 HTTP를 구분해 `FAILED` detail에 기록. **토큰은 detail·로그에 절대 노출 안 함**(원칙 6.5, 테스트로 보장).
- **비밀 관리**: 실제 토큰은 `.env`(gitignore)에만. `.env.example`은 키 이름·안전한 설명·플레이스홀더만(원칙 6.2). 전구 UUID는 미확보 → `SMARTTHINGS_DEVICE_TARGETS={}` 상태. 실물 연결하려면 `GET /devices`로 UUID 얻어 채워야 함(아래 이슈).
- 테스트 18개: unconfigured·미매핑·power set/toggle·brightness set·increment/decrement clamp(하한 0·상한 100)·오류 4종 분류·토큰 비노출·verify 불가→UNVERIFIED·미지원 capability·from_env 파싱.

### adapters/ (청크 3c — WiZ 전구, 2026-07-22 데모용)

`wiz.py`: Philips WiZ 전구를 **클라우드 없이 로컬 UDP**(포트 38899, JSON-RPC `setPilot`/`getPilot`)로 직접 제어한다. SmartThings와 같은 `DeviceAdapter` 계약이지만 경로가 다르다 — 토큰·계정 불필요, 왕복이 LAN 한 홉(실측 15~47ms)이라 제스처 제어에 유리하다. `room.bulb` 기기의 실제 실행 백엔드(`jarvis.runtime.devices.build_default_registry`가 배선).

- **정직한 상태**(원칙 1.1): `setPilot` 후 `getPilot`으로 되읽어 일치할 때만 `VERIFIED`. 보냈지만 확인 못 하면 `UNVERIFIED`(성공 위조 없음). 설정(`WIZ_DEVICE_TARGETS`) 없으면 `UNCONFIGURED`로 네트워크 미접촉.
- **capability**: `power`(set/toggle), `brightness`·`color_temperature`(set 절대값, increment/decrement는 현재 상태 GET→delta→`[min,max]` clamp→절대 setPilot, clamp는 adapter 몫이라는 decisions.md 결정 구현), **`color`(색상각 hue, 0~360°)**. color는 **순환량**이라 유일하게 클램프하지 않고 360°에서 0°로 감아 돈다(`_apply_color`) — 클램프하면 회전 제스처가 양 끝에서 죽는다. 기기 프로필 범위는 실측 WiZ 모델(ESP25_SHRGB_01): brightness 10~100(하한 10=`minDimLevel`), color_temperature 2700~6500K.
- **주소 지정**: `IP`/`MAC`/`MAC@IP` 세 형식. `MAC@IP` 권장 — 평소 IP로 바로 쏘고(탐색 비용 0), 무응답 시에만 MAC 재탐색으로 DHCP 주소 변경 흡수(AP 클라이언트 격리로 브로드캐스트 탐색이 막히는 네트워크 대비).
- 테스트(`test_adapter_wiz.py`): unconfigured·power·brightness/color clamp·color 순환·verify 불일치→UNVERIFIED·주소 형식 파싱 등.

### adapters/ (청크 3a — Windows + dispatch 코디네이터)

실제 실행 경계. adapter는 실제로 일어난 일을 정직하게 보고하고, 성공을 위조하지 않는다(원칙 1.1).

- **계약 보강**: `Command`에 `device_id` 추가(계약 변경, decisions.md 기록). command만으로 어느 adapter로 라우팅할지 결정하기 위함. dev-3 경계 안에서만 쓰여 Gaze/Gesture/Fusion 영향 없음.
- `base.py`: `AdapterStatus`(ACKNOWLEDGED/VERIFIED/UNVERIFIED/FAILED/UNCONFIGURED) + `AdapterResult` + `DeviceAdapter`(Protocol). `DispatchCoordinator`가 `device_id`로 profile·adapter를 **먼저** 라우팅 → TTL 재검증(원칙 4) → 그 직후에만 `DISPATCHED` 전이 → adapter 결과를 lifecycle 전이로 매핑. 실패·미설정·만료는 모두 미실행이 안전 기본(원칙 2.7).
- **dispatch 순서·상태 정직성 수정(리뷰 후)**: `DISPATCHED`("adapter로 보냄")를 라우팅·TTL 확인 전에 올리던 것을 바로잡음. (1) 대상 기기가 registry에 없으면 `REJECTED`(never dispatched, lifecycle 새 edge `VALIDATED→REJECTED`)로 정직한 터미널 처리 — 예전엔 `DISPATCHED`를 거쳐 `FAILED`. (2) profile이 없는 adapter를 가리키면(배선 오류) `UnknownAdapterError`를 raise하되 command은 `VALIDATED`로 남아(보낸 적 없음이 정직) 예전처럼 `DISPATCHED`에 정체되지 않음. (3) `dispatch()`를 idempotent하게: `VALIDATED`가 아니면 adapter를 건드리지 않고 현재 상태를 리포트 → 재호출로 중복 실행·`IllegalTransitionError` 없음.
- `windows.py`: `WindowsAdapter`가 discrete command(scroll/volume/media/**desktop_switch**)를 `InputSink`로 매핑. 로컬 합성 입력은 OS가 받아들이지만 효과를 되읽지 않으므로 성공은 `ACKNOWLEDGED`가 정직한 상한(VERIFIED 위조 안 함). 처리 못 하는 capability/operation은 추측 없이 `FAILED`. `Win32InputSink`는 user32(keybd_event/mouse_event) 하드웨어 경계로 `ctypes` lazy import — 실물 검증 필요(자동 테스트는 fake sink 사용).
  - **desktop_switch(2026-07-22)**: `_desktop_switch`가 `switch_desktop(forward, count)`로 가상 데스크톱을 전환. Windows는 **Ctrl+Win+→/←**(hold), macOS `MacOSInputSink`는 **Ctrl+→/←**(Space 이동, "Mission Control > 이전/다음 Space" 단축키에 의존, 기본 켜짐). README 15장 "노트북 Swipe Left" 데모 시나리오용. 처음엔 창 전환(`window_switch`, Alt+Tab/Cmd+Tab)으로 구현했으나 좌우 슬라이드의 공간적 은유("옆으로 넘긴다")에 맞춰 데스크톱 전환으로 바꾸고 capability 이름도 함께 개명했다(사용자 지시).
- **macOS 확장(2026-07-18, 이번 세션)**: `WindowsAdapter`는 `InputSink`만 호출해 원래 OS 무관이었다 — `macos.py`에 `MacOSInputSink`(Quartz/AppKit CGEvent, `macos` extra: `pyobjc-framework-Quartz`+`pyobjc-framework-Cocoa`, `sys_platform=='darwin'` 마커)를 추가하고 `windows.py`에 `default_input_sink()`(플랫폼별 sink 선택, 미지원 OS는 `RuntimeError`)를 더했다. `Win32InputSink`·기존 Windows 테스트는 전혀 건드리지 않음(8개 그대로 통과). macOS 미디어 키(볼륨·재생/일시정지)는 표준 keycode가 아니라 `NSEvent`의 system-defined 이벤트(`NX_KEYTYPE_*`)로 보낸다 — Win32InputSink와 마찬가지로 실물 검증 필요, 자동 테스트는 import·매핑 테이블·플랫폼 분기만 확인(실제 CGEvent 발사는 개발자 화면에 부작용을 남겨 자동화하지 않음).
- 커서 연속 경로(Cursor Control Mapper, README 6장)는 `InputSink.move_cursor`를 재사용할 예정이나 이번 범위 밖(pointer/ 모듈, 공동 소유). macOS 쪽도 `MacOSInputSink.move_cursor`로 이미 준비됨.
- 테스트 20개(리뷰 후 +4) + macOS 확장 테스트 5개: windows 매핑(scroll/volume/media·미지원·sink 오류 내성) 8개 + coordinator(5종 status 매핑·만료 미전달·미등록 adapter 정체 없음·미등록 기기 REJECTED·재호출 idempotency) 10개 + lifecycle `VALIDATED→REJECTED` 1개 + Command 계약 테스트 2개(`tests/contract/`) + macOS sink import·매핑·플랫폼 분기 5개.

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
- `source.py` `FrameSource`(Protocol) + `OpenCVCameraSource`: 카메라는 하드웨어 IO 경계로 분리, `cv2` lazy import(vision extra 없이도 코어·테스트 동작). 성공 위조 없음(원칙 1.1). Windows에서는 `CAP_DSHOW` 백엔드 + `CAP_PROP_BUFFERSIZE=1`로 연다 — 기본 MSMF 백엔드의 내부 버퍼링이 지연을 주기적으로 쌓았다 푸는 끊김을 유발하기 때문(2026-07-20 결정).
- `pipeline.py` `CapturePipeline`: 1회 캡처 → 1회 stamp → 모든 소비자에 **동일 Frame** 배포(원칙 5.1). `run_once()`가 결정적 단위(스레드 없이 테스트), `start()/stop()`은 배경 스레드 wrapper.
- **transient miss vs end-of-stream 구분(리뷰 후)**: `read()→None`은 "일시적 미스(이번 tick에 프레임 없음, 재시도)"만 의미하고, 유한 소스의 스트림 종료는 `EndOfStream` 예외로 신호한다. `_loop`은 None이면 계속, `EndOfStream`이면 종료. 웹캠 프레임 하나 놓쳐도 파이프라인이 죽지 않는다. 실제 카메라는 자연 종료가 없으므로 `EndOfStream`을 던지지 않고 `close()`로만 멈춘다.
- **소스 lifecycle(리뷰 후)**: `CapturePipeline`이 컨텍스트 매니저(`with`) + `close()`로 소스 디바이스를 결정적으로 해제한다. `stop()`은 루프만 멈추고 디바이스는 유지(재시작 가능). `close()` 없이는 `cv2.VideoCapture` 핸들이 프로세스 종료까지 열린 채 누수됐던 문제 해소.
- 테스트 21개(리뷰 후 +4): 클럭 monotonic/gapless, queue drop-oldest·카운트, 파이프라인 fan-out 동일 stamp·소비자별 독립 backpressure, transient miss 내성, close/컨텍스트 매니저 해제.

## 진행 상황

- [x] 청크 1: capture/ 구현 + 단위 테스트 (리뷰 수정 후 21개)
- [x] 청크 2: protocol/ 구현 + 단위 테스트 35개
- [x] 청크 3a: adapters/base + DispatchCoordinator + Windows adapter (리뷰 수정 후 76개). Command.device_id 계약 보강 + dispatch 순서·idempotency 수정 + Command contract test 포함
- [x] 청크 3b: adapters/ SmartThings + http transport + config/secrets + `.env.example` + 테스트 18개 (누적 94개, pytest/mypy/ruff 통과)
- [x] 청크 4: telemetry/ events + latency 프리미티브 + 테스트 14개 (누적 108개, pytest/mypy/ruff 통과)
- [x] 청크 3c: WiZ 전구 adapter(로컬 UDP) + `room.bulb` 실행 백엔드 + color(hue) capability (2026-07-22, 데모용)
- [x] 청크 5: composition root 일부 — `jarvis.runtime.devices`가 laptop(로컬 입력)·room.bulb(WiZ) 레지스트리·executor 배선, `IntentExecutor`가 CommitDecision→기기 명령까지 묶음. 모니터 `시연` 탭이 그 앞단(실시간 TargetEstimate·GestureEstimate → FusionEngine)을 `demo_bridge`로 연결 — 전체 시선→제스처→Fusion→실기기 경로 완성
- [ ] 후속: telemetry emit 지점 통합(capture/protocol/adapter가 tracer/aggregator 실제 호출) + 실물 검증(WiZ 전구 IP/MAC·SmartThings UUID·Win32)

## 이슈 / 의사결정 필요 사항

- **(계약 공백) Intent에 생성 timestamp가 없다.** `expires_in_ms`는 상대값인데 기준 시점이 없어, protocol이 "수신 시점 + expires_in_ms"로 해석한다. Fusion→Protocol 전송 지연이 TTL에 반영되지 않는다. 정밀 TTL이 필요하면 Intent에 `created_at_ms`(공통 clock) 추가를 계약 변경으로 논의. 2인(Fusion)과 협의 대상.
- **(계약 공백) enum capability를 Intent가 표현할 수 없다.** `Intent.value`가 `int|float|bool`이라 enum 멤버 문자열(예: aircon `mode`)을 담지 못한다. MVP 기기(전구·노트북)엔 enum이 없어 boolean/number만 구현. 확장 기기 추가 시 `Intent.value`에 `str` 허용을 계약 변경으로 논의.
- **(결정됨)** 상대 연산의 결과 절대값 clamp를 adapter가 담당 — 아래 decisions.md 참고.
- (미정) bounded queue 기본 capacity 값 — 실제 카메라 fps·소비자 처리속도 측정 후 configs로 확정 예정.
- **(실물 검증 필요) SmartThings 전구 UUID 미확보.** 토큰은 `.env`에 저장했으나 `SMARTTHINGS_DEVICE_TARGETS={}` 상태라 실제 라우팅 불가. `GET https://api.smartthings.com/v1/devices`(토큰 필요)로 전구 UUID를 얻어 `.env`에 채워야 실물 동작. HTTP 계층·매핑·verify 로직은 fake transport로만 검증됨 — 실제 API 응답 shape(status JSON 경로)과 전구 실물 동작은 미확인.
- **(실물 검증 필요) Windows Win32 입력 경로**(청크 3a) — 실제 커서·스크롤·볼륨은 fake sink로만 검증. 실물 노트북 확인 필요.

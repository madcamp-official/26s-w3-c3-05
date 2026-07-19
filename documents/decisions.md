# Decisions Log

스프린트 중 threshold·포맷·범위를 바꿀 때마다 한 줄씩 기록한다. 회의 없이도 서로 왜 바뀌었는지 알 수 있게 하는 것이 목적.

| 날짜 | 결정 내용 | 이유 | 결정자 |
| --- | --- | --- | --- |
| 2026-07-18 | `timestamp_ms`·`frame_id`를 모든 프레임 기반 메시지의 정식 계약 필드로 확정 | 시선·제스처 temporal alignment(README 9장 Commit 조건 6)에 필수. messages.py 구현과 문서 상태 불일치 해소 | suh1088 |
| 2026-07-18 | Command에 `capability`/`operation`/`value` 포함 (stateless adapter) | adapter가 intent를 재조회하지 않아 구조 단순, dispatch 전 검증도 payload로 수행 가능 | suh1088 |
| 2026-07-18 | 만료 시각 필드명 `expires_at_ms`로 통일 (단위 접미사 규칙: 절대=`_at_ms`, 상대=`_in_ms`) | 문서(`expires_at`)와 코드(`expires_at_ms`) 불일치 해소, 단위 명시 | suh1088 |
| 2026-07-18 | `gesture`·`capability`·`operation`·`target`은 열린 문자열 키(snake_case), 제스처→capability 매핑은 `configs/` 데이터로 관리 | 커스텀 제스처·신규 기기를 코드 수정 없이 추가하기 위함. 닫힌 enum 가정으로 코드 짜면 확장 시 전면 수정 필요 | suh1088 |
| 2026-07-18 | 노트북 Lock 중 커서/제스처 분기: 기본은 커서 모드, Gesture Spotter `ONSET` 감지 시 커서 일시정지 후 제스처 판정 우선. 판정 실패(`IDLE` 복귀) 시 커서 모드 복귀 | 별도 모드 전환 동작 없이 자연스러움. 제스처 시작 순간의 커서 미세 끌림은 감수 | suh1088 |
| 2026-07-18 | 제스처 모델 추론은 MVP에서 로컬 실행. 단, 추론 부분을 교체 가능한 경계로 분리해 나중에 GPU 서버 스트리밍으로 옮길 수 있게 설계 | 데모 당일 네트워크 리스크 제거 + 향후 무거운 모델(ST-GCN 등) 서버 실행 여지 확보 | suh1088 |
| 2026-07-18 | Gaze 방식: 머리+눈 오프셋을 합성한 시선 방향 단위 벡터 + 코사인 유사도 비교 (README 7장) | 등록 시(고개 돌림)와 실사용 시(눈짓만)의 행동 불일치에도 같은 방향이면 같은 벡터가 나오도록 | 팀 합의 |
| 2026-07-18 | command_id를 `cmd-{intent_id}`로 결정적 생성 | 같은 intent 재시도가 같은 command_id로 collapse되어 dedup만으로 idempotency 보장(원칙 3) | suh1088(3인) |
| 2026-07-18 | 상대 연산(increment/decrement)의 결과 절대값 [min,max] clamp는 protocol이 아니라 adapter가 수행 | clamp에 기기 현재 상태가 필요한데 protocol은 상태를 모름. protocol은 delta의 부호·step 배수만 정적 검증 | suh1088(3인) |
| 2026-07-18 | capability 연산 집합(number: set/increment/decrement, boolean: set/toggle)을 capability spec에 선언(config 주입) | 연산을 코드에 하드코딩하지 않고 기기별로 다르게 선언 가능하게. 열린 문자열 키 원칙과 일관 | suh1088(3인) |
| 2026-07-18 | (미해결·논의필요) Intent에 생성 timestamp 없음 → TTL을 "수신 시점 기준"으로 해석. enum capability는 Intent.value(int/float/bool)로 표현 불가라 보류 | 정밀 TTL·enum 기기는 계약 변경 필요. MVP 범위(전구·노트북)에선 문제 없어 후순위 | 3인 제기, Fusion과 협의 대상 |
| 2026-07-18 | capture 소스의 `read()→None`은 "일시적 미스"로만 정의하고, 유한 소스의 스트림 종료는 `EndOfStream` 예외로 분리 | 리뷰에서 발견: 둘을 혼동해 웹캠 프레임 하나 놓치면 파이프라인이 죽던 버그 수정 | suh1088(3인) |
| 2026-07-18 | `CapturePipeline`을 컨텍스트 매니저 + `close()`로 소스 디바이스를 명시 해제. `stop()`은 루프만 중단 | 리뷰에서 발견: `close` 미호출로 `cv2.VideoCapture` 핸들이 프로세스 종료까지 누수 | suh1088(3인) |
| 2026-07-18 | protocol dedup을 `register()` 원자 연산 하나에만 의존하도록 변경(선(先) `seen()` 검사 제거, `DuplicateCommandError`→`Rejected(DUPLICATE)`) | 리뷰에서 발견: 동시 submit 시 검사-등록 사이 경쟁으로 예외가 새어나가던 TOCTOU 수정 | suh1088(3인) |
| 2026-07-18 | `Command`에 `device_id` 필드 추가 (계약 변경) | dispatch 시 어느 adapter(Windows/SmartThings)로 라우팅할지 결정하려면 command이 대상 기기를 알아야 함. Command은 dev-3 경계 안에서만(protocol 생성·adapter 소비) 쓰여 Gaze/Gesture/Fusion 영향 없음 | suh1088(3인) |
| 2026-07-18 | lifecycle에 `VALIDATED→REJECTED` edge 추가. dispatch 시 대상 기기가 registry에 없어 라우팅 불가하면 `REJECTED`(never dispatched)로 처리 | 리뷰에서 발견: 라우팅을 `DISPATCHED` 전이 후에 하던 순서를 바로잡음. 보낸 적 없는 command에 "보냄" 상태를 붙이지 않도록(정직성·telemetry 정확도) | suh1088(3인) |
| 2026-07-18 | `DispatchCoordinator.dispatch()`를 idempotent하게: command이 `VALIDATED`가 아니면 adapter 미접촉·현재 상태 리포트 | 리뷰에서 발견: 재호출 시 `IllegalTransitionError`가 raw로 튀던 것 해소. 중복 실행은 이전에도 방지됐으나 coordinator 레벨에서 "정확히 한 번" 보장을 명시 | suh1088(3인) |
| 2026-07-18 | Python 최소 버전을 3.12로 통일(`requires-python>=3.12`, mypy `python_version=3.12`) | Gaze 병합으로 추가된 numpy 2.5.1 스텁이 PEP 695 `type` 구문(3.12+)을 써서 mypy 3.11 타깃이 numpy 파싱에 실패→타입체크 전체 중단. 팀 전원이 실제로 3.12 venv에서 개발 중이라 3.11 지원 선언만 현실과 어긋나 있었음. 3.11 지원 포기 | suh1088(3인) |
| 2026-07-18 | Gaze UNKNOWN 거부에 최근접 등록 방향과의 최대 각도 25도를 추가 | 등록 기기가 하나면 기기 간 정규화 확률이 방향과 무관하게 1.0이 되어 먼 시선을 선택하는 결함을 방지 | Gaze Targeting 담당 |
| 2026-07-18 | Fusion(`gesture_fusion.alignment.TargetLockTracker`)은 Gaze의 자체 Gaze Lock(README 7장, 커서 게이팅용)과 별개로 자기만의 Target Lock을 `TargetEstimate`(§1) 스트림에서 독립 계산한다 | interface-contract.md §1에는 lock 여부 필드가 없고, 모듈 경계 규칙상 Gaze 내부 구현(`jarvis.gaze.lock`)을 import할 수 없음. 같은 원시 확률·margin 신호를 각자 목적(커서 통과 vs intent commit)으로 독립 판단 — 한쪽 임계값 변경이 다른 쪽에 영향 없음 | 2인(Fusion) |
| 2026-07-18 | Fusion Target Lock의 `target_lock_ttl_ms`는 고정 만료가 아니라 유효한 프레임마다 `timestamp_ms + ttl`로 갱신되는 슬라이딩 윈도우 | 한 대상을 계속 응시하는데 1.5초마다 재-dwell해야 하는 것은 비현실적. 시선이 짧게 흔들려도(유예 기간 안) lock을 유지해 제스처 판정 도중 끊기지 않게 함 | 2인(Fusion) |
| 2026-07-18 | 제스처→capability 매핑을 `configs/gesture_capability_map.json`(신규 stdlib JSON, 새 의존성 추가 없음)으로 관리하고, 매핑 키를 `(target device_id, gesture)` 쌍으로 둔다 | 같은 제스처도 기기에 따라 다른 동작을 뜻함(README 9·15장: 노트북 swipe_down=스크롤, 전구 swipe_down=밝기 감소). gesture만 키로 쓰면 이 구분을 표현할 수 없음. YAML은 새 의존성이라 JSON(표준 라이브러리)을 택함 | 2인(Fusion) |
| 2026-07-18 | (미해결·논의필요) `gesture_capability_map.json`에 `laptop.swipe_left/right → window_switch` capability를 매핑해 뒀으나, 3인의 `WindowsAdapter`는 아직 `scroll`/`volume`/`media`만 처리하고 `window_switch`는 미구현 | README 15장 필수 데모 시나리오("노트북 Swipe Left 창 전환")를 완성하려면 dev-3이 핸들러를 추가해야 함. 지금은 Protocol이 `FAILED`로 정직하게 거부(성공 위조 없음) | 2인 제기, dev-3과 협의 대상 |
| 2026-07-18 | macOS 입력은 ctypes로 CoreGraphics를 직접 바인딩하지 않고 `pyobjc-framework-Quartz`+`pyobjc-framework-Cocoa`(신규 `macos` extra, `sys_platform=='darwin'` 마커)를 채택 | CGEvent의 CFType·구조체 바인딩을 ctypes로 직접 하면 실패 위험이 크고, 미디어 키(NX_KEYTYPE 기반 system-defined 이벤트)는 AppKit `NSEvent`가 필요해 순수 Quartz만으로 불가능. `Win32InputSink`(ctypes)는 전혀 건드리지 않음 — Windows 동작·의존성 불변 | 팀 합의(사용자 지시) |

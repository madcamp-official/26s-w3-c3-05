# JARVIS 개발 원칙

이 문서는 [README.md](../README.md)의 MVP 범위와 아키텍처, 그리고 `documents/` 아래의 모듈별 계획을 구현할 때 모든 팀원이 따르는 공통 원칙이다. JARVIS의 우선순위는 기능 수가 아니라 **사용자가 의도한 대상에만 명령을 정확히 한 번 실행하는 것**이다.

## 1. 실제 동작과 재현 가능한 검증

1. 프로덕션 경로에서 성공을 가장하는 구현을 금지한다. Windows 또는 SmartThings 연동이 준비되지 않았거나 실패한 경우 `ACKNOWLEDGED`, `VERIFIED`, 성공 응답을 임의로 만들지 않고 `UNCONFIGURED`, `FAILED`, `UNVERIFIED` 등 실제 상태를 반환한다.
2. 런타임 코드에 특정 기기, 특정 사용자, 특정 카메라 위치 또는 데모 시나리오만 통과시키는 값을 하드코딩하지 않는다. 임계값은 이름 있는 설정으로 관리하고, 변경 이유와 평가 결과를 남긴다.
3. 테스트용 synthetic trace, replay fixture, adapter stub과 장애 주입은 허용한다. 단, 테스트 전용 경로에 두고 이름과 메타데이터로 합성 데이터임을 명시하며 실제 성능 지표와 섞지 않는다.
4. Target Selection Accuracy, Gesture Event Recall, Wrong Actuation Rate, latency를 계산할 때 데이터셋, 실행 조건, 분모와 측정 구간을 함께 기록한다. 수동으로 만든 숫자나 재현할 수 없는 결과를 사용하지 않는다.
5. 얼굴·홍채·손의 raw 프레임은 필요한 계산이 끝나면 기본적으로 폐기한다. 저장이 필요한 평가·디버깅 세션은 사용 목적과 보존 범위를 명시하고 별도로 동의를 받는다.

## 2. 안전 실행 원칙

1. 시선만으로 명령을 실행하지 않는다. 이산 명령은 Target Lock, 완결된 gesture event, temporal alignment, confidence 기준, capability 검증을 모두 통과한 경우에만 commit한다.
2. 판단이 불확실하면 실행하지 않는다. 낮은 confidence, 작은 1·2순위 margin, 불안정한 시선, `UNKNOWN` 대상, 추적 손실, 만료된 lock 또는 intent는 모두 거부한다.
3. 한 gesture event는 최대 하나의 intent를 만들고, 한 `command_id`는 최대 한 번만 실행한다. 재시도하더라도 idempotency와 만료 시각을 유지한다.
4. Intent와 Command의 TTL을 실행 직전 다시 검증한다. 늦게 도착했거나 순서가 뒤바뀐 메시지는 폐기하고 그 이유를 trace에 남긴다.
5. 기기의 capability, 타입, 범위, step을 dispatch 전에 검증한다. 지원하지 않는 gesture-capability 조합이나 범위를 벗어난 값은 adapter로 보내지 않는다.
6. Cursor Control Mapper는 이산 명령 경로의 예외다. 단, `Gaze Lock == laptop`인 동안만 좌표 스트림을 전달하고 lock 또는 추적이 풀리면 즉시 중단한다.
7. 장애 시 안전한 기본 동작은 항상 **미실행**이다. 자동 fallback이 다른 기기 선택, 임계값 완화 또는 중복 명령으로 이어져서는 안 된다.

## 3. 모듈 소유권과 협업 경계

JARVIS는 다음 세 영역을 독립적으로 개발한다.

- Gaze Targeting: [gaze.md](gaze.md)의 얼굴·홍채 추적, calibration, target classifier, `UNKNOWN` rejection, smoothing, Gaze Lock과 정확도 평가
- Gesture & Intent Fusion: [gesture-fusion.md](gesture-fusion.md)의 hand landmark, dynamic gesture spotting, phase, temporal alignment, safe commit과 hard-negative 평가
- Runtime & Device Protocol: [runtime-protocol.md](runtime-protocol.md)의 카메라 파이프라인, 공통 timestamp, bounded queue, capability model, Windows·SmartThings adapter, ACK·timeout·dedup과 latency 측정

각 담당자는 자기 모듈 내부 구현을 독립적으로 바꿀 수 있다. 다른 모듈의 내부 코드를 편의상 직접 참조하거나 우회하지 않는다. 모듈 경계를 넘는 데이터와 동작 변경은 아래 계약 변경 절차를 따른다.

공통 UI는 핵심 런타임을 관찰하고 calibration·상태·오류를 보여 주는 최소 도구다. UI 편의를 위해 인식 결과나 명령 상태를 임의로 성공 처리하지 않는다.

## 4. 인터페이스 계약과 시간 기준

1. Gaze → Fusion, Gesture → Fusion, Fusion → Protocol, Protocol → Device 포맷의 단일 기준은 [interface-contract.md](interface-contract.md)다.
2. 계약을 바꿀 때는 다음 순서를 지킨다.
   1. `interface-contract.md`에 필드, 타입, 단위, 필수 여부, 오류 또는 unknown 표현을 먼저 제안한다.
   2. 영향을 받는 송신·수신 모듈 담당자가 함께 검토한다.
   3. 합의 내용과 이유를 [decisions.md](decisions.md)에 기록한다.
   4. 양쪽 구현과 계약 테스트를 같은 변경 단위에서 반영한다.
3. 모든 프레임 기반 메시지는 캡처 시 런타임이 부여한 동일한 monotonic clock의 `timestamp_ms`와 추적 가능한 `frame_id`를 계승한다. 각 모듈에서 벽시계나 별도 기준으로 timestamp를 다시 만들지 않는다.
4. confidence 범위, 시간 단위, enum 대소문자, 좌표계와 만료 시각의 의미를 암묵적으로 가정하지 않고 계약에 명시한다.
5. 계약 필드의 즉시 삭제나 의미 변경을 금지한다. 스프린트 중 불가피한 파괴적 변경은 모든 소비자를 동시에 수정하고 replay/contract test로 검증한 뒤 결정 로그에 남긴다.

## 5. 실시간 파이프라인 원칙

1. 카메라 입력은 한 번 캡처한 프레임과 timestamp를 Gaze·Gesture 파이프라인이 공유하도록 구성한다.
2. queue는 반드시 bounded로 두고, 처리 속도가 입력을 따라가지 못할 때의 drop 또는 latest-frame 정책을 명시한다. 지연된 오래된 프레임을 무한히 처리하지 않는다.
3. 온라인 추론 경로는 미래 프레임에 의존하지 않는 causal 처리만 사용한다. 오프라인 평가와 실시간 결과를 구분한다.
4. latency는 최소한 `capture → inference`, `gesture ending → intent commit`, `commit → adapter dispatch`, `dispatch → ACK/verify`로 나누어 측정한다.
5. 추적 손실, queue drop, lock 전이, intent commit/reject, command 상태 전이는 공통 correlation id와 timestamp로 추적 가능해야 한다. 로그에 영상 원본이나 비밀값을 남기지 않는다.

## 6. 설정, 비밀정보와 외부 연동

1. SmartThings token, device identifier 등 민감하거나 환경에 종속된 값은 코드와 문서에 하드코딩하지 않고 환경 변수 또는 로컬 비밀 저장소로 주입한다. 실제 비밀값은 버전 관리하지 않는다.
2. `.env.example`에는 필요한 key 이름과 안전한 설명만 둔다. 실제 token이나 실제 기기 식별자를 넣지 않는다.
3. 실행 모드에 필수인 설정이 없으면 해당 adapter는 명확한 `UNCONFIGURED` 상태로 실패한다. 단, SmartThings 설정이 없다는 이유로 Gaze 평가나 Windows-only 실행까지 불필요하게 막지 않도록 모드별 필수 설정을 구분한다.
4. 외부 API의 timeout, rate limit, 인증 실패와 네트워크 오류를 구분한다. 재시도는 bounded backoff를 사용하며 만료된 command는 재시도하지 않는다.
5. 로그와 오류 메시지에서 token, 인증 header, 개인정보를 마스킹한다.

## 7. AI 및 학습 모델 사용 경계

1. JARVIS의 실시간 명령 결정은 README에 정의된 Gaze, Gesture, Fusion과 명시적 state machine·검증 규칙으로 수행한다. 외부 생성형 AI의 비결정적 응답을 안전 실행 경로에 추가하지 않는다.
2. 모델 출력은 신뢰하지 않고 범위, shape, finite 여부, confidence와 enum을 검증한 뒤 사용한다. NaN, 누락, 알 수 없는 label은 실행이 아니라 reject로 이어져야 한다.
3. 학습 모델 파일에는 버전, 입력 feature, 전처리, label 집합, 학습 데이터 출처와 평가 결과를 연결할 수 있는 메타데이터를 둔다.
4. AI 개발 도구는 코드·문서 초안과 분석에 사용할 수 있지만 비밀정보나 동의받지 않은 영상·생체정보를 외부 모델에 입력하지 않는다. 생성된 변경은 담당자가 테스트하고 검토한 뒤 반영한다.

## 8. 브랜치, 커밋과 문서화

- 권장 브랜치: `gaze/*`, `gesture-fusion/*`, `runtime-protocol/*`, `contract/*`, `docs/*`
- 권장 커밋 접두사: `feat(gaze):`, `feat(gesture):`, `feat(fusion):`, `feat(runtime):`, `feat(protocol):`, `fix(...)`, `test(...)`, `docs:`, `chore(config):`
- threshold, 포맷, 좌표계, 상태 전이, MVP 범위를 바꾸면 코드뿐 아니라 해당 모듈 문서와 [decisions.md](decisions.md)를 함께 갱신한다.
- 일별 실제 진행과 지연 사항은 [plan.md](plan.md)에 짧게 기록한다.

## 9. Definition of Done

변경은 다음 조건을 충족해야 완료로 본다.

- 변경한 모듈의 단위 테스트와 lint/type check가 통과한다.
- 모듈 경계 변경 시 producer-consumer contract test와 trace replay가 통과한다.
- 정상 경로뿐 아니라 `UNKNOWN`, 낮은 confidence, 추적 손실, TTL 만료, 중복 command, adapter timeout·실패를 검증한다.
- 실제 외부 기기 없이 검증한 경우 그 범위를 명시하며, adapter stub 결과를 실제 기기 검증으로 표현하지 않는다.
- 성능 관련 변경은 동일 조건의 전후 Target Selection Accuracy, Gesture Event Recall, Wrong Actuation Rate 또는 p95 latency 중 영향받는 지표를 기록한다.
- MVP 전체 완료 기준은 README의 목표인 `Wrong Actuation Rate ≤ 1%`, `Target Selection Accuracy ≥ 90%`, `Gesture Event Recall ≥ 90%`, `Duplicate Actuation = 0`, 노트북 p95 `≤ 150ms`, SmartThings 전구 p95 `≤ 1000ms`를 재현 가능한 평가로 충족하는 것이다.
- Windows 노트북과 SmartThings 전구의 실제 명령 경로, 최소 모니터링 화면, calibration, trace replay·benchmark가 README의 최종 시연 시나리오에서 함께 동작한다.

이 원칙의 해석이 충돌할 때는 **오작동 방지 → 인터페이스 계약 → 측정 가능성 → 기능 확장 속도** 순으로 우선한다.

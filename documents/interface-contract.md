# Interface Contract

세 모듈([gaze.md](gaze.md) → [gesture-fusion.md](gesture-fusion.md) → [runtime-protocol.md](runtime-protocol.md))이 서로 독립적으로 개발하면서도
마지막에 문제없이 합쳐지도록, 모듈 경계에서 주고받는 데이터 포맷을 여기에 고정한다.
포맷을 바꿀 때는 반드시 이 파일을 먼저 수정하고 [decisions.md](decisions.md)에 이유를 남긴 뒤 관련자에게 공유한다.
코드 기준 구현은 `src/jarvis/contracts/messages.py`이며, 이 문서와 코드가 어긋나면 이 문서를 먼저 고친 쪽이 맞다.

## 공통 규칙 (확정)

- **시간 기준**: 모든 프레임 기반 메시지의 `timestamp_ms`는 런타임 프로세스의 단일 monotonic clock 기준이다. 프레임 캡처 시점에 런타임이 찍은 timestamp와 `frame_id`를 각 파이프라인이 그대로 물려받으며, 각 모듈이 자체 시계로 timestamp를 다시 만들지 않는다. 나중에 제스처 추론을 원격 서버로 옮기더라도 서버는 자체 timestamp를 찍지 않고 클라이언트가 보낸 `timestamp_ms`·`frame_id`를 그대로 되돌려준다.
- **만료 시각**: 절대 시각 필드는 `_at_ms` 접미사(단위 ms, 동일 monotonic clock), 상대 시간 필드는 `_in_ms` 접미사를 쓴다.
- **confidence 계열 값**(`probability`, `stability`, `*_confidence`, `uncertainty`)은 모두 0.0~1.0 범위다.
- **`gesture`·`capability`·`operation`·`target`은 닫힌 enum이 아니라 열린 문자열 키다.** 소비자는 알 수 없는 값을 만나면 실행이 아니라 reject로 처리한다. 커스텀 제스처·신규 기기를 코드 수정 없이 추가하기 위한 규칙이다. 표기는 lowercase snake_case(`swipe_down`)로 통일한다.
- **제스처→capability 매핑은 코드가 아니라 `configs/` 데이터로 관리한다.** 매핑 추가·변경은 설정 변경이며 계약 변경이 아니다.
- `phase`만 예외적으로 닫힌 enum(`IDLE`/`ONSET`/`ACTIVE`/`ENDING`, 대문자)이다. Fusion의 commit 트리거가 이 상태 전이에 의존하므로 값 추가·삭제는 계약 변경 절차를 따른다.

## 1. Gaze → Fusion: Target 추정 결과 (확정)

```json
{
  "timestamp_ms": 1732010400123,
  "frame_id": 4821,
  "target": "room.bulb",
  "probability": 0.87,
  "second_best_probability": 0.13,
  "stability": 0.91
}
```

`timestamp_ms`·`frame_id`는 시선·제스처의 시간 관계 판단(README 9장 Commit 조건 6)에 필수라 정식 계약에 포함한다. 아무 기기도 보고 있지 않다고 판단하면 `target`은 `"UNKNOWN"`이다.

## 2. Gesture → Fusion: Gesture·Phase 출력 (확정)

```json
{
  "timestamp_ms": 1732010400156,
  "frame_id": 4822,
  "gesture": "swipe_down",
  "gesture_confidence": 0.92,
  "phase": "ENDING",
  "phase_confidence": 0.88,
  "uncertainty": 0.07
}
```

## 3. Fusion → Protocol: Intent (확정)

```json
{
  "intent_id": "intent-1042",
  "target": "room.bulb",
  "gesture": "swipe_down",
  "capability": "brightness",
  "operation": "decrement",
  "value": 1,
  "target_confidence": 0.87,
  "gesture_confidence": 0.92,
  "expires_in_ms": 1000
}
```

## 4. Protocol → Device: Command (확정)

```json
{
  "command_id": "command-3901",
  "intent_id": "intent-1042",
  "device_id": "room.bulb",
  "capability": "brightness",
  "operation": "decrement",
  "value": 1,
  "expires_at_ms": 1732010401123
}
```

Command는 대상 기기(`device_id`)와 실행 내용(`capability`·`operation`·`value`)을 직접 담는다. adapter가 intent를 재조회하지 않는 **stateless adapter** 방식이며, dispatch 시 `device_id`로 어느 adapter(Windows/SmartThings)로 라우팅할지 결정하고, dispatch 전 capability 범위·타입 검증(development-principles 2.5)도 이 payload로 수행한다.

## 변경 이력

포맷을 바꾼 경우 이 표에 한 줄씩 추가한다.

| 날짜 | 변경 내용 | 변경자 |
| --- | --- | --- |
| 2026-07-18 | `timestamp_ms`·`frame_id` 제안 초안 → 정식 계약 확정 | suh1088·Claude |
| 2026-07-18 | Command payload(`capability`/`operation`/`value`) 포함 확정, `expires_at` → `expires_at_ms` | suh1088·Claude |
| 2026-07-18 | 공통 규칙 추가: 열린 문자열 키, snake_case 표기, confidence 범위, 매핑의 config 분리 | suh1088·Claude |
| 2026-07-18 | Command에 `device_id` 추가 (adapter 라우팅에 필수) | suh1088(3인) |

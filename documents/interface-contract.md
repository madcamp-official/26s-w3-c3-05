# Interface Contract

세 모듈([gaze.md](gaze.md) → [gesture-fusion.md](gesture-fusion.md) → [runtime-protocol.md](runtime-protocol.md))이 서로 독립적으로 개발하면서도
마지막에 문제없이 합쳐지도록, 모듈 경계에서 주고받는 데이터 포맷을 여기에 고정한다.
포맷을 바꿀 때는 반드시 이 파일을 먼저 수정하고 [decisions.md](decisions.md)에 이유를 남긴 뒤 관련자에게 공유한다.

모든 메시지의 `timestamp_ms`는 같은 클럭(런타임 프로세스의 monotonic clock)을 기준으로 한다.
Gaze·Gesture 파이프라인이 각자 시계를 쓰면 temporal alignment이 불가능하므로, 프레임 캡처 시점에 런타임이 찍은 timestamp를 그대로 물려받는다.

## 1. Gaze → Fusion: Target 추정 결과

(README 7장 출력 예시 기반, 확정 시 스키마로 명세)

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

`timestamp_ms`·`frame_id`는 제안 초안 — 팀 합의 필요. 시선·제스처의 시간 관계 판단(README 9장 Commit 조건 6)에 필수라 추가했다.

## 2. Gesture → Fusion: Gesture·Phase 출력

(README 8장 모델 출력 기반)

```json
{
  "timestamp_ms": 1732010400156,
  "frame_id": 4822,
  "gesture": "SWIPE_DOWN",
  "gesture_confidence": 0.92,
  "phase": "ENDING",
  "phase_confidence": 0.88,
  "uncertainty": 0.07
}
```

`timestamp_ms`·`frame_id`는 위와 같은 이유의 제안 초안 — 팀 합의 필요.

## 3. Fusion → Protocol: Intent

(README 9장 Intent 예시 기반)

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

## 4. Protocol → Device: Command

(README 10장 Device Capability Model / 명령 상태 기반)

```json
{
  "command_id": "command-3901",
  "intent_id": "intent-1042",
  "capability": "brightness",
  "operation": "decrement",
  "value": 1,
  "expires_at": 1939401300
}
```

`capability`·`operation`·`value`는 제안 초안 — 팀 합의 필요. README 10장 예시에는 id와 만료 시각만 있는데, 그러면 adapter가 intent를 다시 조회해야 실행 내용을 알 수 있다. adapter를 stateless하게 두려면 payload를 command에 포함하는 쪽을 권장한다. intent 조회 방식으로 갈 경우 이 세 필드를 빼고 그 규칙을 여기에 명시할 것.

## 변경 이력

포맷을 바꾼 경우 이 표에 한 줄씩 추가한다.

| 날짜 | 변경 내용 | 변경자 |
| --- | --- | --- |

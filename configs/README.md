# Runtime configuration

버전 관리 가능한 비밀이 아닌 설정의 위치다. threshold는 목적과 단위를 드러내는 이름으로
관리하고 변경 이유를 `documents/decisions.md`에 기록한다. SmartThings token과 실제 device id는
이 디렉터리에 저장하지 않고 환경 변수 또는 로컬 비밀 저장소로 주입한다.

## gesture_capability_map.json

`jarvis.gesture_fusion.intent.GestureCapabilityMap`이 읽는 `(target device_id, gesture)` →
capability/operation/value 매핑이다. 새 제스처나 기기를 추가할 때는 코드가 아니라 이 파일만
수정한다. capability·operation 이름과 value의 step 제약은 dev-3의 device capability model
(`src/jarvis/runtime_protocol/protocol/capability.py`)과 맞아야 한다 — 어긋나면 Protocol이
dispatch 전에 `FAILED`로 거부한다(documents/decisions.md 참고).


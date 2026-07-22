# JARVIS Repository Structure

협업 시 파일 위치와 소유권을 빠르게 판단하기 위한 기준이다.

```text
.
├── README.md                         # 제품 목표·MVP·평가 기준
├── pyproject.toml                    # Python 패키지·검증 도구 설정
├── configs/                          # 비밀값이 아닌 런타임 설정
├── data/                             # 로컬 평가 manifest와 데이터 규칙
├── models/                           # 로컬 ML 모델과 모델 메타데이터
├── documents/                        # 설계·계약·결정·진행 기록
├── src/jarvis/
│   ├── contracts/                    # 세 모듈이 공유하는 유일한 메시지 계약
│   ├── gaze/                         # 1인: Gaze Targeting
│   ├── gesture_fusion/               # 2인: Gesture & Intent Fusion
│   │   ├── pose_protocol.py          #   정적 자세 판정 경계(torch 무관)+기울기 신뢰·feature
│   │   └── pose_classifier.py        #   정적 자세 MLP 추론(torch, ml extra)·전처리 대조
│   │   # pose_state.py: 시간축 상태기계(클릭/드래그/스크롤/전이, 순수)
│   ├── runtime_protocol/             # 3인: Runtime & Device Protocol
│   │   ├── capture/                  # 카메라·timestamp·bounded queue
│   │   ├── protocol/                 # capability·TTL·ACK·dedup
│   │   ├── adapters/                 # Windows·SmartThings·WiZ(전구 로컬 UDP) 실제 실행
│   │   └── telemetry/                # trace·상태 전이·latency
│   ├── runtime/                      # composition root: 기기 레지스트리·executor 배선(devices.py)
│   ├── calibration/                  # 기기 등록 및 사용자 calibration
│   ├── pointer/                      # 커서·pinch click·drag 연속 제어
│   └── monitoring/                   # 로컬 모니터링·시연 UI
│       # pose_control.py: 자세→실제 OS 입력
│       # demo_bridge.py: 실시간 gaze·gesture 스트림→FusionEngine 배선(시연 탭 코어, Qt 무관)
│       # execute_worker.py: 기기 명령 비동기 실행(UDP 타임아웃 GUI 프리징 방지)
│       # virtual_bulb.py: dispatch된 Intent를 화면에 시각화(명령 기준, 실물 응답 아님)
│       # demo_panel.py: 시연 탭 위젯
├── tests/
│   ├── unit/                         # 모듈 내부 로직
│   ├── contract/                     # producer-consumer 메시지 호환성
│   ├── integration/                  # 전체 로컬 pipeline·adapter 경계
│   └── replay/                       # trace replay·성능 지표
├── training/                         # 오프라인 학습(src/jarvis 밖). train_pose.py: 정적 자세 분류기
└── tools/                            # calibration·benchmark·모델 준비 CLI
```

## 소유권 규칙

| 경로 | 주 담당 | 변경 시 필요한 협의 |
| --- | --- | --- |
| `src/jarvis/gaze/**` | Gaze | 모듈 내부 변경은 독립 진행 |
| `src/jarvis/gesture_fusion/**` | Gesture·Fusion | 모듈 내부 변경은 독립 진행 |
| `src/jarvis/runtime_protocol/**` | Runtime·Protocol | 모듈 내부 변경은 독립 진행 |
| `src/jarvis/contracts/**` | 공동 | 송신·수신 담당자 교차 검토 필수 |
| `src/jarvis/calibration/**` | Gaze 중심 공동 | Runtime·Monitoring 영향 검토 |
| `src/jarvis/pointer/**` | Gesture·Fusion 중심 공동 | Gaze gate·Windows adapter 영향 검토 |
| `src/jarvis/monitoring/**` | 공동 | 실제 상태를 가장하지 않는 범위에서 진행 |
| `documents/interface-contract.md` | 공동 | 계약 코드보다 먼저 변경 |
| `documents/decisions.md` | 공동 | threshold·포맷·범위 변경 이유 기록 |

## 의존 방향

```text
gaze ────────────────┐
                     ├─> contracts <─ runtime_protocol
gesture_fusion ──────┘

calibration -> gaze
pointer -> contracts + runtime_protocol/adapters
runtime -> contracts + runtime_protocol (composition root: 기기 레지스트리·executor)
monitoring -> contracts + telemetry + gesture_fusion + runtime (시연 배선)
```

세 핵심 모듈은 서로의 내부 파일을 직접 import하지 않는다. 데이터 교환은
`jarvis.contracts`를 통하고, 실제 조립은 `jarvis.runtime`의 composition root에서 수행한다.
`monitoring`의 시연 탭(`demo_bridge`)은 공동 소유 배선 계층으로, 실시간 두 계약 스트림을
`FusionEngine`에 물려 composition root의 `IntentExecutor`까지 잇는다.


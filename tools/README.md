# Developer tools

Calibration, trace replay, benchmark, 모델 준비처럼 제품 런타임 밖에서 실행하는 도구를 둔다.
도구가 생성한 합성 trace 또는 장애 주입 결과는 실제 사용자 데이터와 명확히 구분한다.

## Gaze tools

개발 설치 후 `jarvis-gaze` 명령을 사용한다.

```powershell
pip install -e ".[vision,dev]"
jarvis-gaze inspect-head-pose --model models/face_landmarker.task
jarvis-gaze calibrate laptop --model models/face_landmarker.task
jarvis-gaze calibrate room.bulb --model models/face_landmarker.task
jarvis-gaze evaluate --input data/evaluation/gaze.csv --dataset-id bright-01 --conditions "bright, no glasses, 60cm"
```

평가 CSV의 필수 열은 `frame_id,timestamp_ms,predicted_target,ground_truth_target`이다.
`inspect-head-pose`는 실제 카메라에서 좌우·상하 움직임에 따른 yaw/pitch 부호와 축을
확인하기 위한 진단 명령이다. 모델 파일 확보 방법은 `models/README.md`를 따른다.

## Pipeline monitor (실시간 데스크탑 앱)

웹캠을 실시간으로 띄우고 파이프라인 각 단계를 시각적으로 검증하는 데스크탑 GUI
(PySide6). 구현은 `src/jarvis/monitoring/`에 있고 `jarvis-monitor` 명령으로 실행한다.

```powershell
pip install -e ".[ui,dev]"        # PySide6 + opencv (웹캠·UI만)
pip install -e ".[ui,vision,dev]" # + mediapipe (gaze 파이프라인 라이브까지)
jarvis-monitor                    # 데스크탑 창 실행
jarvis-monitor --camera 1         # 카메라 장치 인덱스 지정
jarvis-monitor --model models/face_landmarker.task --profiles data/calibration/profiles.json
```

화면 구성(탭):

- **실시간**: 가운데 라이브 웹캠. gaze가 LIVE면 웹캠 위에 **시선 방향 화살표·머리각·
  Lock 상태**가 오버레이된다(추적 손실은 빨간 배너로 정직하게 표시). 오른쪽 사이드바에
  인식된 제스처 목록, 하단에 시스템 메시지 패널.
- **Gaze 파이프라인**: 실제 엔진이 프레임마다 만들어내는 **모든 중간값**을 단계별로 본다.
  - Landmarks(2a): `face_detected`, yaw/pitch/roll, 홍채 L/R, tracking confidence
  - Gaze Vector(2b)·Smoothing(2c): 방향 벡터, confidence·stability 게이지, 버퍼 채움
  - Target 분류(2d): top-1 확률·margin 바(임계선 0.80/0.20), 기기별 각거리, **UNKNOWN 사유**
  - Gaze Lock(2e): 6개 상태 머신 스트립(현재 상태 하이라이트) + 잠긴 기기
  - TargetEstimate(2f): Fusion으로 나가는 실제 계약 메시지
  여기서 보이는 TargetEstimate는 `GazeTargetingEngine.process`가 내보내는 값과 동일하다
  (별도 근사가 아니라 같은 코드 경로를 계측한 것).
- **파이프라인**: 단계별 실행 가능 여부 카드(`LIVE`/`DEGRADED`/`UNAVAILABLE`/`ERROR`)와,
  아직 흐르지 않는 메시지 계약(`GestureEstimate`·`Intent`·`Command`)의 실제 필드 형태.
- **지연·어댑터**: 실측 지연(capture→inference p50/p95/p99, SLO 참고선)과 어댑터 준비 상태
  (Windows 입력, SmartThings 토큰 유무·대상 기기 이름 — **토큰 값은 비노출**), safe-default 안내.

gaze 파이프라인을 LIVE로 만들려면:

1. `pip install -e ".[ui,vision,dev]"` (mediapipe 설치)
2. `models/face_landmarker.task` 확보 (`models/README.md` 참고)
3. `jarvis-gaze calibrate <device_id> ...`로 `data/calibration/profiles.json` 생성 →
   Target 분류가 UNKNOWN을 벗어나 실제 기기를 가리키기 시작한다.

동작 원칙:

- 실제로 도는 것만 표시: 웹캠 캡처, gaze 엔진(위 조건 충족 시), 어댑터/설정 감지, 지연 실측.
- 아직 없는 것(정직하게 표시): **Gesture·Fusion(2인 파트 미구현)** → 제스처 사이드바는
  "제스처 모듈 미구현", 파이프라인 탭은 `UNAVAILABLE`, 계약 형태만 참고로 보여준다.
- 인식되지 않은 것을 인식된 것처럼 꾸미지 않는다. 추적 손실·카메라 열기 실패·모델/프로파일
  부재를 값으로 감추지 않고 그대로 드러낸다. SmartThings 토큰 등 비밀값은 화면·로그에 노출하지 않는다.

라이브 연결 지점: 제스처 사이드바는 `monitoring/gesture_source.py`의 `GestureSource`에
바인딩되어 있다. 2인 파트가 완성되면 `GestureEstimate`를 어댑트하는 실제 소스로
`NullGestureSource`를 교체하기만 하면 UI는 그대로 채워진다.

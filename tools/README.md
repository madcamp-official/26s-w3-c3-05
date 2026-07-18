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
pip install -e ".[ui,dev]"        # PySide6 + opencv
jarvis-monitor                    # 데스크탑 창 실행
jarvis-monitor --camera 1         # 카메라 장치 인덱스 지정
```

화면 구성:

- **실시간 탭**: 가운데 라이브 웹캠(FPS·해상도 HUD 오버레이), 오른쪽 사이드바에
  인식된 제스처 목록, 하단에 시스템 메시지 패널.
- **파이프라인 탭**: 단계별(Capture·Gaze·Gesture·Fusion·Protocol·Adapters) 실제
  실행 가능 여부 카드. `LIVE`(초록)/`DEGRADED`(호박, 의존성·모델·설정 부족)/
  `UNAVAILABLE`(회색, 미구현)/`ERROR`(빨강)로 정직하게 표시한다.

동작 원칙:

- 지금 실제로 도는 것: 웹캠 캡처, 어댑터/설정 감지, 시스템 메시지. 실제 데이터만 쓴다.
- 아직 없는 것(정직하게 표시): **Gesture·Fusion(2인 파트 미구현)** → 제스처 사이드바는
  "제스처 모듈 미구현"으로, 파이프라인 탭은 `UNAVAILABLE`로 나온다. Gaze는 mediapipe
  (`vision` extra)와 `face_landmarker.task` 모델이 있어야 `LIVE`가 된다.
- 인식되지 않은 것을 인식된 것처럼 꾸미지 않는다. 카메라 열기 실패도 메시지 패널에
  그대로 남긴다. SmartThings 토큰 등 비밀값은 화면에 노출하지 않는다.

라이브 연결 지점: 제스처 사이드바는 `monitoring/gesture_source.py`의 `GestureSource`에
바인딩되어 있다. 2인 파트가 완성되면 `GestureEstimate`를 어댑트하는 실제 소스로
`NullGestureSource`를 교체하기만 하면 UI는 그대로 채워진다.

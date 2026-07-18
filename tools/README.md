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

## Pipeline monitor (디버깅 UI)

파이프라인 각 단계(캡처·Gaze·Gesture·Fusion·Command·Adapter·Latency·Trace)의 상태를
한 화면에서 시각적으로 확인하는 디버깅 대시보드. 외부 의존성 없이 자체 완결 HTML을
생성한다. 구현은 `src/jarvis/monitoring/`에 있고 `jarvis-monitor` 명령으로 실행한다.

```powershell
pip install -e ".[dev]"

# 1) HTML 파일로 생성해서 브라우저로 연다
jarvis-monitor --open                       # monitor.html 생성 후 브라우저 실행
jarvis-monitor -o build/monitor.html        # 출력 경로 지정

# 2) 라이브 새로고침 서버 (2초마다 자동 갱신)
jarvis-monitor --serve 8000                 # http://127.0.0.1:8000
jarvis-monitor --serve 8000 --refresh 1     # 갱신 주기 1초
```

- 현재는 **대표 목업 스냅샷**(`monitoring/demo.py`)을 렌더링한다. 화면 상단에
  `source: mock ...`으로 표시되며, Gesture·Fusion 패널은 2인 파트 미구현이라
  `미구현 · mock` 배지가 붙는다. 라이브 배선 후에는 `monitoring/cli.py`의
  `current_snapshot()`만 실제 스냅샷 빌더로 교체하면 되고 렌더링 경로는 그대로다.
- 정직성 원칙상 `UNKNOWN`·`UNVERIFIED`·`FAILED`·`REJECTED`를 색만 다르게 그대로
  표시한다. 상태를 임의로 성공 처리하지 않는다. SmartThings 토큰 등 비밀값은 화면에
  절대 노출하지 않는다.

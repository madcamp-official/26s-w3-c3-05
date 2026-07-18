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

# reference_port — 참고 레포 방법론 이식 (실험)

Kazuhito00의 `hand-gesture-recognition-using-mediapipe`가 쓰는 **정적 손모양
KeyPoint 분류기** 방식을 이 프로젝트 스택 위에 이식해, "참고 레포 방법론이 우리
엔진(Tasks API) 위에서도 잘 인식하는가"를 A/B로 확인하기 위한 **실험용** 패키지다.
프로덕션 `gesture_fusion`(동적 제스처 TCN)과는 완전히 별개다.

## 왜 "그대로 이식"이 아니라 "방법론 이식"인가 (2026-07-20 조사)

| | 참고 레포 | 이 프로젝트 |
|---|---|---|
| mediapipe | 0.8.4 (Solutions API 포함) | 0.10.35 (slim, `solutions/` 없음 — `tasks`만) |
| 랜드마크 엔진 | `mp.solutions.hands` (레거시) | Tasks API `HandLandmarker` |
| 분류기 런타임 | TensorFlow 2.9 / TFLite | (이식본은) 순수 numpy |
| Python | 3.7~3.9 | 3.12 |

참고 레포의 **랜드마크 엔진**(mediapipe 0.8.4 Solutions)은 Python 3.12·Apple
Silicon·slim mediapipe에서 실행 불가라 이식할 수 없다. 따라서 랜드마크는 이
프로젝트의 Tasks API를 그대로 쓰고, **이식한 것은 참고 레포의 방법론**이다:

1. **max-abs 2D 정규화** (`preprocess_landmark_max_abs`) — 참고 `pre_process_landmark`
   그대로: 손목 원점화 → flatten → 최대 절댓값으로 나눔.
2. **학습된 KeyPoint MLP** (`ReferenceKeyPointClassifier`, 42→20→10→3) — 참고 레포에
   포함된 학습 가중치를 hdf5에서 추출(`keypoint_weights.npz`)해 numpy로 forward를
   재구현. 라벨: Open / Close / Pointer (`keypoint_labels.csv`).

## 검증

`test_reference_port.py`가 추출 가중치 + numpy forward를 참고 레포 학습 데이터
(`keypoint.csv`)로 검증한다 — 실측 재현 정확도 **96.5%**(클래스 0~2). 즉 가중치
추출과 forward 재구현이 정확하다. (학습 데이터는 저장소 밖 경로라 없으면 해당
테스트는 skip.)

## 라이브 데모 (웹캠으로 직접 A/B)

```bash
# 프로젝트 루트(models/hand_landmarker.task 있는 곳)에서:
python -m jarvis.reference_port.demo            # 웹캠 0번
python -m jarvis.reference_port.demo --model models/hand_landmarker.task --device 0
```

웹캠에 손을 비추면 Tasks API 랜드마크 → 참고 정규화 → 참고 분류기로 Open/Close/
Pointer를 매 프레임 오버레이한다(거울상 표시, ESC 종료). 참고 레포 기본 검출
신뢰도 0.7을 쓴다(우리 기본 0.5보다 높음 — 이식 비교 변수).

## 알아둘 한계

- 분류 대상은 참고 레포가 학습한 **정적 손모양 3종**뿐이다. swipe·rotate 같은
  **동적 제스처는 이 경로로 인식하지 못한다**(그건 `gesture_fusion`의 TCN 담당).
- 분류기는 참고 레포의 **레거시 Solutions API 랜드마크**로 학습됐고, 여기선
  **Tasks API 랜드마크**를 먹인다. 두 API는 21점 토폴로지가 동일해 호환되지만,
  좌표 분포에 미세한 차이가 있을 수 있다 — 데모의 인식 품질로 그 영향을 관찰한다.

## 참고 레거시 엔진 진짜 실행 (`legacy_engine_demo.py`)

참고 레포의 **랜드마크 엔진 자체**(`mp.solutions.hands`)는 mediapipe 0.10.35(slim)엔
없지만, **solutions가 살아있는 구버전**(예: `mediapipe==0.10.14`)은 arm64/py3.12에
설치되고 레거시 엔진이 그대로 돈다(0.8.4/x86/Docker 불필요). 프로젝트 메인 venv는
Tasks API(0.10.35)를 유지해야 하므로 레거시 엔진은 **별도 격리 venv**로 돌린다.

```bash
# 1) 격리 venv 생성 + solutions 포함 mediapipe & opencv 설치
python -m venv /tmp/legacy-venv
/tmp/legacy-venv/bin/pip install "mediapipe==0.10.14" opencv-python

# 2) 참고 레거시 엔진 라이브 데모 (mp.solutions.hands)
/tmp/legacy-venv/bin/python src/jarvis/reference_port/legacy_engine_demo.py --device 0
```

A/B: Tasks API 쪽은 `compare.py`(메인 venv), 레거시 엔진은 위 데모(격리 venv)로
각각 띄워 눈으로 비교한다. 두 엔진은 mediapipe 버전이 달라 한 프로세스에 공존 불가.

`legacy_engine_demo.py`는 mediapipe를 지연 import하고 `reference_port/__init__`에서
import하지 않으므로, 메인 환경의 정상 테스트/린트 대상이 아니다.

## 가중치 재추출 (참고)

`keypoint_weights.npz`는 참고 레포 hdf5에서 뽑은 것이다. 재추출하려면 `h5py`가
필요하지만(1회성), 런타임은 numpy만 쓴다:

```python
import h5py, numpy as np
f = h5py.File(".../keypoint_classifier.hdf5")
w = f["model_weights"]
np.savez("keypoint_weights.npz",
    W0=w["dense/dense/kernel:0"][:], b0=w["dense/dense/bias:0"][:],
    W1=w["dense_1/dense_1/kernel:0"][:], b1=w["dense_1/dense_1/bias:0"][:],
    W2=w["dense_2/dense_2/kernel:0"][:], b2=w["dense_2/dense_2/bias:0"][:])
```

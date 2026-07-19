# Gesture TCN 학습 파이프라인

`jarvis.gesture_fusion.model.CausalTCNGestureModel`을 학습시키는 오프라인 파이프라인.
`src/jarvis`(런타임 패키지) 밖에 있다 — 이유는 [../documents/decisions.md](../documents/decisions.md)
2026-07-19 항목 참고.

## 준비물

```bash
pip install -e ".[training]"   # torch + mediapipe + tensorboard + pandas
```

- **Jester 데이터셋**: 이 저장소에는 포함되지 않는다. 학습을 실제로 돌리는 VM에
  [20BN-Jester-v1](https://www.qualcomm.com/developer/software/jester-dataset)을
  받아두고, `--jester-dir` 또는 `JARVIS_JESTER_DIR` 환경변수로 경로를 지정한다.
- **손 랜드마크 모델**: `models/hand_landmarker.task` (models/README.md 참고, gitignored).

## 실행 순서

1. **Jester 클래스 매핑 확정** — `training/data/jester_labels.py`의
   `JESTER_TO_OUR_LABEL`에서 `None`(제외)으로 둔 클래스를 필요에 따라 우리
   라벨(현재는 `"none"`만 열려 있음)로 편집한다.

2. **원시 landmark 추출** (몇 시간 소요, 재개 가능):
   ```bash
   python -m training.extract.extract_jester --jester-dir /data/20bn-jester-v1 --limit 50   # 스모크 테스트
   python -m training.extract.extract_jester --jester-dir /data/20bn-jester-v1 --workers 8   # 본 실행
   ```
   결과는 `training/cache/jester/*.npz` + `training/cache/jester_manifest.csv`(검출 실패 클립과 사유 포함).

3. **Jester 사전학습**:
   ```bash
   python -m training.train --stage pretrain
   ```
   `models/gesture_tcn_jester.pt` 생성.

4. **웹캠 파인튜닝 데이터 녹화** (팀원별로):
   ```bash
   python -m training.record_webcam_clips --person-id <이름> --model models/hand_landmarker.task
   ```
   결과는 `training/cache/webcam/*.npz` (Jester 캐시와 같은 포맷).

5. **파인튜닝**:
   ```bash
   python -m training.train --stage finetune --init-from models/gesture_tcn_jester.pt
   ```
   `models/gesture_tcn_finetuned.pt` 생성.

6. **평가**:
   ```bash
   python -m training.evaluate --checkpoint models/gesture_tcn_finetuned.pt
   ```

## TensorBoard

```bash
tensorboard --logdir training/runs
```

## 왜 원시 landmark를 먼저 캐싱하나

`extract_jester.py`는 `normalize_hand`/`HandFeatureExtractor`를 호출하지 않고 MediaPipe가
낸 원시 `(21, 2)` 좌표만 캐싱한다. Feature 조립(정규화·평활화·속도·가속도)은
`training/dataset.py`가 캐시를 읽을 때마다 `jarvis.gesture_fusion`의 실제 함수로
재생한다 — 그래야 `GestureConfig`(One-Euro 파라미터 등)가 나중에 조정돼도 몇 시간짜리
MediaPipe 배치를 다시 돌릴 필요가 없다. 학습·추론이 같은 전처리 코드를 타므로
모델 재현성도 자동으로 보장된다(development-principles.md 7.3).

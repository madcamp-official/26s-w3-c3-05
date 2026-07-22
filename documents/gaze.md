# Gaze Targeting — 담당: 1인

README [7장 핵심 기능 1](../README.md)의 구현 설계/진행 상황을 기록하는 문서.
다른 모듈과 주고받는 데이터 포맷은 여기가 아니라 [interface-contract.md](interface-contract.md)에 정의한다.

## 담당 범위 (README 12장)

- Face·iris landmark
- head pose
- gaze feature 정규화
- 기기별 calibration
- target classifier
- `UNKNOWN` rejection
- gaze smoothing
- Gaze Lock
- Target Selection Accuracy 평가

## 설계 노트

- **시선 방향 벡터 합성 (2026-07-18 팀 합의)**: 머리 yaw/pitch와 눈-머리 상대 오프셋을 따로 feature로 두지 않고 하나의 시선 방향 단위 벡터로 합성한다. 등록 시(고개를 돌려 봄)와 실사용 시(고개는 그대로, 눈짓만)의 행동이 달라도 같은 방향이면 같은 벡터가 나오게 하기 위함. 기기 prototype과의 비교는 코사인 유사도(내적)로 한다. 상세는 README 7장.

구현은 `src/jarvis/gaze/`와 `src/jarvis/calibration/`에 있다. 파이프라인 순서:

```
FaceObservation (landmarks.py, MediaPipe Face Landmarker)
→ compose_gaze_vector (features.py) — 머리 회전 ⊕ 눈-머리 상대 오프셋 → 단위 벡터
→ GazeSmoother (smoothing.py) — confidence-가중 이동 평균 + stability
→ TargetClassifier (classifier.py) — 코사인 유사도 + 등록 분산 정규화 + softmax + UNKNOWN
→ GazeLockStateMachine (lock.py) — SEARCHING→CANDIDATE→TARGET_LOCKED→GESTURE_WAIT→EXPIRED/COMMITTED
→ GazeTargetingEngine.process() (engine.py) — 위 전부를 조립해 TargetEstimate 방출
```

- `jarvis.gaze.landmarks`만 mediapipe(`vision` extra)를 import한다. 나머지는 순수
  `FaceObservation` 값만 다루므로 카메라·모델 파일 없이 단위 테스트한다.
- Calibration(`src/jarvis/calibration/session.py`)은 raw 프레임을 모았다가
  `DeviceGazeProfile`(mean_direction + 각도 분산)로 축약한 뒤 버린다. 저장/불러오기는
  `src/jarvis/calibration/profiles.py`가 README 7장 JSON 포맷대로 처리한다.
- Target Selection Accuracy는 `jarvis.gaze.evaluation.compute_target_selection_accuracy`로
  계산하며 dataset_id·조건을 결과에 강제로 남긴다.
- `UNKNOWN`은 기기 간 상대 확률(`unknown_probability_threshold`)뿐 아니라 가장 가까운
  등록 방향과의 절대 각도(`unknown_max_angle_deg`, 기본 25도)도 함께 검사한다. 등록
  기기가 하나일 때 상대 확률이 항상 1.0이 되는 경우에도 먼 시선을 거부하기 위해서다.
- 3초 확정 대상은 새 후보가 dwell을 모두 채울 때까지 sticky하게 유지한다. 새 후보가
  잠깐 `UNKNOWN`/저신뢰로 취소돼도 이전 대상은 유지되고, 새 후보가 3초를 채운 프레임에서
  공백 없이 원자적으로 교체된다. 단, Gaze classifier가 2초 연속 `UNKNOWN`이면 확정
  target을 해제한다. 그 전에 알려진 target이 나오면 UNKNOWN 타이머는 초기화된다.
  이 규칙은 Gaze 모듈에만 적용하며 Gesture/Fusion 로직은 변경하지 않는다.
- `target_lock_ttl_ms`는 확정 선택 자체를 1.5초마다 지우는 타이머가 아니라, gesture
  wait와 Fusion 입력 스트림 중단 시 안전하게 intent를 거부하는 유효 시간이다.
- `jarvis-gaze calibrate`는 카메라 관측을 기기 프로필로 축약해 저장하고,
  `inspect-head-pose`는 실카메라 축/부호 점검값을 출력한다. `evaluate`는 정답 CSV에서
  dataset_id·환경 조건을 포함한 재현 가능한 정확도 JSON을 만든다(`tools/README.md`).
- 알려진 미검증 항목: `landmarks.py`의 head yaw/pitch/roll 부호 규약은 실제 카메라로
  검증하지 않았다 — models/README.md의 `face_landmarker.task` 항목 참고.

## 진행 상황

- [x] Face·iris landmark 어댑터 (`landmarks.py`)
- [x] gaze feature 정규화 / 단위 벡터 합성 (`features.py`)
- [x] gaze smoothing (`smoothing.py`)
- [x] 기기별 calibration (`calibration/session.py`, `calibration/profiles.py`)
- [x] target classifier (`classifier.py`)
- [x] `UNKNOWN` rejection (classifier의 `unknown_probability_threshold`)
- [x] Gaze Lock 상태 머신 (`lock.py`)
- [x] Target Selection Accuracy 평가 함수 (`evaluation.py`)
- [ ] 실제 카메라로 head pose 부호/축 검증 (pitch는 2026-07-18 실측에서 상하 반전 확인 후
  화면 위쪽이 양수가 되도록 수정 완료, yaw·roll 최종 확인 필요)
- [ ] Gesture·Fusion·Runtime과의 실제 통합(코드 조립은 Runtime composition root 몫)
- [ ] 환경 변화(조명/안경/거리) 조건에서 실측 Target Selection Accuracy 수집
- [x] 2단계 테두리 등록(자세·거리별 20초 + 고개 고정 정밀 경계 16초)
- [x] 8D target profile(gaze/head/face scale/face center) + Mahalanobis 판정
- [x] 겹침 시 홍채 settle 방향 tie-break
- [x] 3D geometry ↔ 각도 모드 통합 classifier(`effective_distance_and_variance`) —
  origin 없거나 깊이 퇴화 시 프레임 단위 폴백
- [x] 모니터링 앱(`gaze_probe.py`/`app.py`) origin 배선 + 재시작 후 3D geometry 보존

## 이슈 / 의사결정 필요 사항

- head yaw/pitch/roll 부호 규약이 실 카메라와 맞는지 아직 확인되지 않음 — Day 1 통합
  테스트에서 확인되면 여기와 models/README.md를 갱신할 것(있으면 [decisions.md](decisions.md)로 옮기기).
## Current deterministic target profile (2026-07-21)

현재 런타임에는 MLP, Ridge, Linear-softmax가 없다. 물체별로 다음 8차원 feature의
평균·공분산·Mahalanobis threshold만 저장한다.

```text
[gaze_yaw, gaze_pitch, head_yaw, head_pitch, head_roll,
 face_scale, face_center_x, face_center_y]
```

1단계(2026-07-22 개정)에서는 물체 **중앙 한 점을 계속 응시한 채** 고개를 좌우
끝까지·위아래로 돌리고 카메라 거리를 바꿔, 자세·거리 문맥과 자세별 gaze 편향
(pose correction)을 모은다. 2단계에서는 고개를 고정하고 눈으로 테두리를 한 바퀴
돌아 최종 area를 확정한다. 2단계 yaw/pitch 경계에서 95퍼센타일 밖 outlier를 제거하고
convex hull 꼭짓점만 JSON에 저장한다. runtime은 polygon 안 후보만 만들고 8D profile로 검증·
정렬한다. area 밖 후보는 profile이 가까워도 항상 `UNKNOWN`이다. area 안에서는 head
pose가 달라도 눈이 보정한 최종 gaze를 우선하며, 8D 문맥은 hard rejection이 아니라
겹친 후보의 순위와 신뢰도에만 사용한다. 겹침에서는 target
중심 방향과, 빠른 홍채 이동이 정지할 때 저장한 마지막 이동 방향을 추가 근거로 쓴다.

테두리 ray는 서로 다른 표면점을 향하므로 새 등록에서는 3D 한 점으로 삼각측량하지
않는다. 기존 JSON의 `position_3d`는 하위 호환으로 읽을 수 있다.

## Historical device registration (superseded on 2026-07-21)

Demo target registration follows README section 7 as two strictly separated phases:
20 seconds on one center point for MLP/center/3D, followed by 16 seconds tracing
the four edges with the head still for the angular target area.

Implemented files:

- `src/jarvis/gaze/direction.py`: vector/yaw-pitch conversion used only at the registration/debug boundary.
- `src/jarvis/calibration/registry.py`: target add/update/rename/delete persistence, JSON auto-load, legacy profile migration,
  and conversion back to README-style `DeviceGazeProfile`.
- `src/jarvis/calibration/target_registration.py`: phase-separated center/boundary samples,
  minimum frame count, confidence/closed-eye/jump filtering, center median, and edge area.
- `src/jarvis/gaze/classifier.py`: registered target matching uses cosine similarity normalized by stored variance,
  then rejects `UNKNOWN` when the nearest registered direction is too far or the first/second target margin is too small.
- `src/jarvis/gaze/smoothing.py`: confidence-aware EMA smoothing is enabled before classification.
- `src/jarvis/gaze/smoothing.py`: short blinks hold the last stable gaze briefly, and tiny gaze changes are absorbed by a
  small angular deadzone to reduce jitter.
- `src/jarvis/gaze/lock.py`: three-second dwell confirmation and hysteresis are handled by the existing state machine.
- `src/jarvis/monitoring/`: debug UI can register/reregister/rename/delete targets and show the live gaze ray,
  candidate/lock state, and pipeline diagnostics without drawing artificial target-area circles.

MVP operating assumptions:

- The camera is fixed during the demo.
- User changes are announced manually, so automatic face identity/profile switching is out of MVP scope.
- Objects are registered every time they are newly added or moved.
- Real-camera yaw/pitch sign and scale still need one final fixed-camera sanity check before demo.
- `CalibratedGaze` uses a validated residual MLP by default when at least three labeled target directions are available;
  otherwise it falls back to Ridge or the raw geometric vector.

## Historical residual MLP vector calibration (removed on 2026-07-21)

`src/jarvis/gaze/mlp_calibration.py` implements a NumPy-only `12 → 24 → 12 → 2`
network. Input is raw yaw/pitch, both iris offsets, head yaw/pitch/roll, face center,
and face scale. Output is not an absolute direction but `(delta_yaw, delta_pitch)`;
the geometric vector therefore remains the safe baseline. Corrections are capped at
35 degrees.

Training data remains in `data/calibration/gaze_regressor.json`. Each sample stores
the 13-value Ridge-compatible feature vector, target yaw/pitch pseudo-label, and (for
new records) `target_id`. At registration completion the current target's identified
samples replace its previous identified samples, all targets are replayed, and both
Ridge and MLP are retrained. Validation is split within each target direction. A new
MLP is activated only when it improves held-out angular error over the raw vector.
Training runs only after registration; live inference is three matrix multiplies.
Only phase-1 center-point frames become MLP rows. Phase-2 edge frames are deliberately
excluded so distinct boundary directions are never mislabeled as the center.

The debug panel exposes `vector model: mlp/ridge/geometric`, raw yaw/pitch, final
yaw/pitch, and whether correction was applied. The checkbox disables correction for
an immediate A/B comparison without changing or deleting the dataset.

## Historical 3D object position update (new registration no longer uses it)

현재 등록 1단계는 중앙의 한 점을 20초 동안 보며 다양한 자세·거리에서 머리
이동(parallax) 광선을 모아 3D 위치를 삼각측량한다. 2단계 경계 프레임은 서로 다른
실제 표면점을 향하므로 삼각측량과 MLP 중앙점 정답에서 제외한다. 3D 신뢰도가 낮으면(baseline 부족,
광선이 거의 평행, 잔차 과다) 2026-07-19의 각도 기반 등록으로 조용히 대체한다
(documents/decisions.md 2026-07-20 항목들).

핵심 설계: 3D 위치+반경도 매 프레임 `(각도 거리, 분산)` 쌍으로 환산해 기존
classifier의 Gaussian score·softmax·UNKNOWN 임계값 로직을 그대로 재사용한다 —
3D 전용 판정 경로를 새로 만들지 않았다.

구현 파일:

- `src/jarvis/gaze/landmarks.py`: `translation_from_transform()`이 MediaPipe
  facial transformation matrix의 `[:3, 3]`(기존에는 버리던 부분)을 머리의
  카메라 기준 3D 위치 근사(`FaceObservation.head_position_mm`)로 추출한다.
- `src/jarvis/gaze/features.py` / `smoothing.py`: `GazeVector`/`SmoothedGaze`에
  `origin` 필드 추가(전부 optional, 기본 None — 하위 호환). smoothing은 버퍼의
  모든 프레임에 origin이 있을 때만 confidence-가중 평균을 낸다.
- `src/jarvis/calibration/triangulation.py`: 여러 시선 광선(origin, direction)의
  최소자승 교차점(`np.linalg.lstsq`, 조건 분기 없음)과 세 가지 독립 품질 지표
  (baseline_mm, min_eigenvalue, residual_rms_mm)를 계산한다. 하나만으로는 서로
  다른 퇴화 상황(머리 고정+눈만 이동 vs 머리 이동+물체가 멀어 광선이 평행)을
  잡아내지 못해 셋 다 게이트로 쓴다.
- `src/jarvis/calibration/target_registration.py`: `finalize()`가 각도 기반
  direction/spread(항상 계산, 기존 동작 그대로)에 더해 3D 삼각측량을 시도하고,
  품질 기준을 만족할 때만 `TargetRecord.position_3d`를 채운다. 실패해도
  `session.triangulation_result`에 진단 정보(baseline·잔차·고유값)를 남겨
  왜 대체됐는지 보여준다(성공을 지어내지 않는다).
- `src/jarvis/calibration/registry.py`: `TargetGeometry3DRecord`(plain tuple —
  `TargetRecord`가 `asdict()`로 직접 JSON 직렬화되므로 numpy 배열 불가)가
  `position_3d`로 영속화된다. 예전 JSON(필드 없음)도 `None`으로 그대로 로드된다.
- `src/jarvis/gaze/classifier.py`: `TargetGeometry3D` 등록 시
  `effective_distance_and_variance()`가 현재 origin에서 물체 중심까지의 방향을
  매 프레임 새로 계산(`atan(radius_mm/depth)`로 각도 분산 변환, 각도 모드 최소
  퍼짐 이하로 떨어지지 않도록 바닥 적용)한다. origin이 없거나 깊이가 퇴화하면
  등록 시 저장한 고정 방향(각도 모드)으로 자동 대체된다 — 이 폴백은 기기 단위
  (등록 품질 미달)와 프레임 단위(이번 프레임만 origin 없음) 둘 다에서 동작한다.
- `src/jarvis/monitoring/gaze_probe.py` / `app.py`: 실제 데모 앱은
  `GazeTargetingEngine`을 감싸지 않고 같은 단계를 독립적으로 재구현하므로
  (`evaluate()`), 여기서도 origin을 별도로 배선하고 `_device_details`가
  `classify()`와 같은 `effective_distance_and_variance`를 재사용하도록
  맞췄다 — 그러지 않으면 테스트는 통과해도 실제 데모는 계속 각도 전용으로만
  동작하는 채로 남는다. `GazeProbe._load_profiles`도 평평한 `profiles.py`
  로더 대신 `TargetRegistry`로 바꿔 앱을 재시작해도 3D geometry가 사라지지
  않게 했다.

실측 캘리브레이션 값(합성 광선으로만 검증, 실제 카메라 미검증):
`minimum_triangulation_baseline_mm=60`, `minimum_triangulation_eigenvalue=0.004`,
`maximum_triangulation_residual_mm=35` — Day 1 통합 테스트에서 재보정 필요할 수 있음
(config.py 필드 docstring에 근거 수치 기록).

## Current Gaze Tuning Values (2026-07-20)

This section records the actual values currently used by the gaze-target branch.
The single source of truth is `src/jarvis/gaze/config.py`; README section 7 keeps a shorter summary.

### Gaze vector composition

| Setting | Current value | Meaning / when to adjust |
| --- | ---: | --- |
| `max_eye_offset_deg` | `45.0` | Converts iris offset `-1..1` into eye rotation degrees. Affects overall eye-motion sensitivity. |
| `head_yaw_weight` | `0.25` | Weight of head yaw in final gaze yaw. Adjust when left/right head motion is too strong or too weak. |
| `head_pitch_weight` | `0.40` | Weight of head pitch in final gaze pitch. Increase if up/down head motion is under-reflected; decrease if it dominates. |
| `horizontal_axis_sign` | `-1.0` | Sign correction between camera/MediaPipe horizontal motion and user yaw direction. Check this if left/right is reversed. |
| `head_only_confidence_scale` | `0.45` | Confidence multiplier when iris/eye data is unavailable and only head pose is used. |

Current composition formula:

```text
eye_yaw_offset_deg   = mean_iris_x * max_eye_offset_deg
eye_pitch_offset_deg = mean_iris_y * max_eye_offset_deg

final_yaw_deg   = (head_yaw_deg * head_yaw_weight + eye_yaw_offset_deg) * horizontal_axis_sign
final_pitch_deg =  head_pitch_deg * head_pitch_weight + eye_pitch_offset_deg
```

### Smoothing / blink / iris-jump handling

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `smoothing_window_frames` | `8` | Confidence-weighted moving average window. |
| `ema_min_alpha` / `ema_max_alpha` | `0.15` / `0.65` | Low-confidence frames move slowly; high-confidence frames move faster. |
| `blink_hold_ms` | `300` | Short eye-closed intervals hold the last stable gaze. |
| `blink_recovery_hold_ms` | `250` | Brief hold after reopening eyes so iris landmarks can settle. |
| `eye_closed_ratio_threshold` | `0.12` | Absolute eyelid-height/eye-width floor for eye-closed detection. |
| `blink_close_ratio` / `blink_reopen_ratio` | `0.68` / `0.82` | Adaptive close/reopen ratios relative to the user's open-eye baseline. |
| `eye_openness_baseline_decay` | `0.01` | Slow per-frame downward adaptation of the open-eye baseline. |
| `iris_jump_threshold` | `0.18` | Frame-to-frame iris-offset jump threshold. |
| `max_valid_eye_offset` | `0.55` | Reject implausible eye-edge iris offsets. |
| `tracking_loss_hold_ms` | `800` | Keep last gaze briefly during full face-landmarker dropouts. |
| `small_motion_deadzone_deg` | `1.0` | Absorb only tiny jitter without visually freezing the arrow. |

Debug monitor policy: simple `iris jump` lowers confidence so the arrow keeps moving while smoothing absorbs the jump. Eye-closed and blink-recovery frames hold the previous gaze for at most `blink_hold_ms`; the deadline is measured from the last real vector, not extended by held frames. If no stable vector exists or the hold expires while the face is still detected, the pipeline emits a low-confidence `head-only` vector. `TRACKING LOST` is reserved for an actual face-landmarker miss. Each eye has its own adaptive open baseline, and both eyes must indicate closure; one foreshortened eye during a head turn no longer freezes the vector. The debug panel exposes the active `gaze source`.

### Deterministic 8D target scoring

The traced yaw/pitch convex hull is the eligibility gate. Radial distance is zero at
the stored center and one where the center-to-gaze ray intersects the polygon boundary.
This preserves traced rectangular corners that the previous ellipse discarded. Candidates inside that area are
ranked with an 8D Mahalanobis soft-context profile: gaze yaw/pitch, head yaw/pitch/
roll, face scale, and normalized face center x/y. Per-feature covariance floors keep
gaze more discriminative while preventing degree units from erasing scale/location.
No learned classifier is trained during registration.

Overlap intent is armed above the iris start speed, emitted only after speed falls
below the stop threshold, and retained briefly. Acceleration is diagnostic only:
during deceleration it points opposite the landing direction and is unsafe as a
target bonus. Blink, recovery, iris-jump and tracking-loss frames reset settle state.

### Pose-conditioned gaze correction (2026-07-22)

`jarvis-gaze diagnose-composition` 실측(모니터 응시 + 고개 스윕, 유효 494프레임,
head yaw -38°~+45°)으로 확인한 사실:

- 중앙 |head yaw| ≤ 10°에서는 iris↔head 회귀가 R²=0.90으로 선형이고 implied
  head_yaw_weight ≈ 0.43이 나온다.
- ±15°를 넘으면 좌우 모두 회귀 기울기의 부호가 반전된다 — MediaPipe iris 추정이
  극단 눈 위치에서 포화·역행하므로, 어떤 전역 `head_yaw_weight`로도 자세 불변을
  만들 수 없다(0.43으로 올리면 중앙은 좋아지고 head yaw 35°는 -4°→-10°로 악화).
- 클램프(`max_valid_eye_offset`) 포화와 근안(近眼) 우선 가설은 기각(p95 0.38,
  양쪽 눈 모두 큰 yaw에서 무신호).

이 편향은 전역 수식으로 고칠 수 없지만 자세별로는 반복 가능하므로, 등록 1단계
샘플을 head-yaw 구간(`pose_correction_bin_edges_deg`)별로 묶어 `bin gaze 중앙값
− area 중심`을 target별 `TargetPoseCorrection`으로 저장한다(`feature_profile.py`
`build_pose_correction`). 2단계(고개 고정)의 median head yaw를 기준 자세로 삼아
그 자세에서 보정이 0이 되게 정규화하고, 런타임에는 classifier의 area 판정
직전에 현재 head yaw로 보간(piecewise-linear, 양끝 상수)한 오프셋을 gaze에서
뺀다. 8D Mahalanobis는 원시 샘플 그대로 쓴다(그 분포 자체가 편향 포함 원시
프레임으로 학습되므로). 표본이 부족한 bin은 만들지 않고, 유효 bin이 2개
미만이면 보정을 저장하지 않으며, 예전 JSON은 보정 없이(None) 그대로 로드된다.
기존 등록 target이 보정을 얻으려면 재등록해야 한다. 디버그 패널의 area 거리도
같은 보정 거리를 쓴다.

**보정은 rescue 전용이다 (2026-07-22 실측).** 재등록 검증에서 자세별 편향이
세션·순간에 따라 -8°~+4°로 요동해(응시 캡처 두 번은 서로 일치했지만 정지 응시
실사용과는 부호가 반대), 보정을 양방향으로 적용하면 원시 gaze로는 area 안이던
프레임을 밖으로 밀어내는 회귀가 실제로 발생했다. 그래서 classifier는 원시
gaze와 보정 gaze 중 **area 거리가 더 작은 쪽을 채택**한다
(`TargetClassifier.area_distance_and_gaze`, min 정책) — 보정은 OUT→IN 구조만
할 수 있고 IN→OUT을 만들 수 없다. |head yaw| > 30° 또는 head pitch > 25°의
자세는 편향 변동폭이 area 반경(6°)과 같은 규모라 보정으로 구제되지 않는다 —
데모 배치에서 그 범위의 고개 돌림이 필요 없게 하는 것이 정답이다.

**1단계 데이터는 반드시 중앙 응시 스윕이어야 한다 (2026-07-22 실측).** 첫
구현은 "테두리를 훑으며 자세를 바꾸는" 기존 1단계 데이터로 오프셋을 배웠는데,
사람은 보는 방향으로 고개를 돌리므로 bin 중앙값이 센서 편향이 아니라 "그
자세에서 보던 테두리 위치"를 학습해 부호까지 뒤집혔다(head yaw -25~-30 실제
편향 -3.4° vs 학습값 0~+3.9°). 그래서 등록 1단계 안내를 "중앙 한 점 응시 +
고개 스윕"으로 바꿨고, `build_pose_correction`은 bin 안 gaze IQR가
`maximum_bin_iqr_deg`(기본 4°)보다 넓으면 그 bin을 버린다 — 스윕 실측에서
|head yaw| ≤ 30° bin은 IQR 1~2°, 그 밖은 9~30°였다.

**재등록-검증 루프는 `jarvis-gaze verify-target`으로 돌린다.** 재등록 직후
`--label after-registration --output <a.json>`으로 응시 스윕을 1회 재검증하고,
시간이 지나거나 자세·조명이 바뀐 뒤 `--compare <a.json>`으로 다시 실행하면
bin별로 "직후부터 OUT(등록 수집 문제)"과 "직후엔 IN이었다가 OUT(세션
드리프트 — 세션 시작 재보정/온라인 편향 추정 필요)"을 갈라 판정해 준다
(`target_verification.py`). 판정은 런타임과 동일한 rescue 전용(min) 거리로
계산하며, 표본 8프레임 미만 bin은 결론에서 제외한다.

**등록 1단계는 시간제가 아니라 조건 충족식이다 (2026-07-22).** 시간제(20초)
등록은 스윕이 한쪽에 치우쳐도 완료돼, 실제로 오른쪽 bin 2개만 저장된 등록이
발생했다. `PoseCoverageTracker`(target_registration.py)가 정면/좌/우/상/하/
근/원 7개 구간의 유효 프레임을 세고, 전 구간이
`registration_coverage_min_frames`(기본 30)를 채워야 2단계로 넘어가며
`finalize()`도 미달이면 거부한다. near/far는 정면 기준 face scale 대비
배율(1.15x/0.87x)로 판정한다. 앱 상태줄이 구간별 `count/required ✓` 현황과
부족 구간을 실시간 안내한다. 완료 시 1단계 원시 샘플이
`data/calibration/raw_samples/<target>_phase1_<ts>.json`으로 저장된다.

**좌/우, 상/하는 짝 중 한쪽만 채우면 된다 (2026-07-22 2차 실측).** 물체가
카메라 기준 한쪽(예: 오른쪽)에 있으면, 반대쪽("고개 왼쪽")을 채우려면 눈이
물체 반대편에서 다시 그만큼 꺾여야 해 보상각이 40도 안팎까지 벌어진다 —
실측 사례에서 오른쪽(234/30)은 쉽게 채웠지만 왼쪽(0/30)은 "너무 어렵다"는
피드백을 받았다. `PoseCoverageTracker.complete()`/`missing_labels()`가
(좌,우)·(상,하)를 그룹으로 묶어 그룹 안 어느 한쪽이든 충족하면 그 그룹을
통과시킨다(정면·근·원은 그대로 개별 필수). `missing_labels()`는 미충족
그룹을 `"고개 왼쪽/고개 오른쪽"`처럼 묶어서 반환한다. 문턱값도
`coverage_yaw_side_threshold_deg` 20°→15°로 낮췄다.

**Ridge residual 보정은 오프라인 A/B 전용이며 런타임에 연결되지 않았다.**
`jarvis-gaze ab-residual <raw_samples.json>`이 leave-one-yaw-bin-out으로
raw / 현재 bin 보정표 / Ridge(자세·문맥 6D → delta) held-out 오차를 비교한다.
raw gaze는 입력에서 의도적으로 제외한다 — 단일 target 데이터에서 gaze를
입력에 넣으면 정답이 `delta = center − gaze`라서 모델이 항상 물체 중심만
출력하는 상수 예측기로 붕괴하고 leave-bin-out으로도 잡히지 않는다(단위
테스트로 확인). residual MLP·Ridge는 2026-07-21에 제거된 이력이 있고 자세별
편향이 세션 간 요동하므로(위), **같은 캡처 안의 A/B 통과만으로는 활성화하지
않고 다른 세션 데이터로 재확인한 뒤 결정한다.** 실측(2026-07-22, λ 0.1/1/10
전부): Ridge는 정면 bin 오차를 3.1°→6.4~6.6°로 2배 악화시켜 기각 — 전역
선형은 가장자리를 맞추려면 정면을 희생하는 구조라 재학습으로 해결되지 않는다.

**비선형 후보의 교차 세션 관문 (2026-07-22).** 채택 기준을 도구로 코드화했다:
`jarvis-gaze ab-residual --train <세션A 파일들> --eval <세션B 파일들>`이 세션
A로 학습해 세션 B에서만 평가하고, raw·bin표를 **둘 다** 이긴 bin이 과반이어야
PASS다(서로 다른 날 eval 2세션 PASS가 활성화 필요조건). 세션 데이터는 등록
export 또는 `verify-target --save-samples <파일>`(등록 없이 스윕 한 번 = 세션
파일 하나)로 모은다. 비선형 1순위 후보로 **가우시안 커널 국소 회귀**
(`train_kernel_residual`)를 구현해 뒀다 — bin 보정표(boxcar 커널)의 연속
일반화라 전역 선형의 정면-희생 문제가 구조적으로 없고, 학습 자세에서 먼
질의는 보정하지 않는다(외삽 금지). 합성 비선형 편향 테스트에서 Ridge와 yaw
단독 bin표를 모두 이김을 확인했으나, **실데이터 2세션 PASS 전까지 런타임
미연결**이다.

### Labeled debugging sessions (2026-07-22)

정확도 디버깅이 매번 "raw CSV를 밖에서 재합성"하는 일이 되지 않도록, 모니터링
앱에 라벨된 세션 레코더를 붙였다.

- 모니터링 앱 실시간 탭에서 **F9**로 녹화 시작/종료. 녹화 중 숫자키로 정답
  라벨을 표시한다: `0` = 아무것도 안 봄(기대 결과 UNKNOWN), `1`~`9` = 등록
  target n번. 라벨을 안 고르면 그 구간은 집계에서 제외된다.
- `data/diagnostics/session_*.jsonl`에 프레임별 파이프라인 전 단계(관측값,
  raw/smoothed gaze, target별 area 거리·보정 적용 여부, 분류 결과·탈락 사유,
  lock 상태)가 남는다. v2부터 눈 뜸 비율/사용자 기준선, 얼굴 중심·크기,
  실제 gaze ray 원점, smoothing 버퍼, 시선 속도·가속도·settle 상태도 함께
  기록한다. 헤더에 당시 GazeConfig와 target 프로필(보정 테이블 포함)이 통째로
  들어가므로 세션 파일 하나로 분석이 완결된다. 원본 카메라 이미지는 저장하지
  않는다.
- F9로 녹화를 끝내면 같은 실시간 탭 아래의 분석 창에 결과가 자동으로 뜬다.
  **최근 세션 분석**은 앱을 다시 실행한 뒤 마지막 JSONL을 다시 읽을 때 쓴다.
- `jarvis-gaze report <session.jsonl>`도 같은 결과를 출력한다. 라벨별 전체
  정확도와 head-yaw 보정 대조뿐 아니라 head pitch, 등록 대비 얼굴 크기,
  화면 내 얼굴 x/y 구간별 정확도·UNKNOWN·no-gaze·in-area 비율을 보여 준다.
  편향이 3도 이상 어긋난 구간, 보정 커버리지 밖, 전체보다 정확도가 20%p 이상
  낮은 자세/거리 구간은 경고하고, 실패 원인별 대표 프레임을 최대 6개 표시한다.
- 구현: `monitoring/session_recorder.py`(기록), `gaze/session_report.py`(집계),
  `AreaProfileDetail.used_gaze_*`(area 판정에 실제 쓰인 보정 전/후 gaze 노출).
- **등록 린터** (`gaze/registration_lint.py`): 등록 완료 직후 앱 로그와
  `jarvis-gaze lint-profiles`가 저장된 프로필의 품질 문제를 즉시 경고한다 —
  1단계 스윕 커버리지가 좁거나 정면(0°)이 빠짐, 기준 자세가 커버리지 밖이거나
  고개를 15° 이상 돌린 채 hull을 그림, 오프셋이 clamp(±10°)에 걸림, bin 표본
  30개 미만, area 반경이 cap에 잘림. 2026-07-22의 speaker(편측 스윕)·monitor
  (+32° bin clamp) 실패는 모두 이 린터가 등록 직후 잡는 패턴이다.
- **오프라인 리플레이** (`jarvis-gaze replay <session.jsonl> --set key=value
  [--no-pose-correction] [--output out.jsonl]`): 녹화된 세션의 원시 관측값을
  `GazeProbe.process_observation`(라이브와 동일한 상태 스레딩)으로 설정만 바꿔
  다시 판정하고, 라벨별 정확도를 원본과 비교해 보여준다(`monitoring/
  session_replay.py`). 재녹화 없이 tolerance·head 가중치·보정 on/off의 효과를
  수치로 확인하고, 재등록 직후 세션을 리플레이해 세션 드리프트를 검증하는 데
  쓴다. 스무더/lock 상태는 세션 시작부터 다시 누적되므로 첫 스무딩 윈도는
  라이브와 미세하게 다를 수 있다.

권장 측정은 target 하나당 최소 세 구간이다. 각 구간에서 콤보 또는 숫자키로
정답을 먼저 고른다.

1. 중앙 응시 5초: 정면 자세의 기준 정확도와 blink/no-gaze 비율 확인.
2. 같은 target 응시 10초: 고개 좌우·상하, 몸의 좌우, 카메라와의 거리를 천천히
   바꿔 자세/거리별 성능 저하 구간 확인.
3. `0` 라벨 10초: 등록 물체 사이와 바깥을 보며 false positive 확인.

해석할 때는 숫자 하나를 바로 조정하지 않는다. `in-area%`가 낮으면 gaze/등록
영역 문제, `in-area%`는 높은데 정확도가 낮으면 후보 순위·confidence 문제,
`no-gaze%`가 높으면 face/iris/blink 입력 문제다. 얼굴 크기 비율 한 구간에서만
무너지면 거리 보정, head pitch/yaw 한 구간에서만 무너지면 pose 보정 데이터를
먼저 다시 수집한다. 대표 실패 프레임 번호로 같은 JSONL의 원시 값을 대조한다.

The personal softmax is now only a ranker for overlapping **current spatial
candidates**. A target must first pass its saved edge-loop area gate (or its direction
variance gate when no area exists) using `target_match_tolerance`. With fewer than two
eligible candidates, deterministic area/direction matching decides; with no candidate,
the result is `UNKNOWN` regardless of softmax confidence. The model softmax is also
restricted to IDs in the current registry, so a deleted class cannot win.

Training rows are bound to a registration fingerprint derived from direction, spread,
area, feature profile, face scale, and optional 3-D geometry—not to `target_id` alone.
Re-registering the same ID replaces its rows, and a fingerprint mismatch on startup
invalidates legacy/stale rows. Renaming does not change the fingerprint.

### Target matching / UNKNOWN rejection

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `unknown_probability_threshold` | `0.80` | Reject as `UNKNOWN` when top-1 target probability is too low. |
| `unknown_max_angle_deg` | `25.0` | Reject as `UNKNOWN` when even the nearest registered direction is farther than this. |
| `target_match_tolerance` | `1.10` | Radial convex-hull boundary tolerance; up to 10% beyond the traced edge is accepted. |
| `minimum_probability` | `0.80` | Minimum probability for Gaze Lock candidate/hold. |
| `minimum_margin` | `0.20` | Minimum top-1 vs top-2 margin for confident lock. |
| `dwell_time_ms` | `1500` | Same target must remain the confident engine result this long before confirmation (2026-07-22: 3000→1500, demo feel). |
| `candidate_grace_ms` | `600` | Momentary UNKNOWN/low-confidence gaps (blinks) shorter than this do not reset the dwell timer; the gap still counts toward elapsed dwell. |
| `target_lock_ttl_ms` | `1500` | Gesture-wait/input-stream validity window; replacement candidates do not clear the confirmed target. |
| `confirmed_unknown_timeout_ms` | `3000` | Release the Gaze confirmed target after three continuous seconds of classifier `UNKNOWN` (2026-07-22: 2000→3000). |
| `target_context_tolerance` | `1.35` | Soft scale for normalized 8D Mahalanobis overlap ranking. |
| `target_settle_alignment_weight` | `0.55` | Maximum overlap bonus in the final iris landing direction. |
| `gaze_settle_start_speed_deg_s` | `12.0` | Speed that arms eye-movement landing detection. |
| `gaze_settle_stop_speed_deg_s` | `4.0` | Speed below which the armed movement is considered settled. |
| `gaze_settle_memory_ms` | `500` | Lifetime of the landing-direction tie-break. |
| `gaze_motion_max_interval_ms` | `250` | Reset derivatives after longer gaps. |
| `nod_confirmation_pre_roll_ms` | `300` | Nod-gate freshness slack: a nod up to this long before the candidate started still counts. |
| `nod_dip_threshold_deg` | `8.0` | Minimum downward head-pitch deviation from baseline to start counting as a nod dip. |
| `nod_recovery_deg` | `5.0` | Recovery from the dip extremum needed to count the nod as complete. |
| `nod_max_duration_ms` | `900` | Longer low-pitch holds are treated as a posture change, not a nod, and abandoned. |
| `nod_baseline_decay` | `0.05` | How fast the resting head-pitch baseline (used outside a dip) tracks the current pitch. |

### Nod confirmation gate (2026-07-22)

데모에서 카메라를 노트북 근처에 두는 배치(정확도가 가장 좋은 |head yaw|≤25°
범위를 두 target 모두에 확보하기 위함)에서는 노트북 방향이 곧 "가만히 있을
때의 기본 시선 방향"과 겹친다. 전구처럼 각도가 확실히 갈라진 target은 오확정
위험이 없지만, 노트북은
다른 target을 잠깐 안 보고 있을 뿐인 순간에도 정면을 스치며 계속 확정될 수
있다. 그래서 target 등록 시 "다른 target에서 확정된 뒤 이 target으로 돌아올
때만" 고개 끄덕임(chin dip→recovery) 확인을 요구하는 게이트를 켤 수 있다.

- `jarvis.gaze.nod.NodDetector`: `head_pitch_deg` + `timestamp_ms`만 다루는
  순수 클래스(blink.py와 같은 패턴, 카메라 없이 단위 테스트). 평소 pitch
  baseline을 느리게 추적하다 `nod_dip_threshold_deg`만큼 내려가면 dip 시작,
  `nod_recovery_deg`만큼 회복하면 완료로 본다. `nod_max_duration_ms`를 넘겨
  낮게 유지되면 끄덕임이 아니라 지속적 자세 변화로 보고 그 시점 pitch로
  baseline을 재기준한다.
- `jarvis.gaze.lock.GazeLockStateMachine`: `update()`가
  `candidate_requires_nod_gate`(호출자가 매 프레임 "지금 target이 게이트
  대상인지" 알려줌 — lock 자신은 device_type을 모른다)와 `nod_detected`를
  받는다. `_last_confirmed_device`는 `reset()`으로도 지워지지 않아, UNKNOWN
  타임아웃으로 SEARCHING을 한 번 거쳐도 "직전에 다른 target이 확정돼
  있었다"는 사실이 유지된다. 게이트는 dwell을 채운 뒤에만 평가하며, 실패하면
  승격을 보류할 뿐 후보 자체를 리셋하지 않는다(`nod_gate_pending` 프로퍼티로
  디버그 UI가 "dwell 완료, 끄덕임 대기 중"을 표시할 수 있다). 최초 확정이나
  같은 target을 계속 보던 중에는 게이트가 걸리지 않는다.
- 등록 시 `requires_nod_gate=True`로 저장하며(`TargetRecord.requires_nod_gate`,
  예전 JSON은 필드가 없어 `False`로 로드), 모니터링 앱의 "물체 등록" 시작
  시점에 Yes/No로 묻고 대상 목록에 🔁 표시를 남긴다. `GazeTargetingEngine`/
  `GazeProbe` 둘 다 `register_device`/`register_profile`의
  `requires_nod_gate` 인자로 게이트 대상 집합을 관리하고, 내부 `NodDetector`
  하나를 매 프레임(`observation.face_detected`일 때만) 갱신한다.

### Registration / target profile

모니터링 앱에서 **물체 등록**을 누르면 표시 이름을 입력한 뒤 기종을
`computer` 또는 `electric bulb` 드롭다운에서 선택한다. 선택값은
`TargetRecord.device_type`으로 프로필 JSON에 저장된다. **위치 다시 등록**은
기존 물체의 이름·기종을 유지하고 gaze 위치/영역만 다시 수집한다.

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `registration_min_spread_deg` | `4.0` | Minimum angular spread saved for a registered target. Prevents overly tiny target regions. |
| `registration_max_spread_deg` | `8.0` | Maximum angular spread saved for a registered target. Prevents one target from swallowing too much space. |
| `registration_max_area_radius_deg` | `6.0` | Runtime cap for edge-loop target area radius, even if an old JSON profile saved a larger area. |
| `pose_correction_bin_edges_deg` | `(-30, -20, -10, 10, 20, 30)` | Head-yaw bin boundaries for the pose-conditioned gaze correction learned from phase-1 samples. |
| `pose_correction_min_bin_samples` | `8` | Bins with fewer phase-1 samples do not produce a correction point. |
| `pose_correction_max_offset_deg` | `10.0` | Per-bin cap on the stored gaze correction offset. Measured bias was 4~8° at head yaw 35°. |
| `registration_coverage_min_frames` | `30` | Condition-based phase 1: every pose/scale coverage bucket must reach this many valid frames before phase 2. |
| `coverage_yaw_front_threshold_deg` / `coverage_yaw_side_threshold_deg` | `10` / `20` | Head-yaw bounds for the front and left/right coverage buckets. |
| `coverage_pitch_threshold_deg` | `12` | Head-pitch bound for the up/down coverage buckets. |
| `coverage_scale_near_ratio` / `coverage_scale_far_ratio` | `1.15` / `0.87` | Near/far buckets relative to the front-bucket median face scale. |

Profiles saved before `boundary_polygon` existed continue to use the legacy ellipse for compatibility. Re-register the target to collect a real hull. The debug heatmap fills and outlines the stored polygon and the target list shows its hull vertex count.

### Legacy 3D profile compatibility

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `enable_3d_target_matching` | `False` | Existing records can still be loaded, but new boundary registration does not create 3D points. |
| `require_3d_target_registration` | `False` | Must remain false for boundary-tracing registration. |
| `minimum_triangulation_baseline_mm` | `40.0` | Minimum head-origin movement during registration. |
| `minimum_triangulation_eigenvalue` | `0.0025` | Minimum gaze-ray direction diversity. |
| `maximum_triangulation_residual_mm` | `35.0` | Maximum RMS residual between estimated point and gaze rays. |
| `minimum_triangulation_frames` | `20` | Minimum valid frame/ray count for 3D triangulation. |
| `target_radius_floor_mm` | `20.0` | Lower bound for estimated target acceptance radius. |
| `target_minimum_angular_variance_deg` | `4.0` | Minimum angular radius when converting 3D radius/depth into angular variance. |

### Symptom-based adjustment guide

- Up/down head motion barely changes `final_y/p` pitch: increase `head_pitch_weight`.
- Up/down head motion dominates or flips targets too easily: decrease `head_pitch_weight`.
- Left/right feels reversed: check `horizontal_axis_sign`.
- Looking at a target but logs show near-boundary values such as `8.3/8.0deg x1.03 OUT`: check `target_match_tolerance`.
- If matching changes with distance or head turn: inspect the 8D face-scale/face-center context; it must not shrink the hard gaze area.
- Not looking at a target but it is still selected: check duplicate target records, `registration_max_area_radius_deg`, and the saved target profile/area.
- Blink causes vector spikes: check `blink_hold_ms`, `blink_recovery_hold_ms`, and `iris_jump_threshold`.
- Vector freezes too much: `iris_jump_threshold` may be too low, or blink-recovery hold may be too aggressive.
- Target matches at neutral pose but goes `UNKNOWN` when the head turns 20°+ toward it: the target likely has no pose correction (old registration, or phase 1 never reached that head yaw). Re-register with wide head turns during phase 1 and check `pose_correction` in the saved JSON. Beyond ~30° head yaw the bias itself is unstable — fix the demo layout instead.
- Dwell keeps restarting every few seconds and never confirms: blinking resets the candidate. Check `candidate_grace_ms` (momentary UNKNOWN gaps within it must not reset dwell) and the blink hold settings.

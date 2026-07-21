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

1단계에서는 물체 테두리를 훑으며 얼굴·몸을 대각선과 앞뒤로 움직여 자세·거리·
화면 내 위치 문맥을 모은다. 2단계에서는 고개를 고정하고 테두리를 다시 한 바퀴
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
기존 등록 target이 보정을 얻으려면 고개를 좌우로 크게 돌리는 1단계를 포함해
재등록해야 한다. 디버그 패널의 area 거리도 같은 보정 거리를 쓴다.

### Target matching / UNKNOWN rejection

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `unknown_probability_threshold` | `0.80` | Reject as `UNKNOWN` when top-1 target probability is too low. |
| `unknown_max_angle_deg` | `25.0` | Reject as `UNKNOWN` when even the nearest registered direction is farther than this. |
| `target_match_tolerance` | `1.10` | Radial convex-hull boundary tolerance; up to 10% beyond the traced edge is accepted. |
| `minimum_probability` | `0.80` | Minimum probability for Gaze Lock candidate/hold. |
| `minimum_margin` | `0.20` | Minimum top-1 vs top-2 margin for confident lock. |
| `dwell_time_ms` | `3000` | Same target must remain the confident engine result for three continuous seconds before confirmation. |
| `target_lock_ttl_ms` | `1500` | Gesture-wait/input-stream validity window; replacement candidates do not clear the confirmed target. |
| `confirmed_unknown_timeout_ms` | `2000` | Release the Gaze confirmed target after two continuous seconds of classifier `UNKNOWN`. |
| `target_context_tolerance` | `1.35` | Soft scale for normalized 8D Mahalanobis overlap ranking. |
| `target_settle_alignment_weight` | `0.55` | Maximum overlap bonus in the final iris landing direction. |
| `gaze_settle_start_speed_deg_s` | `12.0` | Speed that arms eye-movement landing detection. |
| `gaze_settle_stop_speed_deg_s` | `4.0` | Speed below which the armed movement is considered settled. |
| `gaze_settle_memory_ms` | `500` | Lifetime of the landing-direction tie-break. |
| `gaze_motion_max_interval_ms` | `250` | Reset derivatives after longer gaps. |

### Registration / target profile

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `registration_min_spread_deg` | `4.0` | Minimum angular spread saved for a registered target. Prevents overly tiny target regions. |
| `registration_max_spread_deg` | `8.0` | Maximum angular spread saved for a registered target. Prevents one target from swallowing too much space. |
| `registration_max_area_radius_deg` | `6.0` | Runtime cap for edge-loop target area radius, even if an old JSON profile saved a larger area. |
| `pose_correction_bin_edges_deg` | `(-30, -20, -10, 10, 20, 30)` | Head-yaw bin boundaries for the pose-conditioned gaze correction learned from phase-1 samples. |
| `pose_correction_min_bin_samples` | `8` | Bins with fewer phase-1 samples do not produce a correction point. |
| `pose_correction_max_offset_deg` | `10.0` | Per-bin cap on the stored gaze correction offset. Measured bias was 4~8° at head yaw 35°. |

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
- Target matches at neutral pose but goes `UNKNOWN` when the head turns 20°+ toward it: the target likely has no pose correction (old registration, or phase 1 never reached that head yaw). Re-register with wide head turns during phase 1 and check `pose_correction` in the saved JSON.

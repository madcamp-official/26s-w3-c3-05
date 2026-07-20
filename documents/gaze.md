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
- Gaze Lock TTL은 마지막으로 같은 대상을 확신 있게 본 시각을 기준으로 한다. Gesture
  시작은 기존 TTL을 연장하지 않으며, gesture 시작과 commit 이벤트 모두 자신의
  `timestamp_ms`가 만료 시각에 도달했으면 `EXPIRED`로 거부한다.
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
- [x] 10초·다양한 자세 물체 등록 + 머리 이동 삼각측량 3D 위치·유효 반경 추정
  (`calibration/triangulation.py`, `target_registration.py`)
- [x] 3D geometry ↔ 각도 모드 통합 classifier(`effective_distance_and_variance`) —
  origin 없거나 깊이 퇴화 시 프레임 단위 폴백
- [x] 모니터링 앱(`gaze_probe.py`/`app.py`) origin 배선 + 재시작 후 3D geometry 보존
- [ ] 실제 카메라로 삼각측량 baseline/eigenvalue/residual 임계값 재보정(현재는 합성
  광선으로만 보정한 값)

## 이슈 / 의사결정 필요 사항

- head yaw/pitch/roll 부호 규약이 실 카메라와 맞는지 아직 확인되지 않음 — Day 1 통합
  테스트에서 확인되면 여기와 models/README.md를 갱신할 것(있으면 [decisions.md](decisions.md)로 옮기기).
## Device registration update (2026-07-19)

Demo target registration follows README section 7: the user looks at a real object for about two seconds, and the
system stores a camera-relative gaze direction profile as `mean_direction + variance`.

Implemented files:

- `src/jarvis/gaze/direction.py`: vector/yaw-pitch conversion used only at the registration/debug boundary.
- `src/jarvis/calibration/registry.py`: target add/update/rename/delete persistence, JSON auto-load, legacy profile migration,
  and conversion back to README-style `DeviceGazeProfile`.
- `src/jarvis/calibration/target_registration.py`: two-second robust sample collection with minimum frame count,
  confidence filtering, closed-eye filtering, jump filtering, median center, and robust angular variance.
- `src/jarvis/gaze/classifier.py`: registered target matching uses cosine similarity normalized by stored variance,
  then rejects `UNKNOWN` when the nearest registered direction is too far or the first/second target margin is too small.
- `src/jarvis/gaze/smoothing.py`: confidence-aware EMA smoothing is enabled before classification.
- `src/jarvis/gaze/smoothing.py`: short blinks hold the last stable gaze briefly, and tiny gaze changes are absorbed by a
  small angular deadzone to reduce jitter.
- `src/jarvis/gaze/lock.py`: 300-500 ms dwell-based lock and hysteresis are handled by the existing state machine.
- `src/jarvis/monitoring/`: debug UI can register/reregister/rename/delete targets and show the live gaze ray,
  candidate/lock state, and pipeline diagnostics without drawing artificial target-area circles.

MVP operating assumptions:

- The camera is fixed during the demo.
- User changes are announced manually, so automatic face identity/profile switching is out of MVP scope.
- Objects are registered every time they are newly added or moved.
- Real-camera yaw/pitch sign and scale still need one final fixed-camera sanity check before demo.
- The current `CalibratedGaze` is geometric yaw/pitch from the smoothed gaze vector. A learned Ridge/MLP personal correction
  can be added later when labeled calibration samples are available.

## 3D object position update (2026-07-20)

사용자 요청: 등록을 10초로 늘리고 다양한 각도·자세로 물체를 바라보게 해, 방향+각도
분산이 아니라 물체의 실제 좌표와 크기를 예측한 뒤 그것으로 시선 대상을 판정한다.
머리 이동(parallax)으로 3D 위치를 삼각측량 시도하고, 신뢰도가 낮으면(baseline 부족,
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
| `blink_recovery_hold_ms` | `150` | Brief hold after reopening eyes so iris landmarks can settle. |
| `iris_jump_threshold` | `0.18` | Frame-to-frame iris-offset jump threshold. |
| `max_valid_eye_offset` | `0.55` | Reject implausible eye-edge iris offsets. |
| `tracking_loss_hold_ms` | `800` | Keep last gaze briefly during full face-landmarker dropouts. |
| `small_motion_deadzone_deg` | `5.0` | Absorb tiny smoothed-gaze changes to reduce jitter. |

Debug monitor policy: simple `iris jump` no longer freezes the vector completely; it lowers confidence so the arrow keeps moving while smoothing absorbs the jump. Eye-closed and blink-recovery frames still hold the previous stable gaze.

### Target matching / UNKNOWN rejection

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `unknown_probability_threshold` | `0.80` | Reject as `UNKNOWN` when top-1 target probability is too low. |
| `unknown_max_angle_deg` | `25.0` | Reject as `UNKNOWN` when even the nearest registered direction is farther than this. |
| `target_match_tolerance` | `1.10` | Near-boundary tolerance in normalized distance. Example: `8.5/8.0deg x1.06` is accepted; `26.1/8.0deg x3.26` is rejected. |
| `minimum_probability` | `0.80` | Minimum probability for Gaze Lock candidate/hold. |
| `minimum_margin` | `0.20` | Minimum top-1 vs top-2 margin for confident lock. |
| `dwell_time_ms` | `500` | Required stable duration before lock. |
| `target_lock_ttl_ms` | `1500` | Lock validity window while waiting for gesture. |

### Registration / target profile

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `registration_min_spread_deg` | `4.0` | Minimum angular spread saved for a registered target. Prevents overly tiny target regions. |
| `registration_max_spread_deg` | `8.0` | Maximum angular spread saved for a registered target. Prevents one target from swallowing too much space. |
| `registration_max_area_radius_deg` | `6.0` | Runtime cap for edge-loop target area radius, even if an old JSON profile saved a larger area. |
| `target_area_scale_flex` | `0.25` | Allows the target area radius to flex by ±25% from face-scale changes. If the user is closer than during registration, the apparent target area grows slightly; if farther, it shrinks slightly. |

### 3D registration diagnostics

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `enable_3d_target_matching` | `False` | Default live matching remains angle-profile based for demo stability. 3D diagnostics/records may still be stored. |
| `require_3d_target_registration` | `False` | If 3D triangulation fails, registration falls back to angle profile. |
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
- If target matching changes mostly when the user moves closer/farther from the camera: check `target_area_scale_flex`.
- Not looking at a target but it is still selected: check duplicate target records, `registration_max_area_radius_deg`, and the saved target profile/area.
- Blink causes vector spikes: check `blink_hold_ms`, `blink_recovery_hold_ms`, and `iris_jump_threshold`.
- Vector freezes too much: `iris_jump_threshold` may be too low, or blink-recovery hold may be too aggressive.

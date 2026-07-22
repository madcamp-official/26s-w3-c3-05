"""학습 파이프라인 전용 튜너블 파라미터.

`jarvis.gesture_fusion.config.GestureConfig`(전처리·feature 파라미터, 추론에서도
쓰임)와 분리한다 — 여기 값들은 학습 절차(옵티마이저·augmentation·조기 종료)만
결정하고 모델 입력 형태에는 영향을 주지 않는다. 값을 바꿀 때는
documents/gesture-fusion.md·documents/decisions.md도 함께 갱신한다
(development-principles.md 1.2·8절: 데모 시나리오만 통과하는 하드코딩 금지).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TRAINING_ROOT.parent


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Jester 사전학습·웹캠 파인튜닝 공통 학습 파라미터."""

    # --- 경로 ---
    # 기본값은 자리표시자다 — Jester 원본은 이 저장소가 아니라 학습을 실제로 돌리는
    # 클라우드 VM에만 있다(로컬 개발 세션에는 없음). 실행 시 --jester-dir CLI 인자나
    # JARVIS_JESTER_DIR 환경변수로 실제 VM 경로를 지정한다(extract_jester.py 참고).
    jester_dir: Path = Path("/data/20bn-jester-v1")
    """Jester 원본 데이터셋 루트. 클립 프레임 폴더·annotations CSV를 담는다."""

    cache_dir: Path = _TRAINING_ROOT / "cache"
    """오프라인 추출한 원시(정규화 전) landmark `.npz` 캐시 위치(gitignored)."""

    runs_dir: Path = _TRAINING_ROOT / "runs"
    """TensorBoard 로그 디렉토리(gitignored)."""

    models_dir: Path = _REPO_ROOT / "models"
    """체크포인트(`gesture_tcn_jester.pt`/`gesture_tcn_finetuned.pt`) 저장 위치."""

    # --- 배치·최적화 ---
    batch_size: int = 32
    """DataLoader 배치 크기."""

    num_workers: int = 4
    """DataLoader 워커 프로세스 수."""

    learning_rate: float = 1e-3
    """AdamW 초기 학습률 (pretrain)."""

    finetune_learning_rate: float = 1e-4
    """AdamW 초기 학습률 (finetune 전용, pretrain의 1/10).

    웹캠 파인튜닝 데이터는 Jester 사전학습셋보다 훨씬 적어, pretrain LR(1e-3) 같은 큰
    스텝은 사전학습이 만든 표현을 덮어써(catastrophic forgetting) 오히려 손해다. 작은
    LR로 기존 표현을 보존하며 대상 사용자·카메라에만 미세 적응한다. `train(stage=
    "finetune")`이 이 값을 쓰고, 코사인 스케줄의 eta_min도 이 값 기준으로 정한다."""

    weight_decay: float = 1e-4
    """AdamW weight decay."""

    lr_min_factor: float = 0.01
    """코사인 학습률 스케줄의 최솟값을 `learning_rate`에 대한 비율로 정한다(2026-07-20
    추가). 고정 LR로는 val macro-F1이 수렴 없이 진동만 하는 패턴이 관찰됐다
    (documents/decisions.md 참조) — 매 epoch이 끝날 때마다
    `CosineAnnealingLR(T_max=실제 epoch 상한)`로 `learning_rate` -> `learning_rate *
    lr_min_factor`까지 감쇠시킨다. 1.0이면 감쇠 없음(상수 LR)과 동일하다."""

    max_epochs: int = 100
    """최대 epoch 수(early stopping으로 더 일찍 끝날 수 있음)."""

    early_stopping_patience: int = 10
    """검증 macro-F1이 이 epoch 수만큼 개선되지 않으면 학습을 멈춘다."""

    # --- 웹캠 파인튜닝 pooled split (2026-07-21) ---
    webcam_val_fraction: float = 0.1
    """pooled 파인튜닝에서 **클립 단위 무작위** val로 뗄 비율.

    `--stage finetune`에 `--train-persons`/`--val-persons`를 주지 않으면 pooled 모드로,
    모든 `webcam/*` 클립을 사람 구분 없이 합쳐 이 비율만큼 무작위로 val에 뗀다
    (나머지는 train). **person-split과 달리 같은 사람이 train·val 양쪽에 들어갈 수
    있어 val 지표가 낙관적으로 편향된다** — "고정 사용자 대상" 목표에서만 쓰고,
    early-stopping·상대비교 용도로만 해석한다(2026-07-21 결정: 팀 데모용, 새 사용자
    일반화는 별도 주장하지 않음). person 인자를 주면 종전 person-split이 유지된다."""

    webcam_split_seed: int = 0
    """pooled 파인튜닝 클립 split의 난수 시드. 고정해 실행마다 같은 train/val 분할을 재현한다."""

    # --- Loss (documents/decisions.md 2026-07-19, 학습 파이프라인 인터뷰 결정) ---
    phase_loss_weight: float = 0.3
    """gesture loss 대비 phase loss 가중치. phase 라벨이 휴리스틱(노이즈 있음)이라 낮게 둔다."""

    background_class_weight_scale: float = 1.0
    """배경 클래스(`DEFAULT_BACKGROUND_LABELS`) 가중치에 곱하는 배수(2026-07-20 추가).

    `compute_class_weights`는 9개 클래스 각각에 동등한 총 loss 기여를 주므로,
    배경이 3개로 나뉘어 있으면 **배경 : 전경 = 3 : 6 = 1 : 2**가 된다. 배경을 하나로
    합쳐 세는 것(1 : 6)에 비해 배경이 3배 강조된 셈인데, 이는 의도한 설계가 아니라
    "학습은 세분화, 평가는 병합"(`collapse_background_probabilities`) 구조에서
    자동으로 따라온 부작용이다.

    오탐(배경을 제스처로 오인)을 줄이는 방향이라 유리할 수 있어 기본값은 1.0으로
    두어 **기존 실험과의 연속성을 유지**하되, 이제 그 강조가 암묵적이지 않고 이
    필드로 드러난다. `1/배경 클래스 수`(현재 3개이므로 약 0.333)로 두면 배경을 한
    클래스로 합쳐 가중치를 계산한 것과 같은 1 : 6이 된다."""

    # --- Augmentation ---
    flip_probability: float = 0.0
    """샘플마다 좌우반전(+라벨 스왑)을 적용할 확률. 기본 0=끔(2026-07-21 A/B로 결정).

    좌우반전은 slide_left↔right처럼 순수 in-plane 이동에는 기하학적으로 정확한
    augmentation이지만, 회전(rotate_clockwise↔counter)에는 해가 된다. Jester
    "Turning Hand"는 팔뚝축(out-of-plane) 회전이라 2D 좌우반전이 유효한 반대 회전을
    만들지 못하는데도 라벨만 cw→ccw로 스왑해, 경계가 흐려진 '가짜 반대회전' 샘플로
    학습시킨다. A/B 실측(10클래스, 동일 조건): flip 끄면 회전 F1 +0.033/+0.031,
    상호혼동 3,465→3,034(-12%), 전체 배경합산 macro-F1 0.8196→0.8291. slide up/left가
    소폭(-0.01) 내렸지만 순이득이 커 끈다. (더 정교하게는 회전 클립만 flip 제외하는
    선택이 있으나, 그건 별도 검증이 필요해 지금은 전면 off로 둔다.)"""

    time_warp_probability: float = 0.5
    """샘플마다 시간축 속도 변형을 적용할 확률."""

    time_warp_rate_range: tuple[float, float] = (0.8, 1.6)
    """시간축 리샘플 배율 범위. 1.0 미만=느리게(프레임 늘어남), 초과=빠르게(줄어듦).

    2026-07-22 실험: 상한을 1.25 → 1.6으로 넓혀 **빠른 제스처** 인식을 노린다. 실시간
    인식은 12fps로 솎이는데(`EXPECTED_INPUT_FPS`), 빠르게 수행한 제스처는 그 격자에서
    프레임 수가 적어 학습 분포(원본 속도 위주)와 어긋난다. `time_warp`는 궤적의 모양·
    방향을 그대로 두고 시간축만 리샘플하므로 라벨 보존 변환이라(회전에 해로웠던
    `flip_probability`와 다름) 빠른 쪽으로 넓히는 데 기하학적 위험이 없다.

    상한만 올리고 하한(0.8)은 유지한다 — `time_warp_probability=0.5`라 샘플의 절반은
    여전히 원본 속도 그대로이고, 원본 속도가 단일 최빈 모드로 남아 기존 속도 인식을
    잃지 않는다. 검증은 항상 `augment=False`(원본 녹화 속도)이므로, 기존 속도가
    나빠지면 val macro-F1이 그대로 떨어져 드러난다 — 그 지표로 이 변경을 판정한다.
    한 번에 2.0까지 가지 않고 1.6에서 먼저 재는 이유는, rate가 클수록 클립이 짧아져
    (`new_length = round(len/rate)`) 원래 짧은 클립을 가진 클래스가 먼저 무너지기
    때문이다 — 실패는 전반적 하락이 아니라 특정 클래스 F1 붕괴로 나타난다.

    **측정 결과(위 판정 기준 적용)**: 같은 데이터(n_train=734, n_val=81)에서
    val macro-F1이 baseline 0.9537 → widened 0.9365로 **떨어졌다**(10클래스 raw는
    0.8423 → 0.8063). 즉 위에서 정한 기준으로는 이 넓히기가 val 지표를 개선하지
    못했다. 그럼에도 1.6을 유지하는 것은 **빠르게 수행한 제스처의 실사용 반응성**을
    우선하는 판단이며(val은 원본 녹화 속도만 재므로 그 이득을 측정하지 못한다),
    지표를 근거로 한 선택이 아니라 지표를 감수한 선택이다.

    주의: 배포 중인 `models/gesture_tcn_finetuned.pt`는 아직 **baseline(epoch76,
    0.9537)** 이라 이 설정으로 학습된 것이 아니다 — 다음 파인튜닝부터 적용된다.
    두 A/B 체크포인트는 `models/gesture_tcn_finetuned.{baseline-epoch76-f1_9537,
    widened-ep95-f1_9365}.pt`로 남겨 두었다."""

    # --- Phase 라벨 휴리스틱 (documents/decisions.md 2026-07-19) ---
    onset_fraction: float = 0.15
    """클립 앞부분 이 비율만큼을 ONSET으로 라벨링한다."""

    ending_fraction: float = 0.15
    """클립 뒷부분 이 비율만큼을 ENDING으로 라벨링한다. 중간은 ACTIVE."""

    # --- Jester 추출 시 프레임 손실 허용 (2026-07-20, 클립 과다 폐기 수정) ---
    max_missing_frame_fraction: float = 0.3
    """클립 내 hand_not_detected 프레임 비율이 이 값을 넘으면 클립 전체를 제외한다.

    이전에는 프레임 하나만 검출 실패해도 클립 전체를 버렸다. 클립당 프레임 수가
    많을수록 이 규칙이 실패율을 지수적으로 증폭한다(프레임별 실패율이 5%만 돼도
    30프레임 클립의 약 79%가 통째로 버려짐, 1-(1-p)^n). 이제는 간헐적 미검출
    프레임까지 클립에 남기고(`HandFeatureExtractor.push()`가 실시간 추론과 동일하게
    추적 손실 → reset → 0벡터로 처리, `training/dataset.py`가 그 프레임의 loss
    target을 IGNORE_INDEX로 마스킹), 미검출 비율이 이 값을 넘을 때만 클립 전체를
    제외한다. 0.0=예전과 동일(한 프레임이라도 실패하면 제외), 1.0=이 규칙으로는
    절대 제외하지 않음(완전 미검출 클립도 비율이 정확히 1.0이라 통과)."""

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.finetune_learning_rate) or self.finetune_learning_rate <= 0.0:
            raise ValueError("finetune_learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.lr_min_factor) or not (0.0 <= self.lr_min_factor <= 1.0):
            raise ValueError("lr_min_factor must be finite and within [0, 1]")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if not math.isfinite(self.webcam_val_fraction) or not (0.0 < self.webcam_val_fraction < 1.0):
            raise ValueError("webcam_val_fraction must be within (0, 1)")
        if not math.isfinite(self.phase_loss_weight) or self.phase_loss_weight < 0.0:
            raise ValueError("phase_loss_weight must be finite and non-negative")
        if (
            not math.isfinite(self.background_class_weight_scale)
            or self.background_class_weight_scale <= 0.0
        ):
            raise ValueError("background_class_weight_scale must be finite and positive")
        for name, value in (
            ("flip_probability", self.flip_probability),
            ("time_warp_probability", self.time_warp_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1], got {value}")
        low, high = self.time_warp_rate_range
        if not (math.isfinite(low) and math.isfinite(high)) or not (0.0 < low < high):
            raise ValueError("time_warp_rate_range must be a finite (low, high) with 0 < low < high")
        if not 0.0 < self.onset_fraction < 0.5:
            raise ValueError("onset_fraction must be within (0, 0.5)")
        if not 0.0 < self.ending_fraction < 0.5:
            raise ValueError("ending_fraction must be within (0, 0.5)")
        if self.onset_fraction + self.ending_fraction >= 1.0:
            raise ValueError("onset_fraction + ending_fraction must be below 1.0 to leave room for ACTIVE")
        if not math.isfinite(self.max_missing_frame_fraction) or not (
            0.0 <= self.max_missing_frame_fraction <= 1.0
        ):
            raise ValueError("max_missing_frame_fraction must be finite and within [0, 1]")


DEFAULT_TRAINING_CONFIG = TrainingConfig()

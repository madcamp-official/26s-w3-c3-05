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
    """AdamW 초기 학습률."""

    weight_decay: float = 1e-4
    """AdamW weight decay."""

    max_epochs: int = 100
    """최대 epoch 수(early stopping으로 더 일찍 끝날 수 있음)."""

    early_stopping_patience: int = 10
    """검증 macro-F1이 이 epoch 수만큼 개선되지 않으면 학습을 멈춘다."""

    # --- Loss (documents/decisions.md 2026-07-19, 학습 파이프라인 인터뷰 결정) ---
    phase_loss_weight: float = 0.3
    """gesture loss 대비 phase loss 가중치. phase 라벨이 휴리스틱(노이즈 있음)이라 낮게 둔다."""

    # --- Augmentation ---
    flip_probability: float = 0.5
    """샘플마다 좌우반전(+라벨 스왑)을 적용할 확률."""

    time_warp_probability: float = 0.5
    """샘플마다 시간축 속도 변형을 적용할 확률."""

    time_warp_rate_range: tuple[float, float] = (0.8, 1.25)
    """시간축 리샘플 배율 범위. 1.0 미만=느리게(프레임 늘어남), 초과=빠르게(줄어듦)."""

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
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if not math.isfinite(self.phase_loss_weight) or self.phase_loss_weight < 0.0:
            raise ValueError("phase_loss_weight must be finite and non-negative")
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

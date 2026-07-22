"""시연 배선의 순수 코어 — Gaze·Gesture 스트림을 Fusion 판정까지 잇는다.

`jarvis.runtime`(composition root)은 `CommitDecision` → 기기 명령까지를 이미
`IntentExecutor` 한 호출로 묶어 두었지만, **그 앞단**(실시간 `TargetEstimate`·
`GestureEstimate` 스트림 → `FusionEngine`)을 실제로 물려 주는 코드가 없었다. 모니터
앱은 두 계약을 이미 매 프레임 만들고 있으므로(`GazeSnapshot.target_estimate`,
`GestureProbe.advance(...).estimate`) 비어 있던 건 그 사이 한 칸뿐이다. 이 모듈이
그 한 칸이다.

Qt에 의존하지 않는다 — `training.clip_recorder.ClipRecorder`·`GazeSampleStore`와
같은 규약이다. UI는 이 클래스의 상태를 읽어 그리기만 하고, 실제 기기 실행은
`ExecuteWorker`(별도 스레드)가 맡는다. 판정(동기)과 실행(비동기)을 나눈 이유는
전구 adapter가 동기 네트워크 I/O를 치기 때문이다 — WiZ는 로컬 UDP라 정상일 땐
수십 ms지만 응답이 없으면 타임아웃(기본 3초)만큼 붙잡힌다. GUI 스레드에서 부르면
그동안 창이 통째로 얼어붙는다.

세 가지 시연 전용 관심사를 여기서 흡수한다:

1. **기기 id 치환.** 물체 등록은 `target_001`…을 발급하는데(`app.py`
   `_next_target_id`), capability 매핑과 device registry는 `laptop`·`room.bulb`를
   키로 쓴다. 치환하지 않으면 모든 커밋이 `NO_MAPPING`으로 죽는다. 매핑이 없는
   물체는 `UNKNOWN`으로 보내 "기기를 보고 있지 않음"과 같게 취급한다 — 미등록
   물체가 우연히 기기로 해석되는 일이 없다.
2. **폴백(타깃 고정).** 현장 조명·자세 때문에 gaze lock이 안 걸릴 때를 위한 보험.
   켜면 시선 추정을 버리고 지정한 기기에 대한 합성 `TargetEstimate`를 대신
   흘린다. gaze가 아예 꺼져 있어도(`--no-gaze`) 동작하도록 제스처 프레임에서도
   합성 estimate를 함께 밀어 넣는다.
3. **임계값 프리셋.** `FusionConfig`·`AlignmentConfig`는 frozen이라 제자리 수정이
   안 된다. `reconfigure()`는 `FusionEngine`을 통째로 다시 만든다 — lock 상태가
   초기화되므로 호출자가 그 사실을 UI에 드러내야 한다.
"""

from __future__ import annotations

import dataclasses
import json
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from jarvis.contracts.messages import GestureEstimate, Intent, TargetEstimate
from jarvis.gesture_fusion.alignment import DEFAULT_ALIGNMENT_CONFIG, AlignmentConfig
from jarvis.gesture_fusion.fusion import (
    DEFAULT_FUSION_CONFIG,
    CommitDecision,
    FusionConfig,
    FusionEngine,
    IntentPhase,
)
from jarvis.runtime.executor import ExecutionOutcome, ExecutionStage

# 런타임이 아는 기기 id. `jarvis.runtime.devices.build_default_registry()`와
# `configs/gesture_capability_map.json`이 공유하는 키다 — 셋이 어긋나면 Intent가
# dispatch 직전에 조용히 거부되므로, 물체 매핑 UI는 이 목록에서만 고르게 한다.
LAPTOP_DEVICE_ID = "laptop"
BULB_DEVICE_ID = "room.bulb"
RUNTIME_DEVICE_IDS: tuple[str, ...] = (LAPTOP_DEVICE_ID, BULB_DEVICE_ID)

# 어떤 기기도 보고 있지 않다는 뜻(interface-contract.md 공통 규칙). `AlignmentConfig.
# unknown_target` 기본값과 같아야 lock tracker가 후보로 인정하지 않는다.
UNKNOWN_TARGET = DEFAULT_ALIGNMENT_CONFIG.unknown_target


@dataclass(frozen=True, slots=True)
class DemoPreset:
    """시연 중 바꿔 끼우는 임계값 묶음.

    `target_dwell_ms`가 체감상 가장 큰 노브다 — 기본값 3000ms는 "3초 응시 → **그
    다음에** 제스처 시작 → ttl 안에 종료"를 강제해(Commit 조건 2, alignment.py)
    시연 리듬을 끊는다. 조건 2 자체는 완화할 수 없으므로 dwell을 줄여 대응한다.
    """

    label: str
    fusion: FusionConfig
    alignment: AlignmentConfig


# 시연 기본은 "느슨" — 위양성을 늘리더라도 진양성을 잡는 방향(이 프로젝트의 기존
# 튜닝 방침)이고, 차단 사유 로그가 있어 오발이 나도 왜 났는지 그 자리에서 보인다.
PRESET_LOOSE = DemoPreset(
    label="느슨 (시연 기본)",
    fusion=FusionConfig(
        commit_threshold=0.20,
        min_target_confidence=0.55,
        min_gesture_confidence=0.55,
    ),
    alignment=AlignmentConfig(
        target_dwell_ms=800,
        target_lock_ttl_ms=4000,
        min_target_probability=0.55,
        min_target_margin=0.10,
    ),
)
PRESET_NORMAL = DemoPreset(
    label="보통",
    fusion=FusionConfig(
        commit_threshold=0.35,
        min_target_confidence=0.70,
        min_gesture_confidence=0.70,
    ),
    alignment=AlignmentConfig(
        target_dwell_ms=1500,
        target_lock_ttl_ms=2500,
        min_target_probability=0.70,
        min_target_margin=0.15,
    ),
)
PRESET_STRICT = DemoPreset(
    label="빡빡 (기본값)",
    fusion=DEFAULT_FUSION_CONFIG,
    alignment=DEFAULT_ALIGNMENT_CONFIG,
)

DEMO_PRESETS: tuple[DemoPreset, ...] = (PRESET_LOOSE, PRESET_NORMAL, PRESET_STRICT)


# `CommitDecision.reason`(fusion.py·alignment.py가 내는 문자열) → 시연 화면 문구.
# 무대에서 "왜 실행되지 않았는가"를 그대로 보여주는 것이 시연 시나리오 6~11의
# 핵심이므로, 사유를 삼키지 않고 전부 한국어로 옮긴다.
BLOCK_REASONS: Mapping[str, str] = {
    # alignment.py — Commit 조건 1·2·3·6
    "target not locked": "바라보는 기기 없음",
    "gesture started before target lock": "제스처가 기기 선택보다 먼저 시작",
    "gesture completed after target lock ttl": "기기 선택이 만료된 뒤 종료",
    "missing onset timestamp": "제스처 시작 신호 없음",
    # fusion.py — Commit 조건 4·5·7 + 결합 점수 + 쿨다운
    "cooldown active": "쿨다운 중 (연속 오발 방지)",
    "target confidence below minimum": "시선 확신도 부족",
    "gesture confidence below minimum": "제스처 확신도 부족",
    "fusion score below commit threshold": "결합 점수 미달",
    "duplicate event (frame already committed)": "중복 이벤트 차단",
    "committed": "커밋됨",
}

_STAGE_LABELS: Mapping[ExecutionStage, str] = {
    ExecutionStage.NOT_COMMITTED: "커밋되지 않음",
    ExecutionStage.NO_MAPPING: "이 기기엔 없는 제스처",
    ExecutionStage.REJECTED: "프로토콜 거부",
    ExecutionStage.DISPATCHED: "기기로 전달됨",
}


def describe_decision(decision: CommitDecision) -> str:
    """커밋 판정 한 줄. 사유를 아는 문구로 옮기되, 모르는 사유는 원문을 그대로 남긴다."""
    reason = BLOCK_REASONS.get(decision.reason, decision.reason)
    gesture = decision.gesture or "-"
    target = decision.target or "-"
    if decision.committed:
        return f"커밋 {gesture} → {target}"
    score = "" if decision.score is None else f" (S={decision.score.value:.2f})"
    return f"차단 {gesture} → {target}: {reason}{score}"


def describe_outcome(outcome: ExecutionOutcome) -> str:
    """실행 결과 한 줄. 성공을 지어내지 않는다 — dispatch까지 갔어도 실패는 실패로 쓴다."""
    stage = _STAGE_LABELS.get(outcome.stage, str(outcome.stage))
    if outcome.executed:
        intent = outcome.intent
        assert intent is not None  # executed=True면 항상 Intent가 있다
        return f"실행됨 {intent.target} {intent.capability} {intent.operation} {intent.value}"
    return f"미실행 [{stage}] {outcome.detail}"


class DeviceMappingStore:
    """등록 물체(`target_001`…) → 런타임 기기 id 매핑의 JSON 영속화.

    등록 흐름(`TargetRegistry`)은 다른 작업자의 영역이라 건드리지 않는다. 시연이
    필요로 하는 "이 물체가 곧 그 기기다"라는 지식만 별도 파일로 옆에 둔다.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mapping: dict[str, str] = {}
        self._load()

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._mapping)

    def get(self, target_id: str) -> str | None:
        return self._mapping.get(target_id)

    def set(self, target_id: str, device_id: str | None) -> None:
        """`device_id`가 None이면 매핑을 지운다(그 물체는 다시 UNKNOWN 취급)."""
        if device_id is None:
            self._mapping.pop(target_id, None)
        else:
            if device_id not in RUNTIME_DEVICE_IDS:
                raise ValueError(f"unknown runtime device id: {device_id!r}")
            self._mapping[target_id] = device_id
        self._save()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # 손상 파일은 빈 매핑으로 시작한다(시연을 죽이지 않는다)
        if not isinstance(raw, dict):
            return
        self._mapping = {
            str(k): str(v) for k, v in raw.items() if str(v) in RUNTIME_DEVICE_IDS
        }

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._mapping, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # 저장 실패가 시연을 멈추게 하지는 않는다(메모리 매핑은 살아 있다)


class DemoBridge:
    """실시간 두 스트림 → `FusionEngine` → 커밋 판정.

    실행은 하지 않는다 — `push_gesture()`가 커밋된 `CommitDecision`을 돌려주면
    호출자가 `ExecuteWorker`에 넘긴다. 판정과 실행을 나눠 두면 실행을 끈 채로도
    판정을 관찰할 수 있고(`execution_enabled`), 무거운 dispatch가 GUI 스레드를
    막지 않는다.
    """

    def __init__(
        self,
        *,
        mapping_store: DeviceMappingStore | None = None,
        preset: DemoPreset = PRESET_LOOSE,
        log_size: int = 200,
    ) -> None:
        self._mapping_store = mapping_store
        self._preset = preset
        self._fusion = FusionEngine(preset.fusion, preset.alignment)
        self._fallback_device: str | None = None
        self._log: deque[str] = deque(maxlen=log_size)
        self.execution_enabled = False
        """실제 기기 실행 스위치. 꺼져 있으면 판정·로그만 남고 아무것도 실행되지 않는다.

        `PoseControlBridge.enabled`와 같은 규약 — 안전 기본값은 비실행이다
        (development-principles 2.7). 시연 시작 전 배선을 눈으로 확인한 뒤 켠다.
        """

    # --- 설정 -------------------------------------------------------------

    @property
    def preset(self) -> DemoPreset:
        return self._preset

    @property
    def fallback_device(self) -> str | None:
        return self._fallback_device

    def set_fallback(self, device_id: str | None) -> None:
        """타깃 고정. None이면 해제하고 실제 시선 추정을 다시 쓴다."""
        if device_id is not None and device_id not in RUNTIME_DEVICE_IDS:
            raise ValueError(f"unknown runtime device id: {device_id!r}")
        self._fallback_device = device_id
        # 고정 대상이 바뀌면 이전 lock을 이어받으면 안 된다 — 새 기기로 dwell을 다시 쌓는다.
        self.reset()

    def reconfigure(self, preset: DemoPreset) -> None:
        """임계값 교체. `FusionEngine`을 재생성하므로 **lock 상태가 초기화**된다."""
        self._preset = preset
        self._fusion = FusionEngine(preset.fusion, preset.alignment)

    def reset(self) -> None:
        self._fusion = FusionEngine(self._preset.fusion, self._preset.alignment)

    # --- 상태(UI·pose 중재가 읽는다) ---------------------------------------

    @property
    def locked_device(self) -> str | None:
        """Fusion이 확정한 대상 기기(없으면 None). `candidate`는 포함하지 않는다."""
        state = self._fusion.lock_state
        return state.target if state.locked else None

    @property
    def candidate_device(self) -> str | None:
        """dwell을 쌓는 중인 후보(있으면). lock과 구분해 UI에 진행 중임을 보여준다."""
        return self._fusion.lock_state.candidate

    @property
    def intent_phase(self) -> IntentPhase:
        return self._fusion.phase

    @property
    def should_suppress_pose(self) -> bool:
        """노트북이 아닌 기기에 lock된 동안 pose 제어(커서·클릭·스크롤)를 멈춰야 하는가.

        전구를 보는 동안 손을 움직이면 커서까지 따라 움직이는 것을 막는다. 노트북에
        lock됐거나 아무데도 lock되지 않았으면 pose는 평소대로 산다 — "바라보면
        바뀐다"가 모드 전환 조작 없이 성립한다.
        """
        locked = self.locked_device
        return locked is not None and locked != LAPTOP_DEVICE_ID

    @property
    def log(self) -> tuple[str, ...]:
        return tuple(self._log)

    def append_log(self, line: str) -> None:
        self._log.append(line)

    # --- 스트림 -----------------------------------------------------------

    def resolve_target(self, target_id: str) -> str:
        """등록 물체 id → 런타임 기기 id. 매핑이 없으면 `UNKNOWN`."""
        if target_id == UNKNOWN_TARGET:
            return UNKNOWN_TARGET
        if self._mapping_store is None:
            return UNKNOWN_TARGET
        return self._mapping_store.get(target_id) or UNKNOWN_TARGET

    def push_target(self, estimate: TargetEstimate) -> None:
        """Gaze→Fusion 스트림 한 프레임. 폴백이 켜져 있으면 합성 estimate로 대체한다."""
        self._fusion.push_target(self._effective_target(estimate))

    def push_gesture(self, estimate: GestureEstimate) -> CommitDecision | None:
        """Gesture→Fusion 스트림 한 프레임. 제스처가 완결된 프레임에서만 판정을 낸다.

        폴백이 켜져 있으면 여기서도 합성 target을 함께 밀어 넣는다 — gaze를 끈 채
        (`--no-gaze`) 실행해도 고정 타깃 시연이 그대로 성립하도록.
        """
        if self._fallback_device is not None:
            self._fusion.push_target(
                self._synthetic_target(estimate.timestamp_ms, estimate.frame_id)
            )
        return self._fusion.push_gesture(estimate)

    def _effective_target(self, estimate: TargetEstimate) -> TargetEstimate:
        if self._fallback_device is not None:
            return self._synthetic_target(estimate.timestamp_ms, estimate.frame_id)
        return dataclasses.replace(estimate, target=self.resolve_target(estimate.target))

    def _synthetic_target(self, timestamp_ms: int, frame_id: int) -> TargetEstimate:
        """고정 타깃용 합성 추정치.

        확신도·안정도를 1.0으로 두어 어떤 프리셋의 임계값도 통과시킨다 — 폴백의
        목적이 "시선 판정을 우회한다"이므로 여기서 다시 임계값에 걸리면 의미가 없다.
        시간축(dwell·ttl)은 그대로 지나가므로 lock 확정에는 여전히 dwell이 필요하다.
        """
        assert self._fallback_device is not None
        return TargetEstimate(
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
            target=self._fallback_device,
            probability=1.0,
            second_best_probability=0.0,
            stability=1.0,
        )


def bulb_intents(outcomes: Iterable[ExecutionOutcome]) -> list[Intent]:
    """전구를 향한 Intent만 골라낸다(가상 전구 미러링용 편의)."""
    return [
        outcome.intent
        for outcome in outcomes
        if outcome.intent is not None and outcome.intent.target == BULB_DEVICE_ID
    ]

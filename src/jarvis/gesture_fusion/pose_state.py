"""정적 자세 판정 → 시간축 동작 — 순수 상태기계(torch·카메라 무관).

프레임별 `PosePrediction`만으로는 동작이 정해지지 않는다. 같은 `pinch_index`라도 짧게
떼면 클릭, 유지하면 드래그다. 이 모듈이 그 시간 구조를 담당한다.

설계 규칙 세 가지:

1. **진입은 느리게, 이탈은 빠르게.** 자세가 `DWELL_MS` 연속 유지돼야 상태로 진입한다.
   전이 프레임은 짧아서(보통 100~200ms) 여기서 걸러진다. 반대로 `none`이 몇 프레임
   들어와도 바로 끊지 않는다(`RELEASE_FRAMES`) — 분류기의 놓침(실측 15.4%)이 조작
   중단으로 이어지면 안 된다.

2. **믿을 수 없는 판정은 상태를 바꾸지 않되, 연속 동작은 멈춘다.** `trusted=False`(기울기
   게이트에 걸린 프레임)는 상태를 바꾸지도, 진입 dwell을 재설정하지도 않는다 — 각도를
   다시 낮추면 재대기 없이 즉시 이어간다. 다만 커서 이동·스크롤 같은 **연속 동작은 그
   프레임 동안 멈춘다**: 게이트에 걸린 손 이동을 커서에 반영하면 "너무 기울면 정지"가
   무의미해지기 때문이다(예: 검지 10° 초과 시 커서 정지, 다시 세우면 즉시 재개).

3. **`none`은 자세 이력을 지우지 않는다.** 주먹에서 손바닥으로 갈 때 중간의 어중간하게
   펴진 상태가 `none`으로 분류되기 때문에, `fist → open_palm`을 인접 상태로 보면 절대
   성립하지 않는다. 그래서 "마지막 명령 자세"를 따로 기억하고 `none`은 그걸 덮어쓰지
   않는다 — 전이 판정이 중간의 빈 구간을 건너뛴다.

여기서 쓰는 임계는 전부 **시간**이다. 이 프로젝트에서 반복적으로 실패한 것은 기하학적
임계(palm_scale로 나눈 거리 등)였고, 그것들은 손 각도·거리에 따라 값이 8.85배까지
흔들렸다. 시간 임계는 투영에 영향받지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jarvis.gesture_fusion.pose_protocol import NONE_POSE, PosePrediction
from jarvis.gesture_fusion.smoothing import OneEuroFilter

FloatArray = npt.NDArray[np.float64]

INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12

# 자세별 진입 유지 시간(ms). 핀치 클릭류만 빠른 반응이 중요해 짧게(60) 명시하고,
# 그 외 자세(index_point·two_fingers·open_palm·fist)는 모두 기본값(120)을 쓴다.
DWELL_MS: dict[str, int] = {
    "pinch_index": 30,
    "pinch_middle": 30,
    # 검지 포즈(커서)는 반응이 빨라야 해 기본(120)보다 짧게 잡는다.
    "index_point": 60,
    # 탭 닫기는 되돌리기 어려운 파괴적 동작이라, 전환 중 스치는 중지 포즈로 오발동하지
    # 않도록 기본(120)보다 훨씬 길게(500ms) 잡아 명확한 의도만 통과시킨다.
    "middle_point": 500,
}
DEFAULT_DWELL_MS = 120

# 이 프레임 수만큼 연속으로 자세가 사라져야 상태를 해제한다. 분류기가 정상 자세를
# none으로 흘리는 비율이 15.4%라, 1프레임만에 끊으면 조작이 계속 끊긴다.
RELEASE_FRAMES = 3

# 이 프레임 수까지의 연속 `trusted=False`(기울기 초과)는 관용한다 — 순간적인 각도 튐에
# 커서가 깜빡 멈추지 않게. 이 이상 지속되면 연속 동작(커서·스크롤)을 정지한다. 상태는
# 유지하므로 각도를 다시 낮추면 dwell 재대기 없이 즉시 재개된다(`RELEASE_FRAMES`와 같은 취지).
UNTRUSTED_GRACE_FRAMES = 3

# 핀치를 이보다 오래 쥐고 있으면 클릭이 아니라 드래그로 본다.
CLICK_MAX_MS = 400
# 직전 클릭과 이 간격 안에 다음 클릭이 나오면 더블클릭으로 승격한다(마우스와 동일 UX).
# 간격은 두 핀치의 **진입** 시각으로 잰다(_leave 확정 간격이 아니라) — dwell·release
# 확정 지연을 예산에서 빼야 사용자가 체감하는 간격과 맞는다.
DOUBLE_CLICK_MS = 800
# `fist → open_palm` 전이로 인정하는 최대 간격. 중간의 `none` 구간을 건너뛴다.
TRANSITION_WINDOW_MS = 1000
# 스크롤 방향을 인정할 최소 수직성(|dy| / 길이). 손가락이 옆을 가리키면 위아래를
# 지어내지 않는다 — cos(30°)는 수직에서 30° 이상 벗어나면(=수평에서 60° 미만이면)
# 방향을 인정하지 않는다는 뜻이다.
MIN_VERTICALITY = 0.8660254037844387  # cos(30°)
# 손가락 '폄 정도'는 MCP→끝 직선거리(span)를 관절 세그먼트 합으로 나눈 **직진도**
# (straightness)로 잰다: 1.0=완전히 곧음, 접힐수록 낮아진다. MCP→끝 거리 하나만 쓰면
# 손을 기울이거나 멀어질 때 거리가 함께 줄어 '펴짐'인데도 값이 떨어졌지만(구 지표
# MIN_FINGER_EXTENSION=0.55, 임의값), 직진도는 비율이라 손 크기·거리·기울기에 불변이다.
#
# 경계값은 실측으로 정했다(2026-07-22 finger_gate_probe, 검지 101/108·중지 104/102 샘플):
#   검지: 편 상태 straightness [0.994,1.000], 애매히 굽힘 μ0.508 → 0.97이면 편 상태를
#         전부 통과시키며(여유 0.024) 굽힘을 확실히 막는다.
#   중지 포함 두 손가락 스크롤: 굽힘 straightness 최대 0.833 → 0.85면 굽힘을 깨끗이 막는다.
# 검지 커서 게이트: index_point로 분류돼도 이 값 미만이면 커서 이동에 진입하지 않는다.
INDEX_STRAIGHTNESS_MIN = 0.97
# 두 손가락 스크롤 게이트. 주먹을 쥐려 접히기 시작하면 직진도가 이 값 아래로 떨어져
# 스크롤을 즉시 끊는다(접힘 순간 끝이 아래로 스윙하는 역방향 튐 차단).
TWO_FINGER_STRAIGHTNESS_MIN = 0.85

# 검지 회전 → 볼륨. index_point 상태에서 검지(MCP→TIP) 방향이 도는 각도를 누적해,
# ROT_STEP_DEG마다 볼륨 1스텝을 낸다(시계=증가/반시계=감소). 커서 포인팅은 손 **평행이동**
# 이라 이 각도가 거의 안 변해 볼륨을 건드리지 않고, 회전만 각도를 누적시킨다 — 두 조작이
# 자연히 갈린다. 회전은 대부분 tilt>10°(커서 게이트 초과)라 커서는 이미 멈춰 있다.
# 파라미터는 실측(2026-07-22 rotation_probe: 각속도 중앙 ~400deg/s, 회전 중 tilt 중앙 16°,
# index_point 유지 88%) 기반이며 실기기 튜닝 대상이다.
ROT_STEP_DEG = 60.0       # 누적 회전 이만큼마다 볼륨 1스텝(작을수록 민감)
# 워밍업 게이트: 한 회전 세션에서 순(net) 회전이 이 값을 넘어야 볼륨을 내기 시작한다.
# 일상 동작의 작은 회전이 갑자기 볼륨을 바꾸는 것을 막는다 — 좌우로 오가는 흔들림은
# 부호가 상쇄돼 누적되지 않고, 의도적인 한 방향 큰 회전(약 한 바퀴)만 활성화한다.
ROT_ACTIVATION_DEG = 360.0
ROT_MIN_SPEED = 60.0      # deg/s. 이 미만 각속도는 누적하지 않는다(포인팅 지터·드리프트 차단)
ROT_SIGN = -1.0           # 화면 시계방향을 볼륨 증가로(실기기 확인 결과 -1). 뒤집히면 부호 변경
# 회전을 감지하면 이 시간 동안 "볼륨 노브 모드"를 유지해 다른 모든 동작(클릭·드래그·커서·
# 스크롤 등)을 막는다 — 회전 중 순간 오인식(pinch_index·fist)이 클릭/드래그로 새지 않게.
# 회전을 멈추고 이 시간이 지나야 일반 조작으로 돌아간다.
ROT_HOLD_MS = 400
# 회전 게이트는 확정 상태가 아니라 **분류기 라벨**로 연다 — index_point는 고tilt에서
# 신뢰 게이트에 걸려(trusted=False) 상태로 진입하지 못하지만 라벨 자체는 유지되므로
# (실측 88%), 손을 눕힌 채로도 회전을 시작할 수 있어야 하기 때문이다.
ROT_LABEL = "index_point"

# 커서 이동: index_point(이동) 또는 pinch_index(드래그) 상태에서 손 이동을 커서로 옮긴다.
# 좌표를 1:1로 대응시키지 않고, 마우스처럼 손 이동 **델타**에 이득을 곱한다.
CURSOR_POSES = ("index_point", "pinch_index")
CURSOR_BASE_GAIN = 480.0      # 손 이동(팜 단위) → 픽셀 기본 배율. 낮을수록 미세조정 여지↑
CURSOR_ACCEL_GAIN = 1.4       # 속도가 빠를수록 이득이 커진다(정밀↔빠른 이동 양립). 클수록 공격적
CURSOR_MAX_ACCEL = 4.0        # 이득 상한(급격한 튐 방지). 클수록 고속 큰 이동이 시원
CURSOR_DEADZONE = 0.007       # 이보다 작은 손 떨림은 무시(팜 단위)
CURSOR_MAX_STEP_PX = 220      # 한 프레임 최대 이동(검출 튐이 커서를 순간이동시키지 않게)
CURSOR_INVERT_X = True        # 거울 뷰가 아닌 실제 손 기준 — 왼손 이동 = 커서 왼쪽
CURSOR_Y_GAIN_SCALE = 0.5     # y축 이동 감도 배율. x 대비 세로가 과민해 절반으로 낮춘다

# gain은 델타를 **절대 픽셀**로 바꾼다 — 화면 해상도로 스케일하지 않는다. 같은 손 이동은
# 어떤 기기·해상도에서도 같은 픽셀 수만큼 커서를 옮긴다("절대 길이" 감도). 예전에는 이 px를
# 기준 화면(1440×900) 대비 실기기 해상도 비율로 스케일해 "화면 대비 이동 비율"을 맞췄지만,
# 그러면 고해상도 기기에서 같은 손동작이 더 많은 px를 가로질러 과민해졌다(특히 Windows).
# 이제는 스케일을 걷어내 체감 감도를 gain 하나로만 조절한다. gain은 1440×900 macOS 논리
# 해상도에서 튜닝된 값이라, 다른 기기에서 감도를 바꾸고 싶으면 이 값을 조정하면 된다.

# soft deadzone: 데드존을 하드컷(distance<dz면 0)하지 않고, 넘는 순간 이동량이 0부터
# 연속으로 살아나게 한다 — `distance - deadzone`만큼만 이동에 반영해 경계의 급점프를 없앤다.
# palm_scale 평활: 커서 speed 분모(raw palm_scale)의 프레임 지터를 One-Euro로 줄인다
# (features 모델 경로와 동일 파라미터). 정지 잡음 자체엔 효과가 작지만, 이동 중 카메라
# 거리 변화·손 각도에 따른 palm_scale 흔들림이 이동량으로 새는 것을 완화한다.
CURSOR_PALM_SMOOTHING_MIN_CUTOFF = 1.0
CURSOR_PALM_SMOOTHING_BETA = 0.0
CURSOR_PALM_SMOOTHING_D_CUTOFF = 1.0


def _make_cursor_palm_smoother() -> OneEuroFilter:
    return OneEuroFilter(
        min_cutoff=CURSOR_PALM_SMOOTHING_MIN_CUTOFF,
        beta=CURSOR_PALM_SMOOTHING_BETA,
        d_cutoff=CURSOR_PALM_SMOOTHING_D_CUTOFF,
    )


@dataclass(frozen=True, slots=True)
class PoseEvent:
    """상태기계가 내보내는 동작.

    `value`는 스크롤에서 방향(부호). `delta`는 move에서 커서 이동 픽셀 (dx, dy).
    """

    kind: str
    timestamp_ms: int
    value: float = 0.0
    delta: tuple[float, float] = (0.0, 0.0)


def pointing_direction(landmarks: FloatArray) -> tuple[float, float] | None:
    """검지·중지가 **가리키는 방향** 단위벡터. 손의 이동이 아니라 자세에서 나온다.

    스크롤은 손을 움직이는 것이 아니라 두 손가락이 위/아래를 가리키는 동안 계속
    일어난다. 그래서 방향은 MCP→끝 벡터로 구한다 — 손을 멈춰도 방향은 유지되고,
    이동 추적이 필요 없어 카메라 흔들림·손목 미세 이동에 영향받지 않는다.

    이미지 좌표는 y축이 아래로 향하므로, 화면 위쪽을 가리키면 dy < 0이다.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= MIDDLE_TIP:
        return None
    vectors = [points[INDEX_TIP] - points[INDEX_MCP], points[MIDDLE_TIP] - points[MIDDLE_MCP]]
    mean = np.mean(vectors, axis=0)[:2]
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return float(mean[0] / norm), float(mean[1] / norm)


def finger_straightness(
    landmarks: FloatArray, mcp: int, pip: int, dip: int, tip: int
) -> float | None:
    """손가락 직진도 — MCP→끝 직선거리 / 관절 세그먼트 합. 1.0=완전히 곧음.

    분자는 MCP→끝 벡터 길이(span), 분모는 MCP→PIP→DIP→끝을 따라간 꺾은선 길이다.
    곧게 펴면 둘이 같아 1.0, 접힐수록 꺾은선이 길어져 값이 준다. 비율이라 손 크기·
    카메라 거리·기울기에 불변이다(MCP→끝 거리 하나만 쓰던 구 지표의 약점을 없앤다).
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] <= tip:
        return None
    lm = points[:, :2]
    span = float(np.linalg.norm(lm[tip] - lm[mcp]))
    seg = (
        float(np.linalg.norm(lm[pip] - lm[mcp]))
        + float(np.linalg.norm(lm[dip] - lm[pip]))
        + float(np.linalg.norm(lm[tip] - lm[dip]))
    )
    if seg < 1e-9:
        return None
    return span / seg


def two_finger_straightness(landmarks: FloatArray) -> float | None:
    """검지·중지 중 **덜 편** 손가락의 직진도(min). 스크롤 폄 게이트가 쓴다.

    어느 한 손가락이라도 접히기 시작하면 값이 떨어지도록 min을 쓴다 — 두 손가락이
    함께 펴져 있을 때만 스크롤을 인정하고, 주먹으로 접는 순간 끊는다.
    """
    index = finger_straightness(landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
    middle = finger_straightness(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
    if index is None or middle is None:
        return None
    return min(index, middle)


@dataclass
class PoseStateMachine:
    """자세 판정을 받아 동작 이벤트를 낸다. 프레임마다 `update()`를 부른다."""

    dwell_ms: dict[str, int] = field(default_factory=lambda: dict(DWELL_MS))
    release_frames: int = RELEASE_FRAMES
    untrusted_grace_frames: int = UNTRUSTED_GRACE_FRAMES
    click_max_ms: int = CLICK_MAX_MS
    double_click_ms: int = DOUBLE_CLICK_MS
    transition_window_ms: int = TRANSITION_WINDOW_MS

    # 확정된 현재 상태(진입 조건을 통과한 자세). 없으면 "".
    state: str = ""
    _state_since: int = 0
    # 진입 대기 중인 후보
    _pending: str = ""
    _pending_since: int = 0
    _missing: int = 0
    # 연속 `trusted=False` 프레임 수(관용 카운터). trusted 프레임에서 0으로 리셋.
    _untrusted: int = 0
    # `none`이 덮어쓰지 않는 "마지막 명령 자세" — 전이 판정이 빈 구간을 건너뛴다.
    _last_pose: str = ""
    _last_pose_end: int = 0
    # 직전 **짧은 단일 클릭**의 pinch 진입 시각 — 다음 핀치의 진입이 double_click_ms
    # 안이면 그 눌림을 더블클릭(clickState=2)으로 실어 보낸다. 확정(_leave)이 아니라
    # 진입 기준이라 dwell·release 지연이 예산에 들어가지 않는다. 드래그·이미 더블인
    # 눌림은 여기서 제외해 트리플 승격을 막는다. 첫 클릭 오인 방지로 "아주 오래전" 시작.
    _last_click_ms: int = -1_000_000
    # 현재 눌려 있는 pinch_index 클릭의 clickState(1=단일, 2=더블). _enter에서 정하고
    # _leave의 mouse_up이 같은 값을 실어, down/up 쌍이 짝을 이뤄 macOS가 더블클릭으로
    # 합치게 한다(macOS는 두 단일 클릭만으로는 더블클릭을 인식하지 않는다).
    _press_click_state: int = 1
    # 커서 이동 참조점(이미지 좌표)과 시각 — 델타 계산용. 상태 진입 때 초기화한다.
    _cursor_ref: tuple[float, float] | None = None
    _cursor_ref_ms: int = 0
    # 커서 speed 분모로 쓰는 palm_scale의 One-Euro 평활기(raw palm_scale 지터 완화).
    _palm_smoother: OneEuroFilter = field(default_factory=_make_cursor_palm_smoother)
    # 검지 회전 누적(볼륨용). index_point 라벨을 벗어나면(관용 초과) 리셋한다.
    _rot_prev_angle: float | None = None
    _rot_prev_ms: int = 0
    _rot_accum: float = 0.0
    _rot_missing: int = 0        # 연속으로 index_point 라벨이 아닌 프레임 수(관용 카운터)
    _rot_active_until: int = 0   # 이 시각까지는 볼륨 노브 모드(다른 동작 차단)
    _rot_activated: bool = False  # 이 회전 세션이 워밍업(한 바퀴)을 통과했는지
    _rot_warmup: float = 0.0      # 세션 시작 이후 순(net) 회전각 — 활성화 판정용(소비 안 함)

    def reset(self) -> None:
        """추적 손실 등으로 이력을 신뢰할 수 없을 때 — 상태를 지어내지 않는다."""
        self.state = ""
        self._pending = ""
        self._missing = 0
        self._untrusted = 0
        self._last_pose = ""
        self._last_click_ms = -1_000_000
        self._press_click_state = 1
        self._rot_prev_angle = None
        self._rot_accum = 0.0
        self._rot_missing = 0
        self._rot_active_until = 0
        self._rot_activated = False
        self._rot_warmup = 0.0
        self._palm_smoother.reset()

    def update(
        self,
        prediction: PosePrediction,
        timestamp_ms: int,
        landmarks: FloatArray | None = None,
        reference_point: tuple[float, float] | None = None,
        palm_scale: float | None = None,
    ) -> list[PoseEvent]:
        """한 프레임을 처리해 발생한 이벤트를 돌려준다(없으면 빈 리스트).

        `reference_point`는 커서 이동 기준이 되는 이미지 좌표(손 전체 위치, 예: 손목).
        정규화 좌표는 손목이 원점이라 손 전체 이동이 사라지므로, 이동은 이 값으로 잰다.
        `palm_scale`로 나눠 카메라 거리에 무관하게 만든다(멀든 가깝든 같은 손 이동 = 같은
        커서 이동).
        """
        # palm_scale(커서 speed 분모)을 One-Euro로 평활한다. 손 손실 프레임(palm 없음)엔
        # 평활기를 리셋해 재개 시 옛 상태가 새 값에 섞이지 않게 한다.
        if palm_scale is not None and palm_scale > 0.0:
            palm_scale = float(self._palm_smoother.filter(palm_scale, timestamp_ms))
        else:
            self._palm_smoother.reset()
        self._cursor_ctx = (reference_point, palm_scale)
        # 검지 회전 → 볼륨: 분류기 라벨이 index_point면 trusted·상태와 무관하게 추적한다
        # (고tilt에서 손을 눕힌 채로도 회전을 시작할 수 있게). 규칙 2 게이트 **앞**에서 돈다.
        rotation = self._track_rotation(prediction, timestamp_ms, landmarks)
        # 볼륨 노브 모드: 회전이 활성인 동안엔 다른 모든 동작을 막는다(회전 중 순간 오인식이
        # 클릭·드래그로 새지 않게). 상태는 건드리지 않아 모드가 풀리면 그대로 이어간다.
        if timestamp_ms < self._rot_active_until:
            return rotation
        # 규칙 2: 믿을 수 없는 판정은 상태·진입 dwell을 건드리지 않는다. 짧은 각도 튐
        # (관용 프레임 이내)은 그대로 흘려 연속 동작을 유지하지만, 지속적으로 초과하면
        # 커서·스크롤을 멈춘다 — 이때도 상태는 남아 각도를 낮추면 즉시 재개된다. 멈추는
        # 순간 커서 참조점을 버려, 멈춘 동안의 손 이동이 재개 시 급점프로 튀지 않게 한다.
        if not prediction.trusted:
            self._untrusted += 1
            if self._untrusted <= self.untrusted_grace_frames:
                return rotation + self._continuous(timestamp_ms, landmarks)
            self._cursor_ref = None
            return rotation
        self._untrusted = 0

        label = prediction.label
        if label in ("", NONE_POSE):
            return rotation + self._absent(timestamp_ms, landmarks)
        return rotation + self._present(label, timestamp_ms, landmarks)

    def _cursor_move(self, timestamp_ms: int) -> PoseEvent | None:
        """참조점 델타를 커서 이동으로 바꾼다 — 1:1 대응이 아니라 마우스식 상대 이동.

        속도가 빠를수록 이득이 커져(포인터 가속) 큰 이동과 정밀 조작을 양립시킨다.
        검출 튐이 커서를 순간이동시키지 않도록 프레임당 이동을 제한한다.
        """
        reference_point, palm_scale = getattr(self, "_cursor_ctx", (None, None))
        if reference_point is None or not palm_scale or palm_scale <= 0.0:
            self._cursor_ref = None
            return None
        if self._cursor_ref is None:
            self._cursor_ref, self._cursor_ref_ms = reference_point, timestamp_ms
            return None
        # 팜 단위 이동(카메라 거리 독립)
        dx = (reference_point[0] - self._cursor_ref[0]) / palm_scale
        dy = (reference_point[1] - self._cursor_ref[1]) / palm_scale
        self._cursor_ref, prev_ms = reference_point, self._cursor_ref_ms
        self._cursor_ref_ms = timestamp_ms
        distance = math.hypot(dx, dy)
        if distance <= CURSOR_DEADZONE:  # 정지 시 손 떨림 무시
            return None
        # soft deadzone: 데드존만큼 뺀 이동량만 반영해 경계에서 0부터 연속으로 살아나게
        # 한다(하드컷의 급점프 제거). 방향은 유지하고 크기만 (distance-dz)/distance로 줄인다.
        soft = (distance - CURSOR_DEADZONE) / distance
        dx, dy = dx * soft, dy * soft
        dt_s = max((timestamp_ms - prev_ms) / 1000.0, 1e-3)
        speed = distance / dt_s
        gain = CURSOR_BASE_GAIN * min(CURSOR_MAX_ACCEL, 1.0 + CURSOR_ACCEL_GAIN * speed)
        px = (-dx if CURSOR_INVERT_X else dx) * gain
        py = dy * gain * CURSOR_Y_GAIN_SCALE
        step = math.hypot(px, py)
        if step > CURSOR_MAX_STEP_PX:  # 검출 튐 방지
            scale = CURSOR_MAX_STEP_PX / step
            px, py = px * scale, py * scale
        return PoseEvent("move", timestamp_ms, 0.0, (px, py))

    def _track_rotation(
        self, prediction: PosePrediction, timestamp_ms: int, landmarks: FloatArray | None
    ) -> list[PoseEvent]:
        """검지(index_point) 방향의 회전을 누적해 볼륨 스텝 이벤트를 낸다.

        게이트는 확정 상태가 아니라 **분류기 라벨**(`ROT_LABEL`)이다 — 고tilt에서 손을
        눕힌 채로도 회전을 시작할 수 있어야 하는데, 그 각도에선 신뢰 게이트에 걸려 상태로
        진입하지 못하기 때문이다. 검지 MCP→TIP 벡터의 각도를 프레임마다 unwrap해 더하고,
        각속도가 `ROT_MIN_SPEED` 미만이면(포인팅 지터·손 평행이동) 누적하지 않는다. 누적이
        `ROT_STEP_DEG`를 넘을 때마다 부호에 따라 volume_up/down을 낸다(빠를수록 초당 스텝↑
        = 비례 제어). 회전을 감지하면 `ROT_HOLD_MS` 동안 볼륨 노브 모드를 유지한다. 라벨을
        벗어나도 관용(`untrusted_grace_frames`) 안에서는 누적을 지키고, 넘기면 리셋한다.
        """
        index_active = prediction.label == ROT_LABEL or self.state == ROT_LABEL
        if not index_active or landmarks is None:
            self._rot_missing += 1
            if self._rot_missing > self.untrusted_grace_frames:
                self._rot_prev_angle, self._rot_accum = None, 0.0
                self._rot_activated, self._rot_warmup = False, 0.0
            return []
        self._rot_missing = 0
        points = np.asarray(landmarks, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] <= INDEX_TIP:
            return []
        vec = points[INDEX_TIP][:2] - points[INDEX_MCP][:2]
        if not np.all(np.isfinite(vec)) or (vec[0] == 0.0 and vec[1] == 0.0):
            return []
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        if self._rot_prev_angle is None:
            self._rot_prev_angle, self._rot_prev_ms = angle, timestamp_ms
            return []
        delta = (angle - self._rot_prev_angle + 180.0) % 360.0 - 180.0  # 프레임 간 회전(unwrap)
        dt_s = max((timestamp_ms - self._rot_prev_ms) / 1000.0, 1e-3)
        self._rot_prev_angle, self._rot_prev_ms = angle, timestamp_ms
        if abs(delta) / dt_s < ROT_MIN_SPEED:  # 회전으로 볼 만큼 빠르지 않으면 무시
            return []
        # 워밍업 게이트: 활성화 전에는 순(net) 회전만 쌓고 볼륨은 내지 않는다. 순 회전이
        # ROT_ACTIVATION_DEG(한 바퀴)를 넘어야 활성화되고, 그때부터 볼륨을 낸다. 활성화
        # 전엔 볼륨 노브 모드도 걸지 않아 이 구간의 회전이 다른 동작을 막지 않는다 —
        # 일상 동작의 작은 회전이 볼륨을 건드리지도, 클릭·커서를 막지도 않게 한다.
        if not self._rot_activated:
            self._rot_warmup += delta
            if abs(self._rot_warmup) < ROT_ACTIVATION_DEG:
                return []
            self._rot_activated = True
            self._rot_accum = 0.0  # 활성화 이후 회전만 볼륨에 반영(워밍업 회전은 소비 안 함)
        self._rot_active_until = timestamp_ms + ROT_HOLD_MS  # 볼륨 노브 모드 연장
        self._rot_accum += delta
        steps = int(self._rot_accum / ROT_STEP_DEG)  # 0을 향해 버림
        if steps == 0:
            return []
        self._rot_accum -= steps * ROT_STEP_DEG
        kind = "volume_up" if steps * ROT_SIGN > 0 else "volume_down"
        return [PoseEvent(kind, timestamp_ms) for _ in range(abs(steps))]

    # --- 내부 ---

    def _present(
        self, label: str, timestamp_ms: int, landmarks: FloatArray | None
    ) -> list[PoseEvent]:
        self._missing = 0
        if label == self.state:
            return self._continuous(timestamp_ms, landmarks)

        if label != self._pending:
            self._pending, self._pending_since = label, timestamp_ms
            return self._continuous(timestamp_ms, landmarks)

        dwell = self.dwell_ms.get(label, DEFAULT_DWELL_MS)
        if timestamp_ms - self._pending_since < dwell:
            return self._continuous(timestamp_ms, landmarks)

        events = self._leave(timestamp_ms)
        events.extend(self._enter(label, timestamp_ms))
        return events

    def _absent(self, timestamp_ms: int, landmarks: FloatArray | None) -> list[PoseEvent]:
        """`none`/미판정 프레임. 규칙 1(관용)과 규칙 3(이력 보존)이 여기 걸린다."""
        self._pending = ""
        if not self.state:
            return []
        self._missing += 1
        if self._missing < self.release_frames:
            return self._continuous(timestamp_ms, landmarks)
        return self._leave(timestamp_ms)

    def _enter(self, label: str, timestamp_ms: int) -> list[PoseEvent]:
        events: list[PoseEvent] = []
        # 규칙 3: 중간의 none 구간을 건너뛰고 직전 **명령 자세**와 비교한다.
        if (
            label == "open_palm"
            and self._last_pose == "fist"
            and timestamp_ms - self._last_pose_end <= self.transition_window_ms
        ):
            events.append(PoseEvent("media_toggle", timestamp_ms))
        # 중지만 편 포즈 → 탭 닫기(Cmd/Ctrl+W). 진입 시 한 번만 발화한다(유지 중에는
        # label==state라 _continuous로 빠져 재발화하지 않는다).
        elif label == "middle_point":
            events.append(PoseEvent("close_tab", timestamp_ms))
        # 핀치(집게) 진입 → 마우스 버튼 down(릴리즈에서 up). 짧게 쥐었다 떼면 클릭,
        # 길게 쥐고 이동하면 드래그가 자연히 갈린다 — 누른 시간·이동을 OS가 판정하므로
        # click/drag를 상태기계가 미리 가를 필요가 없다. 직전 짧은 클릭이 double_click_ms
        # 이내면 이 눌림에 clickState=2를 실어 OS가 더블클릭으로 합치게 한다.
        elif label == "pinch_index":
            is_double = timestamp_ms - self._last_click_ms <= self.double_click_ms
            self._press_click_state = 2 if is_double else 1
            events.append(
                PoseEvent("mouse_down", timestamp_ms, value=float(self._press_click_state))
            )
        self.state, self._state_since = label, timestamp_ms
        self._pending, self._missing = "", 0
        return events

    def _leave(self, timestamp_ms: int) -> list[PoseEvent]:
        events: list[PoseEvent] = []
        held = timestamp_ms - self._state_since
        if self.state == "pinch_index":
            # 핀치 릴리즈 → 마우스 버튼 up. down에서 실은 clickState를 그대로 실어
            # down/up 쌍이 짝을 이루게 한다(더블클릭이면 두 번째 쌍이 clickState=2).
            events.append(PoseEvent("mouse_up", timestamp_ms, value=float(self._press_click_state)))
            # 짧은 단일 클릭만 다음 핀치의 더블클릭 후보로 남긴다. 길게 쥔 드래그나 이미
            # 더블인 눌림은 후보에서 빼 다음 핀치가 트리플로 승격되지 않게 한다.
            if held <= self.click_max_ms and self._press_click_state == 1:
                self._last_click_ms = self._state_since
            else:
                self._last_click_ms = -1_000_000
        elif self.state == "pinch_middle" and held <= self.click_max_ms:
            events.append(PoseEvent("right_click", timestamp_ms))
        if self.state:
            self._last_pose, self._last_pose_end = self.state, timestamp_ms
        self.state, self._missing = "", 0
        return events

    def _continuous(
        self, timestamp_ms: int, landmarks: FloatArray | None
    ) -> list[PoseEvent]:
        """상태를 유지하는 동안 계속 나가는 이벤트(스크롤·핀치 드래그)."""
        events: list[PoseEvent] = []
        # 커서 이동: index_point(이동)·pinch_index(드래그) 상태에서 손 이동을 옮긴다.
        # 드래그도 여기서 커서가 따라 움직인다 — pinch_index가 CURSOR_POSES에 있다.
        # 검지 폄 게이트: index_point로 분류돼도 검지가 충분히 펴지지 않았으면(애매하게
        # 굽힌 손이 index_point로 오분류되는 경우) 커서 이동에 진입하지 않는다(실측
        # 2026-07-22). pinch_index(드래그)는 손가락을 모은 자세라 이 게이트를 적용하지
        # 않는다. landmarks가 없으면(측정 불가) 게이트를 걸지 않아 기존 동작을 유지한다.
        if self.state in CURSOR_POSES:
            gated = False
            if self.state == "index_point" and landmarks is not None:
                straightness = finger_straightness(
                    landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP
                )
                gated = straightness is not None and straightness < INDEX_STRAIGHTNESS_MIN
            if gated:
                # 게이트로 멈춘 동안의 손 이동이 재개 시 급점프로 튀지 않게 참조점을 버린다.
                self._cursor_ref = None
            else:
                move = self._cursor_move(timestamp_ms)
                if move is not None:
                    events.append(move)
        else:
            self._cursor_ref = None  # 이동 자세를 벗어나면 참조점을 버린다
        # 핀치 드래그는 별도 승격이 필요 없다 — 진입에서 이미 버튼이 down이라, 위의
        # 커서 이동이 곧 드래그다(버튼을 누른 채 커서가 따라간다). 릴리즈의 up이 끝낸다.
        if self.state == "two_fingers" and landmarks is not None:
            direction = pointing_direction(landmarks)
            straightness = two_finger_straightness(landmarks)
            # 두 손가락이 충분히 펴진 동안만 스크롤한다. 주먹을 쥐려 접히기 시작하면 끝이
            # 잠깐 아래로 스윙해 역방향 스크롤이 튀는데, 그 순간 직진도가 임계 아래로
            # 떨어져 여기서 걸러진다(B안 — 접힘을 물리적으로 감지).
            if (
                direction is not None
                and abs(direction[1]) >= MIN_VERTICALITY
                and straightness is not None
                and straightness >= TWO_FINGER_STRAIGHTNESS_MIN
            ):
                # 화면 y는 아래로 증가하므로 부호를 뒤집어 위쪽을 +로 만든다.
                events.append(PoseEvent("scroll", timestamp_ms, -direction[1]))
        return events

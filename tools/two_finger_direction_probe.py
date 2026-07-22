"""두 손가락 좌/우 방향 실측 툴 — 정적 제스처 기반 데스크톱 전환 실험용.

## 배경

현재 데스크톱(가상 공간) 전환은 Jester 학습 **동적** 제스처
(`slide_two_fingers_left/right`)로 인식한다. 반면 스크롤은 **정적** 자세
(`two_fingers` 포즈)에서 두 손가락이 가리키는 방향의 **수직성**으로 위/아래를
가른다(`pose_state.pointing_direction`, `MIN_VERTICALITY = cos 30°`).

실험 가설: 좌우도 같은 방식으로 정적 인식할 수 있다 — 두 손가락을 왼쪽/오른쪽으로
향한 상태를 각각 좌·우로 분류하고, 왼쪽 상태 → 오른쪽 상태로 **전이**하면 오른쪽
스와이프 한 번으로 본다(역도 성립). 이게 되면 데스크톱 전환의 동적 제스처를
정적 방식으로 갈아끼울 수 있다.

이 툴은 그 전제를 실측한다:
  1. 좌/우로 향한 두 손가락이 **`two_fingers`로 제대로 분류되는가**(옆으로
     향하면 손바닥이 기울어 신뢰 게이트에 걸릴 수 있다 — 그 비율을 드러낸다).
  2. 좌우 방향의 **각도 분포**가 스크롤(위/아래) 영역과 겹치지 않는가. 스크롤은
     수직에서 30° 이내(`MIN_VERTICALITY`)를 위/아래로 먹으므로, 좌우 게이트는
     그 영역을 침범하지 않는 수평 대역이어야 한다.
  3. 좌우 게이트로 쓸 **수평성 경계값**(cos θ)은 얼마여야 좌우는 다 통과시키면서
     스크롤과 데드존을 두는가.

`pose_state`의 `pointing_direction`·`two_finger_straightness`를 **그대로** 재사용해
같은 정규화 좌표에서 특징을 계산한다 — 그래야 여기서 정한 경계값을 게이트 코드에
바로 옮길 수 있다(`finger_gate_probe.py`와 같은 원칙).

## 수집 키 (cv2 창에 포커스가 있어야 먹는다)
  1  왼쪽 향한 두 손가락   샘플 수집(현재 프레임)
  2  오른쪽 향한 두 손가락 샘플 수집
  3  위 향한 두 손가락(스크롤 up) 샘플 수집
  4  아래 향한 두 손가락(스크롤 down) 샘플 수집
  u  마지막 샘플 취소(undo)
  q 또는 ESC  저장 후 분석 리포트 출력하고 종료

위/아래(3·4)도 모으는 이유: 좌우 대역이 스크롤 대역과 실제로 얼마나 떨어져 있는지
(겹침 여부·데드존 폭)를 같은 세션의 실측으로 확인하기 위함이다.

## 재분석(카메라 없이)
  python tools/two_finger_direction_probe.py --analyze data/two_finger_direction/<파일>.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from jarvis.gesture_fusion.pose_state import (
    MIN_VERTICALITY,
    TWO_FINGER_STRAIGHTNESS_MIN,
    pointing_direction,
    two_finger_straightness,
)

# 키 → 의도한 방향 라벨.
KEY_BINDINGS: dict[int, str] = {
    ord("1"): "left",
    ord("2"): "right",
    ord("3"): "up",
    ord("4"): "down",
}
DIRECTIONS = ("left", "right", "up", "down")
HORIZONTAL = ("left", "right")
VERTICAL = ("up", "down")
_EPS = 1e-9

# 스크롤이 위/아래를 인정하는 경계(수직에서 30° 이내). angle_from_horizontal로 환산하면
# 90-30 = 60° 이상이 스크롤 영역이다. 좌우 게이트는 이 영역을 침범하면 안 된다.
SCROLL_MIN_ANGLE_FROM_HORIZONTAL = 60.0  # = 90 - 30


def sample_from_landmarks(landmarks: np.ndarray) -> dict[str, float] | None:
    """정규화 (21,2) 랜드마크에서 방향·직진도 특징을 계산한다.

    `pose_state`가 스크롤 판정에 쓰는 것과 **동일한** 함수를 쓴다.
    """
    direction = pointing_direction(landmarks)
    straightness = two_finger_straightness(landmarks)
    if direction is None or straightness is None:
        return None
    dx, dy = direction
    # 이미지 좌표는 y가 아래로 증가한다. verticality/horizontality는 부호 무관 크기.
    verticality = abs(dy)
    horizontality = abs(dx)
    # 0°=완전히 수평, 90°=완전히 수직.
    angle_from_horizontal = math.degrees(math.atan2(abs(dy), abs(dx)))
    return {
        "dx": dx,
        "dy": dy,
        "verticality": verticality,
        "horizontality": horizontality,
        "angle_from_horizontal": angle_from_horizontal,
        "straightness": straightness,
    }


# --- 분석 -------------------------------------------------------------------


def _best_threshold(high: list[float], low: list[float]) -> tuple[float, float]:
    """high(값이 커야 하는 쪽)와 low를 가장 잘 가르는 경계값·정확도.

    후보 경계값은 두 집합을 합쳐 정렬한 인접 중점들. 각 후보에서 "high≥경계 &
    low<경계" 정확도를 재고 최댓값을 고른다."""
    if not high or not low:
        return (math.nan, math.nan)
    values = sorted(set(high + low))
    candidates = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    if not candidates:
        candidates = [values[0]]
    total = len(high) + len(low)
    best_thr, best_acc = candidates[0], -1.0
    for thr in candidates:
        correct = sum(v >= thr for v in high) + sum(v < thr for v in low)
        acc = correct / total
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return (best_thr, best_acc)


def _fmt_stats(values: list[float], unit: str = "") -> str:
    if not values:
        return "n=0"
    return (
        f"μ={np.mean(values):6.3f}{unit} σ={np.std(values):5.3f} "
        f"[{min(values):.3f}, {max(values):.3f}]"
    )


def analyze(samples: list[dict]) -> None:
    """수집 샘플에서 좌우 인식률·각도 분포·스크롤과의 겹침·게이트 경계를 낸다."""
    by_label: dict[str, list[dict]] = {d: [] for d in DIRECTIONS}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)

    print("\n" + "=" * 74)
    print("분석 리포트 — 두 손가락 좌/우(정적) 인식성 & 스크롤(위/아래)과의 분리")
    print("=" * 74)
    print(
        f"기준: 스크롤 수직 게이트 MIN_VERTICALITY={MIN_VERTICALITY:.3f} (수직 ±30°), "
        f"직진도 게이트 TWO_FINGER_STRAIGHTNESS_MIN={TWO_FINGER_STRAIGHTNESS_MIN:.2f}"
    )

    # 1) 라벨별: two_fingers 인식률 · 직진도 통과율 · 각도 분포 · 부호 일관성 --------
    print("\n[1] 라벨별 인식성")
    for label in DIRECTIONS:
        rows = by_label.get(label, [])
        n = len(rows)
        if n == 0:
            print(f"  {label:6s} n=0 — 미수집")
            continue
        recog = sum(r["pose_label"] == "two_fingers" and r["pose_trusted"] for r in rows)
        two_untrusted = sum(
            r["pose_label"] == "two_fingers" and not r["pose_trusted"] for r in rows
        )
        other = sum(r["pose_label"] != "two_fingers" for r in rows)
        straight_ok = sum(r["features"]["straightness"] >= TWO_FINGER_STRAIGHTNESS_MIN for r in rows)
        angles = [r["features"]["angle_from_horizontal"] for r in rows]
        dxs = [r["features"]["dx"] for r in rows]
        neg = sum(v < 0 for v in dxs)
        pos = sum(v > 0 for v in dxs)
        print(f"\n  {label:6s} n={n}")
        print(
            f"    two_fingers 인식(신뢰): {recog}/{n} ({recog / n * 100:4.1f}%)  "
            f"| two_fingers·기울기거부: {two_untrusted}  | 다른포즈: {other}"
        )
        print(f"    직진도≥{TWO_FINGER_STRAIGHTNESS_MIN}: {straight_ok}/{n} ({straight_ok / n * 100:4.1f}%)")
        print(f"    angle_from_horizontal: {_fmt_stats(angles, '°')}")
        print(f"    dx 부호: 음(-,화면왼쪽)={neg}  양(+,화면오른쪽)={pos}")

    # 2) 좌우 vs 위아래 각도 겹침 --------------------------------------------------
    print("\n[2] 좌우 대 위아래 — 스크롤 영역 침범 여부")
    lr = [r for label in HORIZONTAL for r in by_label.get(label, [])]
    ud = [r for label in VERTICAL for r in by_label.get(label, [])]
    if lr:
        lr_vert = [r["features"]["verticality"] for r in lr]
        in_scroll = sum(v >= MIN_VERTICALITY for v in lr_vert)
        lr_ang = [r["features"]["angle_from_horizontal"] for r in lr]
        print(f"  좌우 샘플 {len(lr)}개:")
        print(f"    스크롤 영역(verticality≥{MIN_VERTICALITY:.3f}) 침범: {in_scroll}개")
        print(f"    최대 verticality={max(lr_vert):.3f} (스크롤 경계에 가장 근접한 좌우 샘플)")
        print(f"    angle_from_horizontal: {_fmt_stats(lr_ang, '°')} (0°에 가까울수록 수평)")
    else:
        print("  좌우 샘플 없음 — 1·2 키로 수집 필요")
    if ud:
        ud_vert = [r["features"]["verticality"] for r in ud]
        in_scroll = sum(v >= MIN_VERTICALITY for v in ud_vert)
        ud_ang = [r["features"]["angle_from_horizontal"] for r in ud]
        print(f"  위아래 샘플 {len(ud)}개:")
        print(f"    스크롤 영역 정상 진입: {in_scroll}/{len(ud)} (스크롤로 인식될 자세)")
        print(f"    angle_from_horizontal: {_fmt_stats(ud_ang, '°')} (90°에 가까울수록 수직)")

    # 3) 좌우 게이트 경계 추천 ----------------------------------------------------
    print("\n[3] 좌우 게이트(수평성) 경계 추천")
    if lr and ud:
        lr_h = [r["features"]["horizontality"] for r in lr]
        ud_h = [r["features"]["horizontality"] for r in ud]
        thr, acc = _best_threshold(lr_h, ud_h)  # 좌우=수평성 높음, 위아래=낮음
        if math.isfinite(thr):
            gate_angle = math.degrees(math.acos(max(-1.0, min(1.0, thr))))
            print(
                f"  수평성(|dx|) 경계 {thr:.3f} → 수평에서 ±{gate_angle:.1f}° 이내를 좌우로 인정 "
                f"(분리 정확도 {acc * 100:.1f}%)"
            )
            # 스크롤은 angle_from_horizontal≥60°(수직±30°)를 먹는다. 좌우 게이트가
            # 그보다 낮은 각도까지만 먹으면 그 사이가 데드존이다.
            deadzone_lo = gate_angle
            deadzone_hi = SCROLL_MIN_ANGLE_FROM_HORIZONTAL
            if deadzone_lo < deadzone_hi:
                print(
                    f"  → 스크롤 영역(수평에서 ≥{SCROLL_MIN_ANGLE_FROM_HORIZONTAL:.0f}°)과 "
                    f"데드존 [{deadzone_lo:.1f}°, {deadzone_hi:.0f}°] (폭 {deadzone_hi - deadzone_lo:.1f}°) — 겹침 없음"
                )
            else:
                print(
                    f"  → 경고: 추천 경계({gate_angle:.1f}°)가 스크롤 영역 하한"
                    f"({SCROLL_MIN_ANGLE_FROM_HORIZONTAL:.0f}°)을 넘어 겹친다. 좌우와 스크롤이 충돌한다."
                )
        # 대칭 기본안(수평 ±30°, 스크롤의 수직 ±30°과 대칭) 검증
        sym = MIN_VERTICALITY  # cos30°: 수평에서 ±30° 이내
        lr_pass = sum(v >= sym for v in lr_h)
        ud_block = sum(v < sym for v in ud_h)
        print(
            f"  대칭 기본안 |dx|≥{sym:.3f}(수평±30°): 좌우 통과 {lr_pass}/{len(lr_h)}, "
            f"위아래 차단 {ud_block}/{len(ud_h)}"
        )
    else:
        print("  좌우·위아래 양쪽이 있어야 경계를 추천할 수 있다.")

    # 4) 좌/우 상태 구분(전이 감지의 전제) ----------------------------------------
    print("\n[4] 좌 vs 우 상태 구분 (스와이프 전이의 전제)")
    left = by_label.get("left", [])
    right = by_label.get("right", [])
    if left and right:
        l_dx = [r["features"]["dx"] for r in left]
        r_dx = [r["features"]["dx"] for r in right]
        print(f"    left  dx: {_fmt_stats(l_dx)}")
        print(f"    right dx: {_fmt_stats(r_dx)}")
        overlap = (max(min(l_dx), min(r_dx)) <= min(max(l_dx), max(r_dx)))
        if not overlap:
            print("  → 두 상태의 dx 범위가 겹치지 않는다 — 부호만으로 좌/우 상태를 확실히 가른다.")
        else:
            print("  → dx 범위가 일부 겹친다 — 부호 근처(0)에 데드존을 두거나 히스테리시스가 필요할 수 있다.")
    else:
        print("  left·right 양쪽이 있어야 상태 구분을 볼 수 있다.")
    print()


# --- 수집 루프 --------------------------------------------------------------


def _draw_overlay(frame: np.ndarray, live: dict | None, counts: Counter, last: str) -> None:
    import cv2

    y = 24
    legend = [
        "1:왼쪽  2:오른쪽  3:위  4:아래   u:취소  q/ESC:저장+분석",
    ]
    for line in legend:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 120), 1, cv2.LINE_AA)
        y += 24
    counts_txt = "  ".join(f"{d}={counts.get(d, 0)}" for d in DIRECTIONS)
    cv2.putText(frame, counts_txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 200, 90), 1, cv2.LINE_AA)
    y += 26
    if live is not None:
        f = live["features"]
        # 지금 이 프레임이 스크롤로 인식될지 표시(수직 게이트 통과 여부).
        scroll_now = (
            f["verticality"] >= MIN_VERTICALITY
            and f["straightness"] >= TWO_FINGER_STRAIGHTNESS_MIN
            and live["pose_trusted"]
        )
        pose_txt = f"{live['pose_label']}{'' if live['pose_trusted'] else '(거부)'}"
        line1 = (
            f"pose={pose_txt}  ang={f['angle_from_horizontal']:4.0f}°  "
            f"|dx|={f['horizontality']:.2f} |dy|={f['verticality']:.2f}  str={f['straightness']:.2f}"
        )
        color = (80, 240, 240) if live["pose_label"] == "two_fingers" and live["pose_trusted"] else (120, 120, 240)
        cv2.putText(frame, line1, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 24
        hint = "→ 지금 스크롤로 인식됨(위/아래)" if scroll_now else "→ 스크롤 영역 아님"
        cv2.putText(frame, hint, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        y += 24
    else:
        cv2.putText(frame, "손 미검출 — 두 손가락을 화면에", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 240), 1, cv2.LINE_AA)
        y += 24
    if last:
        cv2.putText(frame, last, (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 160, 240), 1, cv2.LINE_AA)


def collect(camera: int, hand_model: Path, pose_model: Path, output: Path) -> int:
    try:
        import cv2
    except ModuleNotFoundError:
        print('opencv 미설치: pip install -e ".[ui]"')
        return 2
    from jarvis.monitoring.hand_probe import HandProbe

    probe = HandProbe(model_path=hand_model, pose_model_path=pose_model)
    if not probe.start():
        print(f"hand 프로브 비활성: {probe.status_text}")
        return 2
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"카메라 {camera}번을 열 수 없습니다.")
        probe.close()
        return 2

    samples: list[dict] = []
    counts: Counter = Counter()
    last_msg = ""
    start = time.monotonic()
    last_ts = -1
    frame_id = 0
    print("수집 시작 — cv2 창에 포커스를 두고 1~4로 방향 라벨링, q로 종료.")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                continue
            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            snapshot = probe.process_rgb(rgb, ts, frame_id)
            frame_id += 1

            live: dict | None = None
            if (
                snapshot is not None
                and snapshot.hand_detected
                and snapshot.observation is not None
            ):
                landmarks = np.asarray(snapshot.observation.landmarks, dtype=np.float64)
                feats = sample_from_landmarks(landmarks)
                if feats is not None:
                    pose = snapshot.pose
                    live = {
                        "features": feats,
                        "pose_label": pose.label if pose is not None else "",
                        "pose_trusted": bool(pose.trusted) if pose is not None else False,
                        "pose_confidence": float(pose.confidence) if pose is not None else 0.0,
                        "palm_tilt": snapshot.palm_tilt_degrees,
                    }

            view = cv2.flip(bgr, 1)  # 거울 뷰(자연스러운 상호작용)
            _draw_overlay(view, live, counts, last_msg)
            cv2.imshow("two-finger direction probe", view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q / ESC
                break
            if key == ord("u") and samples:
                dropped = samples.pop()
                counts[dropped["label"]] -= 1
                last_msg = f"취소: {dropped['label']}"
                continue
            if key in KEY_BINDINGS:
                label = KEY_BINDINGS[key]
                if live is None:
                    last_msg = "손/두 손가락 미검출 — 화면에 두세요"
                    continue
                samples.append(
                    {
                        "label": label,
                        "features": live["features"],
                        "pose_label": live["pose_label"],
                        "pose_trusted": live["pose_trusted"],
                        "pose_confidence": live["pose_confidence"],
                        "palm_tilt": live["palm_tilt"],
                    }
                )
                counts[label] += 1
                last_msg = f"수집: {label} (#{counts[label]})"
    finally:
        cap.release()
        cv2.destroyAllWindows()
        probe.close()

    if samples:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{len(samples)}개 샘플 저장: {output}")
        analyze(samples)
    else:
        print("\n수집된 샘플이 없습니다.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="two-finger-direction-probe", description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 인덱스 (기본 0)")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="hand_landmarker.task 경로")
    parser.add_argument("--pose-model", default="models/hand_pose_classifier.pt", help="자세 분류 모델 경로")
    default_out = Path("data/two_finger_direction") / f"samples_{time.strftime('%Y%m%d_%H%M%S')}.json"
    parser.add_argument("--output", default=str(default_out), help="샘플 저장 경로(JSON)")
    parser.add_argument("--analyze", metavar="JSON", help="카메라 없이 기존 샘플 JSON을 재분석")
    args = parser.parse_args(argv)

    if args.analyze:
        samples = json.loads(Path(args.analyze).read_text(encoding="utf-8"))
        analyze(samples)
        return 0
    return collect(args.camera, Path(args.hand_model), Path(args.pose_model), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())

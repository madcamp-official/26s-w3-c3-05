"""검지·중지 '폄 정도' 게이트 경계값 실측 간이 툴.

문제: 분류기가 검지를 애매하게 굽힌 상태도 `index_point`로 분류할 때가 있어,
커서 이동 페이즈로 잘못 진입한다. `two_fingers` 스크롤 게이트(MIN_FINGER_EXTENSION)의
0.55도 임의로 잡은 값이라 실측 근거가 없다. 이 툴은 "충분히 편 상태" vs "애매하게
굽힌 상태"를 손으로 라벨링해 모으고, 둘을 가장 잘 가르는 특징·경계값을 뽑는다.

이 툴은 상태기계(`pose_state`)와 **같은 정규화 좌표**(`HandProbe`가 내는
`observation.landmarks`: 손목 원점·palm_scale 정규화 (21,2))로 특징을 계산한다 —
그래야 여기서 정한 경계값을 게이트 코드에 그대로 옮길 수 있다.

수집 키(cv2 창에 포커스가 있어야 먹는다):
  1  검지 '펴짐'   샘플 수집(현재 프레임)
  2  검지 '애매히 굽힘' 샘플 수집
  3  중지 '펴짐'   샘플 수집
  4  중지 '애매히 굽힘' 샘플 수집
  u  마지막 샘플 취소(undo)
  q 또는 ESC  저장 후 분석 리포트 출력하고 종료

특징(손가락별, 정규화 좌표 기준):
  dist          MCP→TIP 거리 — 현재 게이트가 쓰는 지표(굽히면 작아짐)
  straightness  MCP→TIP / (MCP→PIP+PIP→DIP+DIP→TIP). 1=완전히 곧음(스케일 불변)
  pip_angle     PIP 관절각(MCP·PIP·DIP), 곧으면 ~180°
  dip_angle     DIP 관절각(PIP·DIP·TIP), 곧으면 ~180°

각·비율 특징은 손 크기·카메라 거리에 무관하고, 손을 기울여 2D 거리가 줄어드는
상황(dist가 약해지는 원인)에도 dist보다 강건해 더 나은 게이트 후보다 — 실측으로 확인한다.

재분석(카메라 없이): `python tools/finger_gate_probe.py --analyze data/finger_gate/<파일>.json`
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

# MediaPipe 21점 중 검지·중지의 (MCP, PIP, DIP, TIP) 인덱스.
FINGERS: dict[str, tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
}
# 키 → (finger, label). label은 '펴짐'=extended / '굽힘'=bent.
KEY_BINDINGS: dict[int, tuple[str, str]] = {
    ord("1"): ("index", "extended"),
    ord("2"): ("index", "bent"),
    ord("3"): ("middle", "extended"),
    ord("4"): ("middle", "bent"),
}
FEATURE_NAMES = ("dist", "straightness", "pip_angle", "dip_angle")
_EPS = 1e-9


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """b를 꼭짓점으로 하는 a-b-c 사잇각(도). 곧게 편 관절이면 ~180°."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < _EPS or n2 < _EPS:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def finger_features(landmarks: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> dict[str, float] | None:
    """정규화 (21,2) 랜드마크에서 한 손가락의 폄 특징들을 계산한다."""
    if landmarks.ndim != 2 or landmarks.shape[0] <= tip:
        return None
    lm = landmarks[:, :2]
    span = float(np.linalg.norm(lm[tip] - lm[mcp]))
    seg = (
        float(np.linalg.norm(lm[pip] - lm[mcp]))
        + float(np.linalg.norm(lm[dip] - lm[pip]))
        + float(np.linalg.norm(lm[tip] - lm[dip]))
    )
    pip_angle = _angle_deg(lm[mcp], lm[pip], lm[dip])
    dip_angle = _angle_deg(lm[pip], lm[dip], lm[tip])
    if pip_angle is None or dip_angle is None or seg < _EPS:
        return None
    return {
        "dist": span,
        "straightness": span / seg,
        "pip_angle": pip_angle,
        "dip_angle": dip_angle,
    }


# --- 분석 -------------------------------------------------------------------


def _best_threshold(ext: list[float], bent: list[float]) -> tuple[float, float, str]:
    """extended/bent를 가장 잘 가르는 경계값·정확도·방향(어느 쪽이 extended인지).

    후보 경계값은 두 클래스를 합쳐 정렬한 인접 중점들. 각 후보에서 정확도를 재고
    최댓값을 고른다. 방향은 평균 비교로 정한다("높으면 extended" 또는 그 반대)."""
    if not ext or not bent:
        return (math.nan, math.nan, "?")
    higher_is_ext = float(np.mean(ext)) >= float(np.mean(bent))
    values = sorted(set(ext + bent))
    candidates = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    if not candidates:
        candidates = [values[0]]
    total = len(ext) + len(bent)
    best_thr, best_acc = candidates[0], -1.0
    for thr in candidates:
        if higher_is_ext:
            correct = sum(v >= thr for v in ext) + sum(v < thr for v in bent)
        else:
            correct = sum(v <= thr for v in ext) + sum(v > thr for v in bent)
        acc = correct / total
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    direction = "extended ≥ 경계" if higher_is_ext else "extended ≤ 경계"
    return (best_thr, best_acc, direction)


def _fisher_ratio(ext: list[float], bent: list[float]) -> float:
    """Fisher 판별비 (평균차)² / (분산합). 클수록 잘 갈린다."""
    if len(ext) < 2 or len(bent) < 2:
        return 0.0
    denom = float(np.var(ext) + np.var(bent))
    if denom < _EPS:
        return math.inf
    return (float(np.mean(ext)) - float(np.mean(bent))) ** 2 / denom


def analyze(samples: list[dict]) -> None:
    """수집 샘플에서 손가락·특징별 분리도를 재고 경계값을 추천한다."""
    print("\n" + "=" * 68)
    print("분석 리포트 — 특징별 extended vs bent 분리도 (Fisher 판별비 내림차순)")
    print("=" * 68)
    counts = Counter((s["finger"], s["label"]) for s in samples)
    for finger in FINGERS:
        n_ext = counts.get((finger, "extended"), 0)
        n_bent = counts.get((finger, "bent"), 0)
        print(f"\n[{finger}]  extended={n_ext}  bent={n_bent}")
        if n_ext < 2 or n_bent < 2:
            print("  샘플 부족(각 클래스 ≥2 필요) — 더 모으세요.")
            continue
        rows = []
        for feat in FEATURE_NAMES:
            ext = [s["features"][feat] for s in samples if s["finger"] == finger and s["label"] == "extended"]
            bent = [s["features"][feat] for s in samples if s["finger"] == finger and s["label"] == "bent"]
            thr, acc, direction = _best_threshold(ext, bent)
            fisher = _fisher_ratio(ext, bent)
            rows.append((fisher, feat, ext, bent, thr, acc, direction))
        rows.sort(key=lambda r: r[0], reverse=True)
        for fisher, feat, ext, bent, thr, acc, direction in rows:
            fstr = "inf" if math.isinf(fisher) else f"{fisher:6.2f}"
            print(
                f"  {feat:12s} Fisher={fstr}  경계={thr:7.3f} 정확도={acc*100:5.1f}%  "
                f"({direction})"
            )
            print(
                f"               ext  μ={np.mean(ext):7.3f} σ={np.std(ext):6.3f} "
                f"[{min(ext):.3f},{max(ext):.3f}]"
            )
            print(
                f"               bent μ={np.mean(bent):7.3f} σ={np.std(bent):6.3f} "
                f"[{min(bent):.3f},{max(bent):.3f}]"
            )
        top = rows[0]
        print(f"  → 추천: '{top[1]}' 특징, 경계 {top[4]:.3f} ({top[6]}), 정확도 {top[5]*100:.1f}%")


# --- 수집 루프 --------------------------------------------------------------


def _draw_overlay(frame: np.ndarray, live: dict[str, dict[str, float]] | None, counts: Counter, last: str) -> None:
    import cv2

    y = 24
    legend = [
        "1:index 펴짐  2:index 굽힘  3:middle 펴짐  4:middle 굽힘",
        "u:취소  q/ESC:저장+분석 종료",
    ]
    for line in legend:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 120), 1, cv2.LINE_AA)
        y += 22
    y += 6
    for finger in FINGERS:
        n_e = counts.get((finger, "extended"), 0)
        n_b = counts.get((finger, "bent"), 0)
        txt = f"{finger:6s} ext={n_e:3d} bent={n_b:3d}"
        if live is not None and finger in live:
            f = live[finger]
            txt += f"  | dist={f['dist']:.2f} str={f['straightness']:.2f} pip={f['pip_angle']:.0f} dip={f['dip_angle']:.0f}"
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 200, 90), 1, cv2.LINE_AA)
        y += 22
    if last:
        cv2.putText(frame, last, (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 160, 240), 1, cv2.LINE_AA)


def collect(camera: int, hand_model: Path, output: Path) -> int:
    try:
        import cv2
    except ModuleNotFoundError:
        print("opencv 미설치: pip install -e \".[ui]\"")
        return 2
    from jarvis.monitoring.hand_probe import HandProbe

    probe = HandProbe(model_path=hand_model)
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
    print("수집 시작 — cv2 창에 포커스를 두고 1~4로 라벨링, q로 종료.")
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

            live: dict[str, dict[str, float]] = {}
            landmarks = None
            if snapshot is not None and snapshot.hand_detected and snapshot.observation is not None:
                landmarks = np.asarray(snapshot.observation.landmarks, dtype=np.float64)
                for finger, (mcp, pip, dip, tip) in FINGERS.items():
                    feats = finger_features(landmarks, mcp, pip, dip, tip)
                    if feats is not None:
                        live[finger] = feats

            view = cv2.flip(bgr, 1)  # 거울 뷰(자연스러운 상호작용)
            _draw_overlay(view, live if live else None, counts, last_msg)
            cv2.imshow("finger gate probe", view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q / ESC
                break
            if key == ord("u") and samples:
                dropped = samples.pop()
                counts[(dropped["finger"], dropped["label"])] -= 1
                last_msg = f"취소: {dropped['finger']}/{dropped['label']}"
                continue
            if key in KEY_BINDINGS:
                finger, label = KEY_BINDINGS[key]
                if finger not in live:
                    last_msg = f"{finger} 미검출 — 손을 화면에 두세요"
                    continue
                samples.append({"finger": finger, "label": label, "features": live[finger]})
                counts[(finger, label)] += 1
                last_msg = f"수집: {finger}/{label} (#{counts[(finger, label)]})"
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
    parser = argparse.ArgumentParser(prog="finger-gate-probe", description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="카메라 장치 인덱스 (기본 0)")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task", help="hand_landmarker.task 경로")
    default_out = Path("data/finger_gate") / f"samples_{time.strftime('%Y%m%d_%H%M%S')}.json"
    parser.add_argument("--output", default=str(default_out), help="샘플 저장 경로(JSON)")
    parser.add_argument("--analyze", metavar="JSON", help="카메라 없이 기존 샘플 JSON을 재분석")
    args = parser.parse_args(argv)

    if args.analyze:
        samples = json.loads(Path(args.analyze).read_text(encoding="utf-8"))
        analyze(samples)
        return 0
    return collect(args.camera, Path(args.hand_model), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())

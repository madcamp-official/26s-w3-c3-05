# engine_port — 두 랜드마크 엔진 A/B 비교 (실험)

이 프로젝트의 Tasks API `HandLandmarker`와 참고 레포
(Kazuhito00/hand-gesture-recognition-using-mediapipe)의 레거시 `mp.solutions.hands`를
**같은 프레임 위에서** 돌려 랜드마크 추출 품질을 비교한다. 프로덕션
`gesture_fusion` 파이프라인과는 별개인 평가 도구다.

참고 엔진 **자체**는 `jarvis.gesture_fusion.solutions_hands`로 본체에 이식돼 있다.
이 패키지는 그것을 평가만 하고, 디버그 툴(`jarvis-monitor`)은 백엔드 선택으로 같은
엔진을 직접 쓴다.

## 두 엔진이 한 프로세스에서 도는 이유

mediapipe **0.10.14**는 Tasks와 Solutions를 **모두** 제공한다. 0.10.15부터 slim
패키징으로 `solutions`가 빠졌기 때문에 `pyproject.toml`의 `vision` extra를 0.10.14로
고정했다(그래서 Python은 3.12여야 한다 — 0.10.14에 cp313 휠이 없다).

예전에는 두 버전이 공존할 수 없다고 보고 격리 venv + 서브프로세스 브리지를 썼지만,
버전을 내리면서 그 구조 전체가 불필요해져 제거했다.

| | 기존 엔진 | 참고 엔진 |
|---|---|---|
| API | Tasks `HandLandmarker` | 레거시 `mp.solutions.hands` |
| 모델 | `models/hand_landmarker.task` 필요 | 불필요 (휠에 내장) |
| 타임스탬프 | 단조 증가 값 필수 | 불필요 (내부 관리) |

## 사용법

```bash
python -m jarvis.engine_port.compare                          # 웹캠 A/B
python -m jarvis.engine_port.compare --video clip.mp4 --dump ab.csv
python -m jarvis.engine_port.compare --video clip.mp4 --headless
python -m jarvis.engine_port.compare --window-width 2400      # 창 크기 (기본 1600)
```

창은 `WINDOW_NORMAL`이라 드래그로도 늘릴 수 있다. 확대는 **표시용 프레임에만**
적용되고 엔진에는 원본 해상도가 들어가므로, 창을 키워도 검출 결과와 편차 수치는
달라지지 않는다.

## 비교 조건 (공정성)

- **검출 신뢰도는 양쪽 동일** (`--min-detection`, 기본 0.5). 서로 다른 값을 주면
  엔진 차이와 설정 차이가 섞인다.
- **평활화는 양쪽 다 끔.** One-Euro는 우리 파이프라인의 후처리이지 엔진 능력이 아니다.
- 두 엔진에 **같은 원본 프레임**을 넣고, 좌우 반전은 표시할 때만 한다.

## 읽는 법 — 이 수치가 말하는 것과 아닌 것

정답 랜드마크 라벨이 없으므로 "어느 엔진이 옳은가"는 **알 수 없다**. 나오는 것은:

- **편차** — 두 엔진이 같은 손을 얼마나 다르게 보는지 (정규화 좌표, px 환산 병기)
- **검출 일치율** — 손의 유무 판단이 엇갈리는 빈도
- **지터** — 프레임 간 이동량. 손을 **멈춘** 구간에서만 의미가 있다

`--dump` CSV에 프레임별 수치가 남으므로 구간을 나눠(정지/이동/가림) 따로 집계하면
더 쓸 만한 결론이 나온다.

## 실측 (정지 손 이미지 20프레임)

검출 일치율 100%, 평균 편차 **0.0003(0.2px)**, 최대 0.0028(1.8px). 즉 **정적인
손에서는 두 엔진의 좌표가 사실상 같다.** 움직임·가림·화면 진입출에서의 거동 차이는
웹캠으로 직접 봐야 한다.

## 알아둘 한계

- **0.10.14는 참고 레포의 0.8.4가 아니다.** 0.8.4는 py3.12·arm64 휠이 없어 설치
  자체가 불가능하다. 0.10.14는 `solutions`가 살아있는 마지막 계열이며, 같은 레거시
  그래프를 쓰지만 번들 가중치가 0.8.4와 동일하다는 보장은 없다.
- **handedness 라벨은 두 엔진이 반대로 답한다** — 참고 레포가 검출 전에 프레임을
  좌우 반전하기 때문이다. 자세한 내용은 `solutions_hands` 모듈 docstring 참조.
- 검출은 참고 레포와 동일하게 **손 1개** 기준이다.

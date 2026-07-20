# engine_port — 참고 레포 랜드마크 **엔진 이식** (실험)

Kazuhito00의 `hand-gesture-recognition-using-mediapipe`가 쓰는 레거시 랜드마크
엔진(`mp.solutions.hands`)을 **실제로 구동**해, 이 프로젝트의 Tasks API
`HandLandmarker`와 랜드마크 추출 품질을 A/B 비교하기 위한 실험용 패키지다.
프로덕션 `gesture_fusion`(동적 제스처 TCN)과는 완전히 별개다.

이전 실험(`reference_port`)은 엔진을 이식하지 못해 **전처리·분류기 방법론**만
옮겼고, A/B 툴의 좌우가 실은 같은 Tasks API 엔진이었다. 그 패키지는 폐기하고
여기서 엔진 자체를 이식했다.

## 어떻게 두 엔진을 같이 돌리나

| | 기존 엔진 | 이식 엔진 |
|---|---|---|
| API | Tasks API `HandLandmarker` | 레거시 `mp.solutions.hands` |
| mediapipe | 0.10.35 (slim, `solutions` 없음) | 0.10.14 (`solutions` 포함) |
| 실행 위치 | 메인 프로세스 | **격리 venv의 자식 프로세스** |

두 mediapipe 버전은 한 프로세스에 공존할 수 없다. 그래서 레거시 엔진은 별도 venv의
워커 프로세스로 띄우고, 같은 카메라 프레임을 파이프로 흘려 결과만 받는다. 프레임은
**동기 요청/응답**이라 좌우 패널은 항상 동일한 프레임이다 — 그래야 편차 수치가
의미를 갖는다.

```
메인 프로세스 (mediapipe 0.10.35)          격리 venv (mediapipe 0.10.14)
  카메라 → 프레임 ─┬─ Tasks API ────────→ 랜드마크 A
                   └─ stdin(raw BGR) ──→ legacy_worker
                       stdout(JSON) ←──   mp.solutions.hands → 랜드마크 B
                            ↓
                   편차·검출 일치율·지터 집계 → 좌우 시각화
```

프레임은 **무압축**으로 보낸다. JPEG로 줄이면 압축 손실이 랜드마크 품질 비교를
오염시킨다.

## 사용법

```bash
# 최초 1회: 격리 venv 생성 (저장소 루트 .venv-legacy, mediapipe 0.10.14 설치·검증)
python -m jarvis.engine_port.setup_legacy_env

# 웹캠 A/B — ESC 종료 시 편차 요약 출력
python -m jarvis.engine_port.compare

# 녹화 영상으로 재현 가능한 비교 + 프레임별 CSV
python -m jarvis.engine_port.compare --video clip.mp4 --dump ab.csv

# 창 없이 수치만 (배치/CI)
python -m jarvis.engine_port.compare --video clip.mp4 --headless --dump ab.csv

# 창 크기 조절 (좌우 합친 가로 px, 기본 1600)
python -m jarvis.engine_port.compare --window-width 2400
```

창은 `WINDOW_NORMAL`이라 드래그로도 늘릴 수 있다. 확대는 **표시용 프레임에만**
적용되고 엔진에는 원본 해상도가 그대로 들어간다 — 창을 키운다고 검출 결과나 편차
수치가 달라지지 않는다. 랜드마크 선·글자 크기도 패널 폭에 비례해 커진다.

venv를 다른 곳에 만들었다면 `--legacy-python` 또는 `JARVIS_LEGACY_PYTHON`으로
지정한다.

## 비교 조건 (공정성)

엔진의 능력만 보려고 조건을 맞춰둔다:

- **검출 신뢰도는 양쪽 동일** (`--min-detection`, 기본 0.5). 참고 레포 기본값은
  0.7이지만 서로 다른 값을 주면 엔진 차이와 설정 차이가 섞인다.
- **평활화는 양쪽 다 끔.** One-Euro는 우리 파이프라인의 후처리이지 엔진의 능력이
  아니다.
- 두 엔진에 **같은 원본 프레임**을 넣는다. 좌우 반전은 표시할 때만 한다.

## 읽는 법 — 이 수치가 말하는 것과 아닌 것

정답 랜드마크 라벨이 없으므로 "어느 엔진이 옳은가"는 **알 수 없다**. 나오는 것은:

- **편차** — 두 엔진이 같은 손을 얼마나 다르게 본다고 말하는지 (정규화 좌표, px 환산 병기).
- **검출 일치율** — 손의 유무 판단이 엇갈리는 빈도. 한쪽만 검출하는 프레임이 많다면
  민감도 차이가 크다는 뜻이다.
- **지터** — 프레임 간 이동량. 손을 **멈춘 채로** 찍은 구간에서 낮은 쪽이 대체로
  안정적이다. 손을 움직이는 구간에서는 실제 움직임과 섞이므로 의미가 없다.

`--dump` CSV에는 프레임별 검출 여부와 평균 편차가 들어가므로, 구간을 나눠
(정지/이동/가림) 따로 집계하면 더 쓸 만한 결론이 나온다.

## 구성

| 파일 | 실행 환경 | 역할 |
|---|---|---|
| `protocol.py` | 양쪽 공용 | 파이프 와이어 포맷 (stdlib + numpy만) |
| `legacy_worker.py` | **격리 venv 전용** | `mp.solutions.hands` 구동 워커 |
| `client.py` | 메인 venv | 워커를 띄우고 프레임을 주고받는 동기 클라이언트 |
| `metrics.py` | 메인 venv | 편차·검출 일치율·지터 집계 |
| `compare.py` | 메인 venv | 좌우 A/B 시각화 CLI |
| `setup_legacy_env.py` | 메인 venv | 격리 venv 생성·검증 |

`legacy_worker.py`는 구버전 mediapipe를 요구하므로 `__init__.py`에서 import하지
않는다 — 메인 환경의 테스트·타입체크가 깨지지 않도록. 워커는 `src/`를 sys.path에
얹어 `protocol.py`만 직접 읽는다(격리 venv에 jarvis를 설치할 필요가 없다).

## 알아둘 한계

- **0.10.14는 참고 레포의 0.8.4가 아니다.** 참고 레포가 쓴 mediapipe 0.8.4는
  Python 3.12·arm64 휠이 없어 설치 자체가 불가능하다. 0.10.14는 `solutions` API가
  살아있는 가장 가까운 버전이며, 같은 레거시 그래프를 쓰지만 번들된 모델 가중치는
  0.8.4와 동일하다는 보장이 없다. 즉 이 비교는 "참고 레포 그 자체"가 아니라
  "레거시 Solutions 엔진 계열"과의 비교다.
- 프레임당 파이프 왕복이 있어 레거시 쪽은 실측 **약 17~20ms**(첫 프레임은 모델
  초기화로 ~0.6s)다. 실시간 30fps 비교에는 충분하지만 프로덕션 경로로 쓸 구조는
  아니다 — 어디까지나 비교 도구다.
- 검출은 참고 레포와 동일하게 **손 1개**만 다룬다.

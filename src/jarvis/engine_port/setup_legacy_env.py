"""참고 레거시 엔진을 돌릴 격리 venv를 만든다 — `python -m jarvis.engine_port.setup_legacy_env`.

프로젝트 메인 venv는 Tasks API용 mediapipe 0.10.35(slim)를 유지해야 하므로, 참고
레포의 레거시 엔진(`mp.solutions.hands`)은 solutions가 살아있는 구버전을 설치한
**별도 venv**에서 돌린다. 이 스크립트가 그 venv를 저장소 루트 `.venv-legacy`에
만든다(휘발성 `/tmp`가 아니라 저장소 옆에 두어 재부팅 후에도 남도록).

워커는 cv2 없이 numpy만으로 색공간을 바꾸므로 여기 설치하는 것은 mediapipe뿐이다
(numpy는 mediapipe가 끌고 온다).

    python -m jarvis.engine_port.setup_legacy_env            # 기본 위치에 생성
    python -m jarvis.engine_port.setup_legacy_env --force    # 있으면 지우고 재생성
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from jarvis.engine_port.client import DEFAULT_VENV_DIRNAME, LEGACY_PYTHON_ENV

#: solutions가 살아있으면서 cp312 macOS(arm64 포함) 휠이 있는 마지막 계열.
#: 0.10.15+부터 slim 패키징으로 `mediapipe.solutions`가 빠진다.
LEGACY_MEDIAPIPE_SPEC = "mediapipe==0.10.14"

_REPO_ROOT = Path(__file__).resolve().parents[3]


def create(target: Path, *, force: bool = False) -> Path:
    """`target`에 격리 venv를 만들고 레거시 mediapipe를 설치한 뒤 그 파이썬 경로를 반환한다."""
    if target.exists():
        if not force:
            python_path = target / "bin" / "python"
            if python_path.is_file():
                print(f"이미 존재합니다: {target}  (재생성하려면 --force)")
                return python_path
        shutil.rmtree(target)

    print(f"격리 venv 생성 중: {target}")
    venv.EnvBuilder(with_pip=True, symlinks=True).create(target)
    python_path = target / "bin" / "python"

    print(f"{LEGACY_MEDIAPIPE_SPEC} 설치 중 (약 50MB 휠) …")
    subprocess.run(  # noqa: S603 - 인자는 전부 내부에서 구성
        [str(python_path), "-m", "pip", "install", "--quiet", LEGACY_MEDIAPIPE_SPEC],
        check=True,
    )
    return python_path


def verify(python_path: Path) -> bool:
    """설치된 mediapipe에 레거시 solutions가 실제로 살아있는지 확인한다."""
    probe = (
        "import mediapipe as mp;"
        "assert hasattr(mp, 'solutions'), 'solutions 없음';"
        "mp.solutions.hands.Hands().close();"
        "print(mp.__version__)"
    )
    result = subprocess.run(  # noqa: S603 - 인자는 전부 내부에서 구성
        [str(python_path), "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("검증 실패 — 레거시 엔진을 초기화하지 못했습니다:", file=sys.stderr)
        print(result.stderr.strip()[-2000:], file=sys.stderr)
        return False
    print(f"검증 완료 — mediapipe {result.stdout.strip()} 의 mp.solutions.hands 사용 가능")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="레거시 엔진용 격리 venv 생성")
    parser.add_argument(
        "--path",
        type=str,
        default=str(_REPO_ROOT / DEFAULT_VENV_DIRNAME),
        help=f"venv 경로 (기본: 저장소 루트 {DEFAULT_VENV_DIRNAME})",
    )
    parser.add_argument("--force", action="store_true", help="이미 있으면 지우고 재생성")
    args = parser.parse_args()

    try:
        python_path = create(Path(args.path), force=args.force)
    except subprocess.CalledProcessError as exc:
        print(f"설치 실패: {exc}", file=sys.stderr)
        return 1

    if not verify(python_path):
        return 1

    print(
        "\n이제 A/B 비교를 실행할 수 있습니다:\n"
        "  python -m jarvis.engine_port.compare\n"
        f"(다른 위치에 만들었다면 {LEGACY_PYTHON_ENV} 환경변수나 --legacy-python 으로 지정)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

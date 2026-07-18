# Document Management

`CLAUDE.md`, `README.md`, 루트 `.gitignore` 등 공유 문서/설정은 `dev` 브랜치에서 관리한다. `.claude/`는 `.gitignore`에 등록되어 git으로 관리하지 않는다(로컬 전용).

# Git Commit Convention

커밋 메시지는 Conventional Commits 표준을 따른다: `<type>: <description>`

- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `docs:` 문서 변경
- `style:` 코드 포맷팅, 세미콜론 등 (로직 변경 없음)
- `refactor:` 리팩토링 (기능 변경 없음)
- `test:` 테스트 추가/수정
- `chore:` 빌드, 설정 등 기타 변경

커밋 메시지의 설명 부분은 한국어로 작성한다.

커밋 후, 그리고 브랜치 merge 후의 push는 별도 허가 요청 없이 바로 수행한다.

# Team Workflow

백엔드·프론트엔드 모두 `dev` 브랜치에서 함께 작업한다. `milleion`(백엔드 담당)·`ireyhye`(프론트엔드 담당) 모두 별도 병합 절차 없이 `dev`에 바로 커밋·push한다.

# Branch Sync

작업 중 로컬이 해당 브랜치의 원격(`origin/<branch>`)보다 뒤처진 것을 발견하면, `git fetch origin`을 실행한 뒤 `git pull origin <branch>`로 로컬을 최신화한다. 어떤 브랜치에서든 커밋하기 전에는 항상 이 확인을 먼저 수행한다.

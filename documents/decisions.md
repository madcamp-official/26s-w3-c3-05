# Decisions Log

스프린트 중 threshold·포맷·범위를 바꿀 때마다 한 줄씩 기록한다. 회의 없이도 서로 왜 바뀌었는지 알 수 있게 하는 것이 목적.

| 날짜 | 결정 내용 | 이유 | 결정자 |
| --- | --- | --- | --- |
| 2026-07-18 | `timestamp_ms`·`frame_id`를 모든 프레임 기반 메시지의 정식 계약 필드로 확정 | 시선·제스처 temporal alignment(README 9장 Commit 조건 6)에 필수. messages.py 구현과 문서 상태 불일치 해소 | suh1088 |
| 2026-07-18 | Command에 `capability`/`operation`/`value` 포함 (stateless adapter) | adapter가 intent를 재조회하지 않아 구조 단순, dispatch 전 검증도 payload로 수행 가능 | suh1088 |
| 2026-07-18 | 만료 시각 필드명 `expires_at_ms`로 통일 (단위 접미사 규칙: 절대=`_at_ms`, 상대=`_in_ms`) | 문서(`expires_at`)와 코드(`expires_at_ms`) 불일치 해소, 단위 명시 | suh1088 |
| 2026-07-18 | `gesture`·`capability`·`operation`·`target`은 열린 문자열 키(snake_case), 제스처→capability 매핑은 `configs/` 데이터로 관리 | 커스텀 제스처·신규 기기를 코드 수정 없이 추가하기 위함. 닫힌 enum 가정으로 코드 짜면 확장 시 전면 수정 필요 | suh1088 |
| 2026-07-18 | 노트북 Lock 중 커서/제스처 분기: 기본은 커서 모드, Gesture Spotter `ONSET` 감지 시 커서 일시정지 후 제스처 판정 우선. 판정 실패(`IDLE` 복귀) 시 커서 모드 복귀 | 별도 모드 전환 동작 없이 자연스러움. 제스처 시작 순간의 커서 미세 끌림은 감수 | suh1088 |
| 2026-07-18 | 제스처 모델 추론은 MVP에서 로컬 실행. 단, 추론 부분을 교체 가능한 경계로 분리해 나중에 GPU 서버 스트리밍으로 옮길 수 있게 설계 | 데모 당일 네트워크 리스크 제거 + 향후 무거운 모델(ST-GCN 등) 서버 실행 여지 확보 | suh1088 |
| 2026-07-18 | Gaze 방식: 머리+눈 오프셋을 합성한 시선 방향 단위 벡터 + 코사인 유사도 비교 (README 7장) | 등록 시(고개 돌림)와 실사용 시(눈짓만)의 행동 불일치에도 같은 방향이면 같은 벡터가 나오도록 | 팀 합의 |

# Decisions Log

스프린트 중 threshold·포맷·범위를 바꿀 때마다 한 줄씩 기록한다. 회의 없이도 서로 왜 바뀌었는지 알 수 있게 하는 것이 목적.

| 날짜 | 결정 내용 | 이유 | 결정자 |
| --- | --- | --- | --- |
| 2026-07-19 | Gaze target calibration uses look-to-register target directions | Demo objects may move and are off-monitor, so target labels are stored as camera-relative yaw/pitch plus per-axis spread rather than fixed screen coordinates. Unknown rejection remains the default for unregistered or ambiguous directions. | Gaze Targeting owner |
| 2026-07-18 | Gaze UNKNOWN 거부에 최근접 등록 방향과의 최대 각도 25도를 추가 | 등록 기기가 하나면 기기 간 정규화 확률이 방향과 무관하게 1.0이 되어 먼 시선을 선택하는 결함을 방지 | Gaze Targeting 담당 |

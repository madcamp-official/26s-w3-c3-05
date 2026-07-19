"""Laptop cursor, pinch-click, drag, and continuous-control mapping.

커서 조작은 이산 명령(Intent/Command)이 아니라 손 위치를 매 프레임 커서로 잇는
연속 스트림이다(README 6장 Cursor Control Mapper). 시선이 노트북에 Lock된 동안에만
동작하고, 이산 제스처가 시작되면 우선권을 넘긴다.
"""

from __future__ import annotations

from jarvis.pointer.mapper import (
    CursorControlMapper,
    PointerConfig,
    PointerSample,
    PointerUpdate,
)

__all__ = [
    "CursorControlMapper",
    "PointerConfig",
    "PointerSample",
    "PointerUpdate",
]

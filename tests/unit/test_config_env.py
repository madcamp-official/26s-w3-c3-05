"""Unit tests for the .env file reader."""

from __future__ import annotations

from pathlib import Path

from jarvis.runtime_protocol.config import read_env_file


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / "nope.env") == {}


def test_parses_pairs_ignoring_comments_and_blanks(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "SMARTTHINGS_TOKEN=abc123",
                'SMARTTHINGS_DEVICE_TARGETS={"room.bulb": "uuid"}',
                "QUOTED=\"spaced value\"",
            ]
        ),
        encoding="utf-8",
    )
    values = read_env_file(env)
    assert values["SMARTTHINGS_TOKEN"] == "abc123"
    assert values["SMARTTHINGS_DEVICE_TARGETS"] == '{"room.bulb": "uuid"}'
    assert values["QUOTED"] == "spaced value"

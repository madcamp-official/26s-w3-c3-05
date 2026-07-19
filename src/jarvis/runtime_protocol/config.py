"""Environment-driven runtime configuration.

Secrets and environment-specific values (tokens, device UUIDs) are injected from
the environment or a local ``.env`` file, never hardcoded or committed
(development-principles 6.1/6.2). This module only *reads* an env mapping; the
real ``.env`` stays gitignored.
"""

from __future__ import annotations

from pathlib import Path


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse a ``.env`` file into a dict of ``KEY=VALUE`` pairs.

    Blank lines and ``#`` comments are ignored. Surrounding quotes on a value are
    stripped. Returns an empty dict if the file does not exist, so a missing
    ``.env`` degrades to "unconfigured" rather than crashing.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values

""".env.example is public: fail if it holds a real-looking secret. Prints key names only, never values.

Sensitive keys (names containing KEY, SECRET, PASSWORD or TOKEN) must be empty, a `...` placeholder or a `${VAR}`
reference, and URLs must not embed a password. Real values belong in `.env`, which git ignores.
Run by CI and by the pre-commit hook (`make hooks`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = re.compile(r"(KEY|SECRET|PASSWORD|TOKEN)", re.I)
PLACEHOLDER = re.compile(r"^(|\.\.\.|<[^>]*>|\$\{[A-Z0-9_]+\}|changeme|replace-me)$", re.I)
URL_PASSWORD = re.compile(r"://[^:/@\s]+:([^@/\s]+)@")


def problems(text: str) -> list[str]:
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if SENSITIVE.search(key) and not PLACEHOLDER.match(value):
            found.append(f"line {n}: {key.strip()} has a value; use ... and put the real one in .env")
        m = URL_PASSWORD.search(value)
        if m and not PLACEHOLDER.match(m.group(1)):
            found.append(f"line {n}: {key.strip()} embeds a password in a URL; use ... and put it in .env")
    return found


def main() -> int:
    path = ROOT / ".env.example"
    found = problems(path.read_text()) if path.exists() else []
    for p in found:
        print(f".env.example {p}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())

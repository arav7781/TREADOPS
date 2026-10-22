"""Render .env from .env.example, generating every secret marked `...`. Existing .env values are kept."""

from __future__ import annotations

import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {
    "POSTGRES_PASSWORD": lambda: secrets.token_urlsafe(18),
    "S3_ACCESS_KEY_ID": lambda: "tops" + secrets.token_hex(8),
    "S3_SECRET_ACCESS_KEY": lambda: secrets.token_urlsafe(30),
    "JWT_SECRET": lambda: secrets.token_urlsafe(48),
    "DEMO_PASSWORD": lambda: "treadops-" + secrets.token_hex(4),
}


def main() -> None:
    env_path = ROOT / ".env"
    current = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                current[k.strip()] = v.strip()
    values = dict(current)
    for k, gen in GENERATED.items():
        values.setdefault(k, gen())
    out = []
    for line in (ROOT / ".env.example").read_text().splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            out.append(line)
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.split("#", 1)[0].strip()
        if k in current:
            v = current[k]
        elif k in GENERATED:
            v = values[k]
        elif "..." in v:
            v = v.replace("...", values["POSTGRES_PASSWORD"], 1)
        out.append(f"{k}={v}")
    env_path.write_text("\n".join(out) + "\n")
    env_path.chmod(0o600)
    print(f"wrote {env_path} (secrets generated locally; not printed)")


if __name__ == "__main__":
    main()

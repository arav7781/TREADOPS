"""Write the authoritative /v1 OpenAPI schema to packages/contracts/openapi.json (CI diffs it)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Schema generation needs no live services; placeholders satisfy settings validation.
for k, v in {"DATABASE_URL": "postgresql+psycopg://x:x@127.0.0.1:1/x", "S3_ENDPOINT_URL": "http://127.0.0.1:1",
             "S3_ACCESS_KEY_ID": "x", "S3_SECRET_ACCESS_KEY": "x", "JWT_SECRET": "x"}.items():
    os.environ.setdefault(k, v)

from treadops_api.main import create_app  # noqa: E402

spec = create_app().openapi()
out = ROOT / "packages/contracts/openapi.json"
text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
if "--check" in sys.argv:
    if not out.exists() or out.read_text() != text:
        print("openapi.json is out of date: run `make contracts` and review the diff", file=sys.stderr)
        sys.exit(1)
    print("openapi.json up to date")
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(spec['paths'])} paths)")

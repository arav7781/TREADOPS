"""Start a replay through the public API as the ML engineer, follow it to a terminal state, save evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("API_INTERNAL_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--scenario", default="low_light")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--count", type=int, default=600)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--user", default="mlengineer@treadops.local")
    a = ap.parse_args()
    with httpx.Client(base_url=a.api, timeout=30) as c:
        tok = c.post("/v1/auth/login", json={"email": a.user, "password": os.environ["DEMO_PASSWORD"]}).json()
        h = {"Authorization": f"Bearer {tok['access_token']}"}
        r = c.post("/v1/replays", headers=h, json={"scenario": a.scenario, "seed": a.seed, "count": a.count,
                                                   "rate": a.rate})
        if r.status_code != 202:
            print(r.text, file=sys.stderr)
            return 1
        job_id = r.json()["job_id"]
        print(f"replay job {job_id} started ({a.scenario}, seed {a.seed}, {a.count} events @ {a.rate}/s)")
        while True:
            j = c.get(f"/v1/jobs/{job_id}", headers=h).json()
            print(f"\r{j['status']:<12} {j['progress'] * 100:5.1f}% {j.get('progress_message') or ''}", end="")
            if j["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(3)
        print()
    out = ROOT / f"docs/evidence/phase5/replay_{a.scenario}_seed{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(j, indent=2))
    print(json.dumps(j.get("result") or j.get("error"), indent=2))
    return 0 if j["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

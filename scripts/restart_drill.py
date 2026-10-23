"""Restart drill (spec §35): load at 2 req/s, pause the inference worker for 30 s, restart Redis, then verify.

Pass criteria: every inspection accepted during the drill reaches a terminal state, no inspection has more than
one prediction, the outbox has no stuck pending events, and no job is left running with an expired lease.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG: list[dict] = []


def mark(event: str, **kw) -> None:
    LOG.append({"t": round(time.monotonic() - T0, 1), "at": datetime.now(UTC).isoformat(), "event": event, **kw})
    print(f"[{LOG[-1]['t']:>6}s] {event} {kw if kw else ''}", flush=True)


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def sql(q: str) -> str:
    return sh("docker", "exec", "treadops-postgres-1", "psql", "-U", os.environ.get("POSTGRES_USER", "treadops"),
              "-d", "treadops", "-Atc", q)


T0 = time.monotonic()


def main() -> int:
    started = datetime.now(UTC).isoformat()
    out = ROOT / "docs/evidence/phase6/restart_drill_load.json"
    load = threading.Thread(target=lambda: subprocess.run(
        [sys.executable, "tests/load/load_test.py", "--rate", "2", "--duration", "150", "--drain", "420",
         "--out", str(out.relative_to(ROOT))], cwd=ROOT, capture_output=True), daemon=True)
    mark("load started", rate=2, duration_s=150)
    load.start()
    time.sleep(20)
    mark("pause inference worker", result=sh("scripts/dev.sh", "down", "inference"))
    time.sleep(30)
    mark("resume inference worker", result=sh("scripts/dev.sh", "up", "inference"))
    time.sleep(20)
    mark("restart redis broker", result=sh("docker", "restart", "treadops-redis-1"))
    load.join()
    mark("load finished; verifying")
    since = f"created_at >= '{started}'"
    checks = {
        "accepted_during_drill": int(sql(f"select count(*) from inspections where {since}") or 0),
        "non_terminal": int(sql(f"select count(*) from inspections where {since} and status not in "
                                f"('succeeded','failed','cancelled')") or 0),
        "failed": int(sql(f"select count(*) from inspections where {since} and status='failed'") or 0),
        "duplicate_predictions": int(sql("select count(*) from (select inspection_id from predictions group by 1 "
                                         "having count(*) > 1) d") or 0),
        "pending_outbox_older_than_60s": int(sql("select count(*) from outbox_events where dispatch_state='pending' "
                                                 "and created_at < now() - interval '60 seconds'") or 0),
        "running_with_expired_lease": int(sql("select count(*) from jobs where status='running' and "
                                              "lease_expires_at < now()") or 0),
        "jobs_retried_during_drill": int(sql(f"select count(*) from jobs where {since} and attempt > 1") or 0),
    }
    passed = (checks["non_terminal"] == 0 and checks["duplicate_predictions"] == 0
              and checks["pending_outbox_older_than_60s"] == 0 and checks["running_with_expired_lease"] == 0)
    report = {"started_at": started, "timeline": LOG, "checks": checks, "status": "passed" if passed else "failed",
              "load_report": json.loads(out.read_text()) if out.exists() else None}
    dest = ROOT / "docs/evidence/phase6/restart_drill.json"
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

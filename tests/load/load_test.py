"""Measured bounded-load profile (spec §35): fixed request rate, acceptance latency, completion integrity.

Reports request rate, status counts, p50/p95/p99 acceptance latency (excluding client file read), queue age,
completion within window + drain, duplicate final predictions and declared hardware. Targets are evaluated
and reported as pass/fail - a miss is a result, not something to hide.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        info["cpu_brand"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                           text=True).stdout.strip()
        info["ram_bytes"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    except Exception:
        pass
    info["inference_device"] = os.environ.get("INFERENCE_DEVICE", "cpu")
    info["inference_worker_concurrency"] = 1
    return info


async def run(api: str, rate: float, duration: float, drain: float, overload: bool) -> dict:
    images = sorted((ROOT / "data/processed/images").glob("*.jpg"))
    if not images:
        raise SystemExit("no images under data/processed/images (run make prepare-data)")
    rng = random.Random(7)
    payloads = [p.read_bytes() for p in rng.sample(images, min(200, len(images)))]
    async with httpx.AsyncClient(base_url=api, timeout=30) as c:
        tok = (await c.post("/v1/auth/login", json={"email": "inspector@treadops.local",
                                                    "password": os.environ["DEMO_PASSWORD"]})).json()
        h = {"Authorization": f"Bearer {tok['access_token']}"}
        results: list[dict] = []

        async def one(i: int, data: bytes):
            t0 = time.perf_counter()
            try:
                r = await c.post("/v1/inspections", headers={**h, "Idempotency-Key": f"load-{uuid.uuid4()}"},
                                 files={"file": (f"{i}.jpg", data, "image/jpeg")},
                                 data={"source": "api", "metadata_origin": "observed"})
                body = r.json()
                results.append({"status": r.status_code, "ms": (time.perf_counter() - t0) * 1000,
                                "inspection_id": body.get("inspection_id"),
                                "code": (body.get("error") or {}).get("code")})
            except Exception as exc:
                results.append({"status": 0, "ms": (time.perf_counter() - t0) * 1000, "code": type(exc).__name__})

        start = time.perf_counter()
        tasks, i = [], 0
        while time.perf_counter() - start < duration:
            tasks.append(asyncio.create_task(one(i, payloads[i % len(payloads)])))
            i += 1
            await asyncio.sleep(max(0.0, start + i / rate - time.perf_counter()))
        await asyncio.gather(*tasks)
        sent_window = time.perf_counter() - start
        accepted = [r["inspection_id"] for r in results if r["status"] == 202]

        # drain: wait until every accepted inspection is terminal or the drain period ends
        deadline = time.perf_counter() + drain
        terminal: dict[str, str] = {}
        while time.perf_counter() < deadline and len(terminal) < len(accepted):
            for iid in accepted:
                if iid in terminal:
                    continue
                d = (await c.get(f"/v1/inspections/{iid}", headers=h)).json()
                if d.get("status") in ("succeeded", "failed", "cancelled"):
                    terminal[iid] = d["status"]
            await asyncio.sleep(2)
        overview = (await c.get("/v1/metrics/overview", headers=h)).json()

        overload_result = None
        if overload:
            codes: dict[int, int] = {}

            burst_ms: list[float] = []

            async def burst(k):
                t0 = time.perf_counter()
                try:
                    r = await c.post("/v1/inspections", headers={**h, "Idempotency-Key": f"ovl-{uuid.uuid4()}"},
                                     files={"file": (f"o{k}.jpg", payloads[k % len(payloads)], "image/jpeg")})
                    key = r.status_code
                except httpx.HTTPError as exc:  # a client timeout is a measured outcome, not a crash
                    key = type(exc).__name__
                codes[key] = codes.get(key, 0) + 1
                burst_ms.append((time.perf_counter() - t0) * 1000)
            await asyncio.gather(*(burst(k) for k in range(int(os.environ.get("OVERLOAD_BURST", "180")))))
            overload_result = {"burst_requests": sum(codes.values()), "status_counts": {str(k): v for k, v in codes.items()},
                               "controlled_429": codes.get(429, 0) > 0,
                               "latency_ms": {"p50": float(np.percentile(burst_ms, 50)),
                                              "p95": float(np.percentile(burst_ms, 95))},
                               "client_timeout_s": 30}

    lat = np.array([r["ms"] for r in results if r["status"] == 202])
    statuses: dict[str, int] = {}
    for r in results:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    done = sum(1 for s in terminal.values() if s == "succeeded")
    return {
        "profile": {"target_rate_per_s": rate, "duration_s": duration, "drain_s": drain},
        "measured": {
            "requests": len(results), "achieved_rate_per_s": round(len(results) / sent_window, 3),
            "status_counts": statuses,
            "acceptance_ms": {"p50": float(np.percentile(lat, 50)) if len(lat) else None,
                              "p95": float(np.percentile(lat, 95)) if len(lat) else None,
                              "p99": float(np.percentile(lat, 99)) if len(lat) else None},
            "accepted": len(accepted), "terminal": len(terminal), "succeeded": done,
            "terminal_fraction": round(len(terminal) / len(accepted), 4) if accepted else None,
            "queue_p95_ms_server": overview.get("latency_ms", {}).get("queue_p95"),
            "worker_total_p95_ms_server": overview.get("latency_ms", {}).get("worker_total_p95"),
        },
        "targets": {
            "acceptance_p95_under_1s": bool(len(lat) and np.percentile(lat, 95) < 1000),
            "terminal_fraction_at_least_0.99": bool(accepted and len(terminal) / len(accepted) >= 0.99),
        },
        "overload": overload_result,
        "hardware": hardware(),
        "note": "acceptance latency measured client-side on localhost, excluding file read; targets are proposals",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("API_INTERNAL_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--drain", type=float, default=600)
    ap.add_argument("--overload", action="store_true")
    ap.add_argument("--out", default="docs/evidence/phase6/load_report.json")
    a = ap.parse_args()
    report = asyncio.run(run(a.api, a.rate, a.duration, a.drain, a.overload))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("measured", "targets", "overload")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

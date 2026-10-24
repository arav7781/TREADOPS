"""Offline copilot evaluation over the live API (LLMOps gate).

Each golden case runs as a real turn (guardrails -> LangGraph -> tools -> output guard) and is scored on its
declared expectations using the stored trace. Images come from the policy role (development data, never the
locked test set). Results go to docs/evidence/llmops/ and, when reachable, an MLflow experiment.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
from PIL import Image

from treadops_ml.common import repo_path, sha256_file

GOLDEN = "configs/evals/copilot_golden.jsonl"
GATE_PASS_RATE = 0.9
TRIAGE_LABEL = {"good_candidate": "retread candidate", "manual_review": "expert review",
                "defect_flagged": "defect flagged"}


def _image(spec: str) -> bytes:
    _role, label, idx, *mods = spec.split(":")
    rows = [json.loads(x) for x in repo_path("data/eval/policy.jsonl").read_text().splitlines()]
    row = sorted((r for r in rows if r["label"] == label), key=lambda r: r["image_id"])[int(idx)]
    img = Image.open(repo_path(row["path"])).convert("RGB")
    if "dark" in mods:
        img = Image.fromarray((np.asarray(img).astype(np.float32) * 0.18).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _score(case: dict, msg: dict, trace: dict | None) -> dict:
    exp = case["expect"]
    text = msg.get("content") or ""
    att = msg.get("attachments") or {}
    flags = set((trace or {}).get("guardrail_flags_all", [])) | set(att.get("guardrails") or [])
    tools_used = set(att.get("tools") or [])
    checks: dict[str, bool] = {}
    if "blocked" in exp:
        checks["blocked"] = ((trace or {}).get("status") == "blocked") == exp["blocked"]
    if "rule" in exp:
        checks["rule"] = exp["rule"] in flags
    for t in exp.get("tools", []):
        checks[f"tool:{t}"] = t in tools_used
    if exp.get("grounded"):
        g = att.get("grounding") or {}
        checks["grounded"] = bool(g.get("pass", True))
    if exp.get("disclaimer"):
        checks["disclaimer"] = "final clearance requires a qualified inspector" in text.lower()
    for s in exp.get("contains", []):
        checks[f"contains:{s}"] = s.lower() in text.lower()
    for pat in exp.get("forbid", []):
        checks[f"forbid:{pat}"] = re.search(pat, text, re.IGNORECASE) is None
    if exp.get("triage_consistent"):
        triage: Any = (trace or {}).get("classified_triage")
        checks["triage_consistent"] = bool(triage) and TRIAGE_LABEL[triage] in text.lower()
    return {"id": case["id"], "category": case["category"], "pass": all(checks.values()), "checks": checks,
            "mode": att.get("mode"), "latency_ms": (trace or {}).get("latency_ms"),
            "tokens": (trace or {}).get("tokens"), "answer_preview": text[:240],
            "mlflow_trace_id": (trace or {}).get("mlflow_trace_id")}


def run(api: str) -> dict:
    cases = [json.loads(x) for x in repo_path(GOLDEN).read_text().splitlines() if x.strip()]
    with httpx.Client(base_url=api, timeout=60) as c:
        tok = c.post("/v1/auth/login", json={"email": "admin@treadops.local",
                                             "password": os.environ["DEMO_PASSWORD"]}).json()
        h = {"Authorization": f"Bearer {tok['access_token']}"}
        status = c.get("/v1/copilot/status", headers=h).json()
        results = []
        for case in cases:
            files = {"file": ("tyre.jpg", _image(case["image"]), "image/jpeg")} if case.get("image") else None
            r = c.post("/v1/copilot/messages", headers=h, data={"message": case["message"]}, files=files)
            if r.status_code != 202:
                results.append({"id": case["id"], "category": case["category"], "pass": False,
                                "checks": {"accepted": False}, "error": r.text[:300]})
                continue
            body = r.json()
            deadline = time.time() + 180
            msg = None
            while time.time() < deadline:
                sess = c.get(f"/v1/copilot/sessions/{body['session_id']}", headers=h).json()
                msg = next(m for m in sess["messages"] if m["id"] == body["assistant_message_id"])
                if (msg.get("attachments") or {}).get("status") in ("complete", "error"):
                    break
                time.sleep(1)
            trace = None
            if msg and msg.get("trace_id"):
                t = c.get(f"/v1/llmops/traces/{msg['trace_id']}", headers=h).json()
                cls_span = next((s for s in t["spans"] if s["name"] == "tool:classify_tyre_image"), None)
                triage = None
                # The agent's inspection tool creates the inspection during the turn and records it on the answer.
                iid = (msg.get("attachments") or {}).get("inspection_id") or body.get("inspection_id")
                if cls_span is not None and iid:
                    d = c.get(f"/v1/inspections/{iid}", headers=h).json()
                    triage = (d.get("prediction") or {}).get("triage")
                trace = {"status": t["status"], "latency_ms": t["latency_ms"],
                         "mlflow_trace_id": (t.get("mlflow") or {}).get("trace_id"),
                         "tokens": t["usage"]["prompt"] + t["usage"]["completion"],
                         "guardrail_flags_all": [e["rule"] for e in t["guardrail_events"] if e["action"] != "pass"],
                         "classified_triage": triage}
            results.append(_score(case, msg or {}, trace))
    passed = sum(r["pass"] for r in results)
    lat = [r["latency_ms"] for r in results if r.get("latency_ms")]
    by_cat: dict[str, list[bool]] = {}
    for res in results:
        by_cat.setdefault(res["category"], []).append(res["pass"])
    report = {
        "at": datetime.now(UTC).isoformat(), "golden_set": GOLDEN, "golden_sha256": sha256_file(GOLDEN),
        "prompt_version": status["prompt_version"], "prompt_sha256": status["prompt_sha256"],
        "llm_mode": status["mode"], "llm_model": status.get("model") or None,
        "cases": len(results), "passed": passed, "pass_rate": passed / len(results),
        "gate": {"min_pass_rate": GATE_PASS_RATE, "status": "passed" if passed / len(results) >= GATE_PASS_RATE
                 else "failed"},
        "by_category": {k: {"passed": sum(v), "total": len(v)} for k, v in by_cat.items()},
        "latency_ms": {"p50": float(np.percentile(lat, 50)) if lat else None,
                       "p95": float(np.percentile(lat, 95)) if lat else None},
        "results": results,
    }
    return report


def _link_traces(mlflow, results: list[dict], run_id: str) -> None:
    """Each case's verdict on the MLflow trace of its turn (traced by the API), so failures open on the spans."""
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    source = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="treadops.copilot_eval")
    for r in results:
        tid = r.get("mlflow_trace_id")
        if not tid:
            continue
        try:
            failed = [k for k, ok in r["checks"].items() if not ok]
            mlflow.log_feedback(trace_id=tid, name="golden_case_pass", value=bool(r["pass"]), source=source,
                                rationale="all checks passed" if not failed else "failed: " + ", ".join(failed),
                                metadata={"case": r["id"], "category": r["category"], "eval_run_id": run_id})
            mlflow.set_trace_tag(tid, "golden_case", r["id"])
            mlflow.set_trace_tag(tid, "copilot_eval_run", run_id)
        except Exception as exc:  # a trace that never reached MLflow is reported, not fatal
            print(f"could not link {r['id']} to trace {tid}: {exc}")


def log_mlflow(report: dict) -> str | None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("treadops-copilot-evals")
        with mlflow.start_run(run_name=f"copilot-{report['prompt_version']}-{report['llm_mode']}") as run:
            mlflow.log_params({"prompt_version": report["prompt_version"], "prompt_sha256": report["prompt_sha256"][:16],
                               "llm_mode": report["llm_mode"], "llm_model": report["llm_model"] or "none",
                               "golden_sha256": report["golden_sha256"][:16]})
            mlflow.log_metrics({"pass_rate": report["pass_rate"], "cases": report["cases"],
                                **{f"pass_{k}": v["passed"] / v["total"] for k, v in report["by_category"].items()}})
            mlflow.log_dict(report, "copilot_eval.json")
            _link_traces(mlflow, report["results"], run.info.run_id)
            return run.info.run_id
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("API_INTERNAL_URL", "http://127.0.0.1:8000"))
    a = ap.parse_args()
    report = run(a.api)
    report["mlflow_run_id"] = log_mlflow(report)
    out = repo_path(f"docs/evidence/llmops/copilot_eval_{report['at'][:19].replace(':', '')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("llm_mode", "prompt_version", "cases", "passed", "pass_rate", "gate",
                                             "by_category")}, indent=2))
    for r in report["results"]:
        if not r["pass"]:
            print("FAIL", r["id"], {k: v for k, v in r["checks"].items() if not v})
    return 0 if report["gate"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

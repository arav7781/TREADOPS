"""Aggregate evidence into docs/evidence/status.json: phases, artifact hashes, metrics, test results, gaps.

Everything is read from files produced by real runs; missing evidence is reported as missing, never filled in.
"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "docs/evidence"


def jload(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def junit(name: str):
    p = EVID / "tests" / f"{name}.xml"
    if not p.exists():
        return None
    root = ET.parse(p).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    a = suite.attrib
    return {k: int(a.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


def serving():
    """Active deployment and model versions as last exported by the release drill (docs/evidence/phase5)."""
    drill = jload("docs/evidence/phase5/release_drill.json")
    if not drill:
        return None
    dep = drill["deployments"]
    active = next((d for d in dep["items"] if d["state"] == "active"), None)
    return {"generation": dep["generation"], "active_registry_version": active and active["registry_version"],
            "active_bundle_sha256": active and active["bundle_sha256"],
            "models": [{k: m[k] for k in ("registry_version", "model_family", "stage", "gate_status")}
                       for m in drill["models"]]}


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def main() -> int:
    bundle = jload("artifacts/bundle/manifest.json")
    gate = jload("artifacts/bundle/gate_report.json")
    metrics = jload("reports/metrics.json")
    parity = jload("artifacts/export/parity.json")
    load = jload("docs/evidence/phase6/load_report.json")
    evals = sorted((EVID / "llmops").glob("copilot_eval_*.json")) if (EVID / "llmops").exists() else []
    status = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git": {"sha": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))},
        "data": {"source_manifest": jload("data/manifests/source_manifest.json"),
                 "split": {k: (jload("data/manifests/split_manifest.json") or {}).get(k)
                           for k in ("counts", "actual_outer_fractions", "leakage", "split_sha256")}},
        "serving": serving(),
        "dvc_pipeline_bundle": {"note": "artifacts/bundle from `dvc repro` (YOLO26n-cls baseline)",
                  "bundle_sha256": (bundle or {}).get("bundle_sha256"),
                  "registry": (bundle or {}).get("registry"), "gate_status": (gate or {}).get("status"),
                  "quality_targets": (gate or {}).get("quality_targets"),
                  "hard_gates": {g["id"]: g["status"] for g in (gate or {}).get("hard_gates", [])},
                  "summary_metrics": metrics, "parity": {k: (parity or {}).get(k) for k in
                                                         ("status", "n_images", "mean_abs_probability_error",
                                                          "max_abs_probability_error", "triage_agreement")}},
        "tests": {n: junit(n) for n in ("unit", "ml", "integration")},
        "load": load and {"measured": load["measured"], "targets": load["targets"], "overload": load.get("overload")},
        "copilot_eval": json.loads(evals[-1].read_text()) if evals else None,
        "missing_evidence": [name for name, v in {
            "bundle": bundle, "gate_report": gate, "parity": parity, "load_report": load,
            "copilot_eval": evals or None}.items() if not v],
    }
    if status["copilot_eval"]:
        status["copilot_eval"] = {k: status["copilot_eval"][k] for k in
                                  ("at", "llm_mode", "prompt_version", "cases", "passed", "pass_rate", "gate")}
    out = EVID / "status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps({"serving": status["serving"], "dvc_bundle_gate": status["dvc_pipeline_bundle"]["gate_status"],
                      "tests": status["tests"], "missing": status["missing_evidence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

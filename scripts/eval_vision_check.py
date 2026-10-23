"""Measure the vision second opinion with the production code path: YOLO alone vs YOLO + second opinion.

Runs the active bundle on a split role, asks the configured vision model about every tyre YOLO would auto-clear,
and writes docs/evidence/vision_check/<role>_<timestamp>.json. Choose rules on the calibration/policy roles; report
the locked test role once.  Usage: python scripts/eval_vision_check.py calibration,policy | test
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from treadops_api.db import new_session
from treadops_api.models import Deployment, ModelVersion
from treadops_api.settings import get_settings
from treadops_ml.preprocess import PreprocessContract, load_canonical, tensor_from_canonical
from treadops_worker import vision_check
from treadops_worker.runtime import bundles

ROOT = Path(__file__).resolve().parents[1]


def main(roles: list[str]) -> int:
    if not vision_check.enabled():
        sys.exit("set VISION_CHECK_MODEL and LLM_BASE_URL first")
    db = new_session()
    dep = db.execute(select(Deployment).where(Deployment.state == "active")).scalar_one()
    mv = db.get(ModelVersion, dep.model_version_id)
    db.close()
    bundle = bundles().get(mv.bundle_sha256, mv.bundle_uri)
    contract = PreprocessContract()
    split = pd.read_parquet(ROOT / "data/manifests/split_assignments.parquet")
    rows = split[split.role.isin(roles)].reset_index(drop=True)
    recs = []
    for r in rows.itertuples():
        data = (ROOT / f"data/processed/images/{r.original_sha256}.jpg").read_bytes()
        rgb = np.array(load_canonical(data))
        p = bundle.predict_canonical_rgb(rgb, tensor_from_canonical(data, contract))
        recs.append({"image_id": r.image_id, "label": r.label, "p": p.p_defective, "policy_triage": p.triage, "rgb": rgb})
    cleared = [r for r in recs if r["policy_triage"] == "good_candidate"]
    with ThreadPoolExecutor(8) as ex:
        opinions = list(ex.map(lambda r: vision_check.second_opinion(r["rgb"]), cleared))
    for r, o in zip(cleared, opinions, strict=True):
        r["opinion"] = o
    for r in recs:
        r["final_triage"], r["decision"], _ = vision_check.decide(r["policy_triage"], r.get("opinion"))
    defective = [r for r in recs if r["label"] == "defective"]
    good = [r for r in recs if r["label"] == "good"]
    report = {
        "at": datetime.now(UTC).isoformat(), "roles": roles, "bundle_sha256": bundle.sha,
        "model_version": mv.registry_version, "vision_model": get_settings().vision_check_model,
        "prompt_version": vision_check.prompt()[1], "prompt_sha256": vision_check.prompt()[2],
        "images": len(recs), "defective": len(defective), "good": len(good), "vision_calls": len(cleared),
        "vision_errors": sum(1 for o in opinions if "damage_visible" not in o),
        "yolo_alone": {"defective_auto_cleared": sum(r["policy_triage"] == "good_candidate" for r in defective),
                       "good_auto_cleared": sum(r["policy_triage"] == "good_candidate" for r in good)},
        "with_second_opinion": {"defective_auto_cleared": sum(r["final_triage"] == "good_candidate" for r in defective),
                                "good_sent_to_expert_by_second_opinion": sum(
                                    r["policy_triage"] == "good_candidate" and r["final_triage"] != "good_candidate"
                                    for r in good)},
        "latency_ms_p50": float(np.median([o["latency_ms"] for o in opinions])) if opinions else None,
        "per_image": [{k: r.get(k) for k in ("image_id", "label", "p", "policy_triage", "final_triage", "decision")}
                      | {"findings": (r.get("opinion") or {}).get("findings")} for r in recs],
    }
    out = ROOT / "docs/evidence/vision_check" / f"{'_'.join(roles)}_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["calibration", "policy"]))

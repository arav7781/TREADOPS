"""Measure the damage locator with the production code path: classifier alone vs classifier + locator escalation.

Runs the active bundle (classifier and its damage locator) on split roles, per source, and writes
docs/evidence/damage_locator/<roles>_<timestamp>.json. The escalation threshold was chosen on the policy role's
good tyres; report the locked test role once.  Usage: python scripts/eval_damage_locator.py calibration,policy | test
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from treadops_api.db import new_session
from treadops_api.models import Deployment, ModelVersion
from treadops_ml.preprocess import PreprocessContract, load_canonical, tensor_from_canonical
from treadops_worker.runtime import bundles

ROOT = Path(__file__).resolve().parents[1]


def _summary(recs: list[dict]) -> dict:
    defective = [r for r in recs if r["label"] == "defective"]
    good = [r for r in recs if r["label"] == "good"]
    return {
        "images": len(recs), "defective": len(defective), "good": len(good),
        "classifier_alone": {"defective_auto_cleared": sum(r["policy_triage"] == "good_candidate" for r in defective),
                             "good_auto_cleared": sum(r["policy_triage"] == "good_candidate" for r in good)},
        "with_locator": {"defective_auto_cleared": sum(r["final_triage"] == "good_candidate" for r in defective),
                         "good_sent_to_expert_by_locator": sum(
                             r["policy_triage"] == "good_candidate" and r["final_triage"] != "good_candidate"
                             for r in good)},
    }


def main(roles: list[str]) -> int:
    db = new_session()
    dep = db.execute(select(Deployment).where(Deployment.state == "active")).scalar_one()
    mv = db.get(ModelVersion, dep.model_version_id)
    db.close()
    bundle = bundles().get(mv.bundle_sha256, mv.bundle_uri)
    if bundle.damage_cfg is None:
        sys.exit("the active bundle has no damage locator")
    contract = PreprocessContract()
    split = pd.read_parquet(ROOT / "data/manifests/split_assignments.parquet")
    rows = split[split.role.isin(roles)].reset_index(drop=True)
    recs = []
    for r in rows.itertuples():
        data = (ROOT / f"data/processed/images/{r.original_sha256}.jpg").read_bytes()
        rgb = np.array(load_canonical(data))
        p = bundle.predict_canonical_rgb(rgb, tensor_from_canonical(data, contract))
        damage = bundle.locate_damage(rgb)
        final = "manual_review" if p.triage == "good_candidate" and damage["escalate"] else p.triage
        top = max(damage["boxes"], key=lambda b: b["confidence"], default=None)
        recs.append({"image_id": r.image_id, "source_id": r.source_id, "label": r.label, "p": p.p_defective,
                     "policy_triage": p.triage, "final_triage": final, "max_confidence": damage["max_confidence"],
                     "top_class": top and top["class"], "ms": damage["ms"]})
    by_source = defaultdict(list)
    for r in recs:
        by_source[r["source_id"]].append(r)
    report = {
        "at": datetime.now(UTC).isoformat(), "roles": roles, "bundle_sha256": bundle.sha,
        "model_version": mv.registry_version, "locator_version": bundle.damage_cfg["version"],
        "escalation_threshold": bundle.damage_cfg["escalation_threshold"],
        **_summary(recs), "by_source": {s: _summary(v) for s, v in sorted(by_source.items())},
        "locator_ms_p50": float(np.median([r["ms"] for r in recs])),
        "per_image": recs,
    }
    out = ROOT / "docs/evidence/damage_locator" / f"{'_'.join(roles)}_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["calibration", "policy"]))

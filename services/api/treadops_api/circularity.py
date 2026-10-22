"""Casing circularity: map a triage decision to a circular-economy pathway and estimate impact.

Model triage only *recommends* a pathway. The impact ledger counts inspector-verified dispositions only, and
every figure is labelled as an estimate derived from Michelin's published truck/bus retreading factors.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from treadops_ml.common import load_yaml, sha256_json


@lru_cache
def config() -> dict[str, Any]:
    return load_yaml("configs/sustainability.yaml")


def recommendation(triage: str, quality_status: str) -> dict[str, Any]:
    cfg = config()
    p = dict(cfg["pathways"][triage])
    f = cfg["factors"]
    p["estimate_if_verified"] = (
        {"raw_material_saved_kg": f["retread_raw_material_saved_kg"],
         "co2e_avoided_kg": round(f["retread_raw_material_saved_kg"] * f["co2e_kg_per_kg_raw_material"], 1)}
        if p["pathway"] == "retread_intake" else None
    )
    p["requires_human_clearance"] = True
    if quality_status != "accepted":
        p["note"] = "Image quality is low: recapture before any casing decision."
    p["factors_version"] = cfg["version"]
    p["scope"] = cfg["scope"]
    return p


def ledger(disposition_counts: dict[str, int], recommended_counts: dict[str, int]) -> dict[str, Any]:
    cfg = config()
    f = cfg["factors"]
    retread = int(disposition_counts.get("retread", 0))
    saved = retread * f["retread_raw_material_saved_kg"]
    return {
        "basis": "inspector-verified dispositions only (model recommendations are shown separately)",
        "verified": {k: int(disposition_counts.get(k, 0)) for k in cfg["dispositions"]},
        "recommended_by_model": {k: int(v) for k, v in recommended_counts.items()},
        "estimates": {
            "raw_material_saved_kg": saved,
            "co2e_avoided_kg": round(saved * f["co2e_kg_per_kg_raw_material"], 1),
            "new_tyre_equivalents_avoided_material": round(saved / f["new_tyre_raw_material_kg"], 2),
        },
        "factors": f,
        "factors_version": cfg["version"],
        "factors_sha256": sha256_json(f),
        "scope": cfg["scope"],
        "sources": cfg["sources"],
        "label": "ESTIMATE - published factors x verified counts; not a measured LCA of these tyres",
    }

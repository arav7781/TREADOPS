"""The damage locator inside the real inference job: a confident box stops an auto-clear, weak boxes do not.
The locator's output is scripted; a policy variant auto-clears every image so the escalation path is exercised.
"""

from __future__ import annotations

import pytest

from .conftest import admin_id, import_and_activate, jpeg, login, requires_bundle, variant_bundle


def _locator(confidence: float, threshold: float = 0.55):
    box = {"class": "cut", "confidence": confidence, "box": [0.2, 0.3, 0.5, 0.6], "escalates": confidence >= threshold}
    return lambda self, rgb: {"version": "v1", "threshold": threshold, "boxes": [box], "max_confidence": confidence,
                              "escalate": confidence >= threshold, "ms": 4.0}


def _inspect(client, key: str, seed: int) -> dict:
    from treadops_worker.tasks import run_job

    h = login(client, "inspector")
    acc = client.post("/v1/inspections", headers={**h, "Idempotency-Key": key},
                      files={"file": ("t.jpg", jpeg(seed), "image/jpeg")}).json()
    run_job(acc["job_id"])
    return client.get(f"/v1/inspections/{acc['inspection_id']}", headers=h).json()


@pytest.fixture()
def clear_all(tmp_path, monkeypatch):
    from treadops_worker import subject_gate

    monkeypatch.setattr(subject_gate, "assess", lambda emb, bundle_dir: {
        "method": "pca_residual", "version": "v1", "score": 0.5, "unusual_above": 0.74, "unusual": False})
    import_and_activate(variant_bundle(tmp_path, low=0.999999, high=0.9999995, version="clear-all"), admin_id())


@requires_bundle
def test_a_confident_damage_box_stops_an_auto_clear(client, clear_all, monkeypatch):
    from treadops_ml.bundle import Bundle

    monkeypatch.setattr(Bundle, "locate_damage", _locator(0.81))
    d = _inspect(client, "dl-cut", 61)
    p = d["prediction"]
    assert p["triage"] == "manual_review" and "damage locator sees cut (0.81)" in p["triage_reason"]
    assert p["damage"]["escalate"] and p["damage"]["boxes"][0]["class"] == "cut"
    classified = next(s for s in d["stages"] if s["name"] == "classified")
    assert classified["params"]["damage"]["boxes"][0]["confidence"] == 0.81
    assert any(line.startswith("damage locator v1: cut 0.81") and "escalated" in line for line in classified["log"])


@requires_bundle
def test_weak_boxes_are_shown_but_do_not_escalate(client, clear_all, monkeypatch):
    from treadops_ml.bundle import Bundle

    monkeypatch.setattr(Bundle, "locate_damage", _locator(0.31))
    p = _inspect(client, "dl-weak", 62)["prediction"]
    assert p["triage"] == "good_candidate" and not p["damage"]["escalate"]
    assert p["damage"]["boxes"][0]["confidence"] == 0.31

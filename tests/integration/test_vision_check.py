"""The vision second opinion inside the real inference job (Postgres, object storage, worker handler).

A policy variant that auto-clears every image forces the second opinion to run; the vision model is scripted.
"""

from __future__ import annotations

import pytest

from .conftest import admin_id, import_and_activate, jpeg, login, requires_bundle, variant_bundle

OPINION = {"model": "fake-vision", "prompt_version": "v1", "prompt_sha256": "0" * 64, "confidence": "high",
           "findings": ["gash on sidewall", "exposed cords"], "location": "sidewall", "latency_ms": 5.0}


def _inspect(client, key: str, seed: int) -> dict:
    from treadops_worker.tasks import run_job

    h = login(client, "inspector")
    acc = client.post("/v1/inspections", headers={**h, "Idempotency-Key": key},
                      files={"file": ("t.jpg", jpeg(seed), "image/jpeg")}).json()
    run_job(acc["job_id"])
    return client.get(f"/v1/inspections/{acc['inspection_id']}", headers=h).json()


@pytest.fixture()
def clear_all(tmp_path):
    import_and_activate(variant_bundle(tmp_path, low=0.999999, high=0.9999995, version="clear-all"), admin_id())


@requires_bundle
def test_a_disagreeing_second_opinion_sends_the_tyre_to_an_expert(client, clear_all, monkeypatch):
    from treadops_worker import vision_check

    monkeypatch.setattr(vision_check, "enabled", lambda: True)
    monkeypatch.setattr(vision_check, "second_opinion", lambda rgb: {**OPINION, "damage_visible": True})
    d = _inspect(client, "vc-disagree", 81)
    p = d["prediction"]
    assert p["triage"] == "manual_review" and "models disagree" in p["triage_reason"]
    assert p["second_opinion"]["decision"] == "disagrees" and p["second_opinion"]["policy_triage"] == "good_candidate"
    assert p["lineage"]["second_opinion"]["model"] == "fake-vision"
    assert p["circularity"]["pathway"] == "expert_inspection"
    classified = next(s for s in d["stages"] if s["name"] == "classified")
    assert any("vision second opinion" in line and "disagrees" in line for line in classified["log"])


@requires_bundle
def test_an_agreeing_second_opinion_keeps_the_auto_clear(client, clear_all, monkeypatch):
    from treadops_worker import vision_check

    monkeypatch.setattr(vision_check, "enabled", lambda: True)
    monkeypatch.setattr(vision_check, "second_opinion", lambda rgb: {**OPINION, "damage_visible": False, "findings": []})
    p = _inspect(client, "vc-agree", 82)["prediction"]
    assert p["triage"] == "good_candidate" and p["second_opinion"]["decision"] == "agrees"


@requires_bundle
def test_without_a_vision_model_nothing_changes(client, clear_all, monkeypatch):
    from treadops_worker import vision_check

    monkeypatch.setattr(vision_check, "enabled", lambda: False)
    p = _inspect(client, "vc-off", 83)["prediction"]
    assert p["triage"] == "good_candidate" and p["second_opinion"] is None

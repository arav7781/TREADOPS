"""The subject gate inside the real inference job: a non-tyre is rejected (no verdict), doubtful images go to an
expert. The gate's score and the vision model are scripted; a policy variant auto-clears every image so the vision
model is always asked.
"""

from __future__ import annotations

import pytest

from .conftest import admin_id, import_and_activate, jpeg, login, requires_bundle, variant_bundle

NO_TYRE = {"model": "fake-vision", "prompt_version": "v2", "prompt_sha256": "0" * 64, "tyre_visible": False,
           "subject": "a company logo", "damage_visible": False, "confidence": "high", "findings": [],
           "location": "none", "latency_ms": 5.0}


def _gate(score: float, unusual: bool):
    return lambda emb, bundle_dir: {"method": "pca_residual", "version": "v1", "score": score, "unusual_above": 0.74,
                                    "unusual": unusual}


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
def test_an_unusual_image_the_vision_model_calls_no_tyre_is_rejected_without_a_verdict(client, clear_all, monkeypatch):
    from treadops_worker import subject_gate, vision_check

    monkeypatch.setattr(subject_gate, "assess", _gate(0.84, True))
    monkeypatch.setattr(vision_check, "enabled", lambda: True)
    monkeypatch.setattr(vision_check, "second_opinion", lambda rgb: dict(NO_TYRE))
    d = _inspect(client, "sg-logo", 71)
    assert d["status"] == "rejected" and d["prediction"] is None and d["job"]["status"] == "succeeded"
    assert d["rejection"]["reason"] == "not_a_tyre" and d["rejection"]["subject"]["vision_subject"] == "a company logo"
    assert [s["name"] for s in d["stages"]][-1] == "quality"  # no YOLO verdict, attribution or classified image
    quality = d["stages"][-1]
    assert quality["params"]["status"] == "rejected" and any("REJECTED" in line for line in quality["log"])
    listed = client.get("/v1/inspections", headers=login(client, "inspector")).json()["items"]
    assert next(i for i in listed if i["inspection_id"] == d["inspection_id"])["rejected_reason"] == "not_a_tyre"


@requires_bundle
def test_one_signal_alone_sends_the_tyre_to_an_expert_instead_of_rejecting_it(client, clear_all, monkeypatch):
    from treadops_worker import subject_gate, vision_check

    monkeypatch.setattr(vision_check, "enabled", lambda: True)
    monkeypatch.setattr(subject_gate, "assess", _gate(0.60, False))  # typical image, but the vision model says no tyre
    monkeypatch.setattr(vision_check, "second_opinion", lambda rgb: dict(NO_TYRE))
    p = _inspect(client, "sg-typical", 72)["prediction"]
    assert p["quality_status"] == "low_quality" and "vision_sees_no_tyre" in p["quality"]["reasons"]
    assert p["triage"] == "manual_review" and "not auto-cleared" in p["triage_reason"]

    monkeypatch.setattr(vision_check, "enabled", lambda: False)  # unusual image, no vision model configured
    monkeypatch.setattr(subject_gate, "assess", _gate(0.80, True))
    p = _inspect(client, "sg-unusual", 73)["prediction"]
    assert p["quality_status"] == "low_quality" and "unusual_image" in p["quality"]["reasons"]
    assert p["triage"] == "manual_review" and p["quality"]["subject"]["score"] == 0.80

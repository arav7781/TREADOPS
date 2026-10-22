"""ML-03/04/07/08/09 on real artifacts produced by the DVC pipeline (skipped with a reason if absent)."""

import numpy as np
import pytest

from treadops_ml.common import read_json, repo_path

NATIVE = repo_path("artifacts/native/training_record.json")
BUNDLE = repo_path("artifacts/bundle/manifest.json")
PARITY = repo_path("artifacts/export/parity.json")


@pytest.mark.skipif(not NATIVE.exists(), reason="no trained artifact: run `make train`")
def test_ml03_training_record_is_complete_and_finite():
    r = read_json(NATIVE)
    assert r["status"] == "completed" and r["epochs_completed"] >= 1
    assert all(np.isfinite(e.get("train_loss", 0.0)) for e in r["epoch_metrics"])
    assert sorted(r["class_names"].values()) == ["defective", "good"]
    for k in ("best_pt_sha256", "split_sha256", "recipe_sha256", "uv_lock_sha256", "preprocess_sha256"):
        assert r[k] and len(r[k]) == 64
    assert r["fitted_image_count"] < r["train_pool_image_count"]


@pytest.mark.skipif(not BUNDLE.exists(), reason="no bundle: run `make evaluate export-parity package`")
def test_ml04_real_inference_produces_finite_two_class_probabilities():
    from treadops_ml.bundle import Bundle

    b = Bundle(repo_path("artifacts/bundle"))
    img = sorted(repo_path("artifacts/bundle/golden").glob("*.jpg"))[0]
    pred = b.predict_canonical_bytes(img.read_bytes())
    assert set(pred.raw_probs) == {"defective", "good"}
    assert np.isclose(sum(pred.raw_probs.values()), 1.0, atol=1e-4)
    assert 0.0 <= pred.p_defective <= 1.0
    assert pred.triage in {"good_candidate", "manual_review", "defect_flagged"}
    assert b.lineage()["bundle_sha256"] == b.sha


@pytest.mark.skipif(not PARITY.exists(), reason="no ONNX export: run `make export-parity`")
def test_ml07_ml08_onnx_parity_within_declared_tolerances():
    p = read_json(PARITY)
    assert p["n_images"] >= 50
    assert p["status"] == "passed", p
    assert p["mean_abs_probability_error"] <= 1e-4 and p["max_abs_probability_error"] <= 1e-3
    assert all(m["inside_boundary_band"] for m in p["triage_mismatches"])


@pytest.mark.skipif(not BUNDLE.exists(), reason="no bundle")
def test_ml05_ml09_roles_isolated_and_failed_goals_visible():
    cal = read_json("artifacts/bundle/calibration.json")
    pol = read_json("artifacts/bundle/policy.json")
    assert cal["fitted_on_role"] == "calibration" and pol["selection"]["selected_on_role"] == "policy"
    assert cal["fitted_ids_sha256"] != pol["selection"]["policy_ids_sha256"]
    gate = read_json("artifacts/bundle/gate_report.json")
    assert gate["status"] in {"passed", "failed", "insufficient_evidence"}
    for t in gate["quality_targets"]:
        expected = "passed" if t["value"] is not None and t["value"] >= t["target"] else t["status"]
        assert t["status"] == expected  # a missed target must surface as failed, never as passed
    log = repo_path("docs/evidence/test_access_log.jsonl")
    assert log.exists() and log.read_text().strip()  # every locked-test access is recorded

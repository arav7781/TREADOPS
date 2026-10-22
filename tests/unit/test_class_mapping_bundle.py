"""ML-01: semantic probabilities survive index permutation; a mismatched label map rejects the bundle."""

import json

import numpy as np
import pytest

from treadops_ml.bundle import Bundle, BundleInvalid, bundle_sha_from_files, verify_bundle
from treadops_ml.calibrate import TemperatureCalibrator
from treadops_ml.common import sha256_file
from treadops_ml.policy import Policy


def _stub(mapping: dict[str, int], t: float = 1.0) -> Bundle:
    b = object.__new__(Bundle)
    b.class_mapping = mapping
    b.calibrator = TemperatureCalibrator(temperature=t)
    b.policy = Policy()
    return b


def test_permuted_indices_give_identical_semantic_probability():
    raw_a = np.array([0.8, 0.2])            # index 0 = defective
    raw_b = raw_a[::-1].copy()              # index 0 = good
    a, b = _stub({"defective": 0, "good": 1}, 1.3), _stub({"defective": 1, "good": 0}, 1.3)
    assert a.p_defective(raw_a)[0] == pytest.approx(b.p_defective(raw_b)[0])
    assert a.semantic(raw_a) == b.semantic(raw_b)


def test_label_map_must_match_model_names():
    b = _stub({"defective": 0, "good": 1})
    b._verify_names({0: "defective", 1: "good"})
    with pytest.raises(BundleInvalid) as e:
        b._verify_names({0: "good", 1: "defective"})
    assert e.value.code == "class_map_mismatch"
    missing = _stub({"good": 0, "cracked": 1})
    with pytest.raises(BundleInvalid):
        missing._verify_names({0: "good", 1: "cracked"})


def _write_bundle(tmp_path, tamper: str | None = None):
    files = {}
    for name in ("best.pt", "class_mapping.json", "preprocess.json", "calibration.json", "policy.json",
                 "metrics.json", "gate_report.json", "model_card.md"):
        (tmp_path / name).write_text(name)
        files[name] = sha256_file(tmp_path / name)
    manifest = {"task": "classify", "files": files, "bundle_sha256": bundle_sha_from_files(files)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if tamper:
        (tmp_path / tamper).write_text("tampered")
    return manifest


def test_verify_bundle_accepts_intact_and_rejects_tampered(tmp_path):
    m = _write_bundle(tmp_path)
    assert verify_bundle(tmp_path)["bundle_sha256"] == m["bundle_sha256"]
    _write_bundle(tmp_path, tamper="calibration.json")
    with pytest.raises(BundleInvalid) as e:
        verify_bundle(tmp_path)
    assert e.value.code == "hash_mismatch"


def test_verify_bundle_rejects_missing_and_wrong_task(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / "policy.json").unlink()
    with pytest.raises(BundleInvalid) as e:
        verify_bundle(tmp_path)
    assert e.value.code == "missing_file"
    _write_bundle(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["task"] = "detect"
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(BundleInvalid) as e:
        verify_bundle(tmp_path)
    assert e.value.code == "wrong_task"

"""ML-06 metric correctness, ML-05 calibration behaviour, policy decisions and threshold selection."""

import numpy as np
import pytest

from treadops_ml.calibrate import apply_temperature, fit_temperature, nll
from treadops_ml.metrics import (
    binary_metrics,
    paired_bootstrap,
    probabilistic_metrics,
    triage_coverage,
    wilson,
)
from treadops_ml.policy import Policy, select_binary_threshold


def test_hand_calculated_confusion_and_rates():
    # 4 defective (3 caught), 6 good (1 false alarm)
    p = [0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.3, 0.6, 0.2, 0.4]
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    m = binary_metrics(p, np.array(y, bool), 0.5)
    assert m["confusion"] == {"tp": 3, "fn": 1, "tn": 5, "fp": 1}
    assert m["defect_recall"]["value"] == pytest.approx(3 / 4)
    assert m["defect_recall"]["numerator"] == 3 and m["defect_recall"]["denominator"] == 4
    assert m["specificity"]["value"] == pytest.approx(5 / 6)
    assert m["precision"]["value"] == pytest.approx(3 / 4)
    lo, hi = m["defect_recall"]["ci95"]
    assert lo < 0.75 < hi


def test_absent_class_is_insufficient_evidence():
    m = binary_metrics([0.1, 0.2], np.array([False, False]), 0.5)
    assert m["defect_recall"]["status"] == "insufficient_evidence" and m["defect_recall"]["value"] is None
    assert probabilistic_metrics([0.1, 0.2], [False, False])["status"] == "insufficient_evidence"


def test_wilson_interval_bounds():
    assert wilson(0, 0) is None
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and 0.65 < lo < 0.75


def test_triage_coverage_counts_false_negatives_routed_to_good():
    cov = triage_coverage(["good_candidate", "good_candidate", "defect_flagged", "manual_review"],
                          [True, False, True, False])
    assert cov["false_negatives_routed_to_good_candidate"] == 1
    assert cov["good_candidate"]["count"] == 2 and cov["defect_flagged"]["defective"] == 1


def test_temperature_scaling_uses_full_vector_and_reduces_nll():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    logits = np.where(y[:, None] == np.arange(2), 4.0, 0.0) + rng.normal(0, 2.5, (400, 2))  # overconfident
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    t, before, after = fit_temperature(probs, y)
    assert t > 1.0 and after < before
    cal = apply_temperature(probs, t)
    assert np.allclose(cal.sum(1), 1.0)
    assert np.array_equal(cal.argmax(1), probs.argmax(1))  # temperature never changes the ranking
    assert nll(apply_temperature(probs, 1.0), y) == pytest.approx(before)
    with pytest.raises(ValueError):
        apply_temperature(probs, 0.0)


def test_policy_triage_and_quality_escalation():
    pol = Policy(binary_threshold=0.4, triage_low=0.2, triage_high=0.8)
    assert pol.decide(0.05)[0] == "good_candidate"
    assert pol.decide(0.05, "low_quality")[0] == "manual_review"  # never auto-clear a bad photo
    assert pol.decide(0.5)[0] == "manual_review"
    assert pol.decide(0.8)[0] == "defect_flagged"
    assert pol.binary(0.41) and not pol.binary(0.39)
    with pytest.raises(ValueError):
        pol.decide(float("nan"))


def test_threshold_selection_maximises_specificity_at_recall_target():
    p = np.array([0.95, 0.9, 0.6, 0.3, 0.2, 0.1, 0.5, 0.05])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    sel = select_binary_threshold(p, y, 0.75)
    assert sel["status"] == "ok" and sel["recall"] >= 0.75
    assert sel["threshold"] == pytest.approx(0.6)  # highest threshold still catching 3/4 defects
    assert select_binary_threshold(p, np.zeros(8, bool), 0.95)["status"] == "insufficient_evidence"


def test_paired_bootstrap_detects_recall_drop_by_parent():
    groups = [f"g{i // 2}" for i in range(40)]
    y = np.array([True] * 20 + [False] * 20)
    a = y.copy()
    b = y.copy()
    b[0:12:2] = False  # challenger misses 6 defects spread over 6 of the 10 defective parents
    r = paired_bootstrap(groups, y, a, b, "recall", resamples=300)
    assert r["status"] == "ok" and r["difference"] == pytest.approx(-0.3)
    assert r["ci95"][1] < 0 and r["parents"] == 20
    clustered = y.copy()
    clustered[:6] = False  # same 6 misses inside 3 parents: resamples can omit them all
    wide = paired_bootstrap(groups, y, a, clustered, "recall", resamples=300)
    assert wide["ci95"][1] == pytest.approx(0.0) and wide["ci95"][0] < r["ci95"][0] + 0.2

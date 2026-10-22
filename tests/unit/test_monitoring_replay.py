"""Drift statistics and deterministic replay schedules."""

import numpy as np
import pytest

from treadops_ml.monitoring import domain_classifier_auc, frozen_bins, psi
from treadops_ml.replay import Parent, generate


def test_psi_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.5, 0.1, 2000)
    assert psi(ref, rng.normal(0.5, 0.1, 500))["psi"] < 0.05
    assert psi(ref, ref * 0.45)["psi"] > 0.2


def test_bins_are_frozen_from_reference_only():
    ref = np.linspace(0, 1, 101)
    edges = frozen_bins(ref, 10)
    assert edges[0] == -np.inf and edges[-1] == np.inf and len(edges) == 11
    out = psi(ref, np.full(50, 5.0))  # values far outside still land in the open top bin
    assert sum(out["current_distribution"]) == pytest.approx(1.0, abs=1e-3)


def test_domain_classifier_auc_reflects_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (300, 8))
    assert domain_classifier_auc(a, rng.normal(0, 1, (300, 8)))["auc"] < 0.65
    assert domain_classifier_auc(a, rng.normal(1.0, 1, (300, 8)))["auc"] > 0.85


POOL = [Parent(f"p{i:03d}", "defective" if i % 2 else "good", f"g{i // 3}") for i in range(60)]


def test_replay_is_deterministic_and_never_training_eligible():
    a = generate("low_light", 42, 600, 2.0, POOL)
    b = generate("low_light", 42, 600, 2.0, POOL)
    assert a == b
    assert all(not e["training_eligible"] and e["metadata_origin"] == "simulated" for e in a)
    assert all(not e["transform"] for e in a[:300])
    shifted = [e for e in a[300:] if e["transform"]]
    assert 0.6 < len(shifted) / 300 < 0.8 and shifted[0]["transform"] == {"brightness_factor": 0.45}
    assert a[1]["offset_s"] == pytest.approx(1.0)  # 2 events/s schedule, independent of labels


def test_class_prior_shift_and_burst_duplicates():
    cps = generate("class_prior_shift", 42, 600, 2.0, POOL)
    late = [e["label"] for e in cps[300:]]
    assert late.count("defective") / len(late) > 0.75
    burst = generate("burst_and_outage", 7, 200, 2.0, POOL)
    dups = [e for e in burst if e["duplicate_of"]]
    assert dups and all(any(o["event_id"] == d["duplicate_of"] for o in burst) for d in dups)
    assert burst[1]["offset_s"] == pytest.approx(0.2)  # burst rate 10/s


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        generate("snowstorm", 1, 10, 1.0, POOL)

"""The unusual-image score: distance of an embedding from the bundle's tyre reference subspace."""

import numpy as np

from treadops_worker import subject_gate


def test_residual_is_zero_inside_the_reference_subspace_and_the_norm_outside(tmp_path):
    components = np.eye(4, 6, dtype=np.float32)  # reference subspace = first 4 axes
    mean = np.zeros(6, dtype=np.float32)
    inside, outside = np.array([0.3, -0.2, 0.5, 0.1, 0, 0]), np.array([0, 0, 0, 0, 0.6, 0.8])
    r = subject_gate.residual(np.stack([inside, outside]), mean, components)
    assert r[0] < 1e-6 and abs(r[1] - 1.0) < 1e-6

    np.savez(tmp_path / "reference_pca.npz", mean=mean, components=components)
    threshold = subject_gate.config()["unusual_above"]
    assert subject_gate.assess(inside, tmp_path)["unusual"] is False
    out = subject_gate.assess(outside, tmp_path)
    assert out["unusual"] is (threshold < 1.0) and out["score"] == 1.0 and out["unusual_above"] == threshold

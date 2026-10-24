"""DATA-01..04 against the real, frozen D1 manifests (skipped with a reason when data is absent)."""

import json

import pandas as pd
import pytest

from treadops_ml.common import load_yaml, read_json, repo_path
from treadops_ml.split import assign_roles

MANIFEST = repo_path("data/manifests/image_manifest.parquet")
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="run `make prepare-data` first (D1 not prepared)")


def test_data01_source_audit_records_actual_counts_and_hashes():
    src = read_json("data/manifests/source_manifest.json")
    assert src["doi"] == "10.17632/bn7ch8tvyp.1" and src["license"] == "CC BY 4.0"
    assert len(src["archive_sha256"]) == 64
    c = src["actual_counts"]
    assert c["canonical_images"] == sum(c["by_label"].values())
    assert c["archive_members_images"] == c["canonical_images"] + c["exact_duplicates"] + c["quarantined"]
    q = read_json("data/manifests/quality_report.json")
    assert q["publisher_hash_checks"].get("mismatch", 0) == 0


def test_data02_no_hash_group_or_ancestor_crosses_roles():
    split = pd.read_parquet(repo_path("data/manifests/split_assignments.parquet"))
    assert (split.groupby("group_id").role.nunique() == 1).all()
    assert (split.groupby("original_sha256").role.nunique() == 1).all()
    dev = set(split[split.role.isin(["dev_train", "dev_val"])].image_id)
    for role in ("calibration", "policy", "test"):
        rows = [json.loads(line) for line in repo_path(f"data/eval/{role}.jsonl").read_text().splitlines()]
        assert not dev & {r["image_id"] for r in rows}
    exported = {p.stem for p in repo_path("data/yolo/dev").rglob("*.jpg")}
    assert exported == dev  # only the development cohort is exported into the training root


def test_data03_split_is_deterministic():
    df = pd.read_parquet(MANIFEST)
    cfg = load_yaml("configs/split.yaml")
    a, _ = assign_roles(df, cfg)
    b, _ = assign_roles(df, cfg)
    cols = ["image_id", "group_id", "role"]
    assert a.sort_values("image_id")[cols].reset_index(drop=True).equals(b.sort_values("image_id")[cols].reset_index(drop=True))
    frozen = pd.read_parquet(repo_path("data/manifests/split_assignments.parquet")).set_index("image_id").role
    assert (a.set_index("image_id").role.reindex(frozen.index) == frozen).all()


def test_data04_only_development_roles_are_training_eligible():
    df = pd.read_parquet(MANIFEST)
    a, _ = assign_roles(df, load_yaml("configs/split.yaml"))
    assert set(a[a.training_eligible].role) == {"dev_train", "dev_val"}
    assert not a[a.role.isin(["calibration", "policy", "test"])].training_eligible.any()


def test_outer_fractions_close_to_target_after_grouping():
    m = read_json("data/manifests/split_manifest.json")
    for k, target in {"train": 0.70, "val": 0.15, "test": 0.15}.items():
        assert abs(m["actual_outer_fractions"][k] - target) < 0.03


def test_data05_extra_sources_never_move_a_d1_image():
    frozen = read_json("docs/evidence/data/d1_frozen_roles.json")
    split = pd.read_parquet(repo_path("data/manifests/split_assignments.parquet"))
    d1 = split[split.source_id == frozen["source_id"]] if "source_id" in split.columns else split
    from treadops_ml.common import sha256_json

    assert sha256_json(sorted(zip(d1.image_id, d1.role, strict=True))) == frozen["image_role_pairs_sha256"]
    for sid, g in split.groupby("source_id") if "source_id" in split.columns else []:
        assert set(g.role) <= {"dev_train", "dev_val", "calibration", "policy", "test"}, sid

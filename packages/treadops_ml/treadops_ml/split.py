"""Stage `split`: grouped, label-stratified, deterministic split into frozen roles.

outer train (70%) -> dev_train / dev_val        fit weights / select checkpoint
outer val   (15%) -> calibration / policy        temperature fit / threshold selection (no weight fitting)
outer test  (15%) -> test                        locked final benchmark only

Groups (exact + reviewed near-duplicate clusters from `prepare`) never cross roles. Groups whose members carry
conflicting publisher labels are excluded from every role and reported.

Each source is carved on its own (a group's home is the primary source when it holds any primary image), with the
same fractions and seed, so adding a source never moves a primary-source image to another role.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from typing import Any

import pandas as pd

from .common import load_yaml, repo_path, sha256_bytes, sha256_json, write_json

ROLES = ("dev_train", "dev_val", "calibration", "policy", "test")
OUTER_OF = {"dev_train": "train", "dev_val": "train", "calibration": "val", "policy": "val", "test": "test"}


def _order(seed: int, gid: str) -> str:
    return sha256_bytes(f"{seed}:{gid}".encode())


def _carve(groups: dict[str, list[str]], label_of: dict[str, str], fraction: float, seed: int, salt: str):
    """Take approximately `fraction` of images per label, whole groups at a time, in seeded-hash order."""
    taken, rest = {}, {}
    by_label: dict[str, list[str]] = defaultdict(list)
    for gid in groups:
        by_label[label_of[gid]].append(gid)
    for _label, gids in sorted(by_label.items()):
        total = sum(len(groups[g]) for g in gids)
        target = round(total * fraction)
        n = 0
        for gid in sorted(gids, key=lambda g: _order(seed, salt + g)):
            if n < target and n + len(groups[gid]) <= target + max(1, len(groups[gid]) // 2):
                taken[gid] = groups[gid]
                n += len(groups[gid])
            else:
                rest[gid] = groups[gid]
    return taken, rest


def _carve_roles(groups: dict[str, list[str]], label_of: dict[str, str], cfg: dict[str, Any]) -> dict[str, str]:
    seed = cfg["seed"]
    test, rest = _carve(groups, label_of, cfg["outer"]["test"], seed, "test:")
    val_frac_of_rest = cfg["outer"]["val"] / (cfg["outer"]["train"] + cfg["outer"]["val"])
    val, train = _carve(rest, label_of, val_frac_of_rest, seed, "val:")
    dev_val, dev_train = _carve(train, label_of, cfg["dev_val_fraction"], seed, "dev:")
    calibration, policy = _carve(val, label_of, cfg["calibration_fraction"], seed, "cal:")
    role_of_group = {}
    for role, part in (("dev_train", dev_train), ("dev_val", dev_val), ("calibration", calibration),
                       ("policy", policy), ("test", test)):
        for gid in part:
            role_of_group[gid] = role
    return role_of_group


def assign_roles(df: pd.DataFrame, cfg: dict[str, Any], primary_source: str | None = None
                 ) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary_source = primary_source or load_yaml("configs/data.yaml")["source"]["id"]
    ok = df[df.status == "ok"].copy()
    if "source_id" not in ok.columns:
        ok["source_id"] = primary_source
    conflict_groups = set(ok[ok.group_label_conflict.astype("boolean").fillna(False).astype(bool)].group_id)
    excluded = ok[ok.group_id.isin(conflict_groups)]
    ok = ok[~ok.group_id.isin(conflict_groups)]
    sources_of = ok.groupby("group_id").source_id.agg(set).to_dict()
    order = [primary_source, *sorted(set(ok.source_id) - {primary_source})]
    home = {g: next(s for s in order if s in srcs) for g, srcs in sources_of.items()}
    # A group's stratification label comes from its home source, so an attached photo can never re-stratify it.
    label_of = {g: lab for g, lab, s in zip(ok.group_id, ok.label, ok.source_id, strict=True) if s == home[g]}

    role_of_group: dict[str, str] = {}
    for src in order:
        # Carve this source's home groups by their members *from this source*: duplicates attached from another
        # source follow the group's role without changing the carve.
        groups: dict[str, list[str]] = defaultdict(list)
        for iid, gid, sid in zip(ok.image_id, ok.group_id, ok.source_id, strict=True):
            if home[gid] == src and sid == src:
                groups[gid].append(iid)
        role_of_group.update(_carve_roles(groups, label_of, cfg))
    ok["role"] = ok.group_id.map(role_of_group)
    ok["outer_split"] = ok.role.map(OUTER_OF)
    ok["training_eligible"] = ok.role.isin(["dev_train", "dev_val"])

    # Leakage assertions: no hash, group or ancestor crosses roles.
    leakage = {
        "groups_in_multiple_roles": int((ok.groupby("group_id").role.nunique() > 1).sum()),
        "original_sha_in_multiple_roles": int((ok.groupby("original_sha256").role.nunique() > 1).sum()),
        "image_ids_unassigned": int(ok.role.isna().sum()),
    }
    if any(leakage.values()):
        raise AssertionError(f"leakage check failed: {leakage}")
    return ok, {"leakage": leakage, "excluded_conflicting_groups": sorted(conflict_groups),
                "excluded_images": int(len(excluded))}


def _link(src, dst, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def run(config_path: str = "configs/split.yaml", manifest: str = "data/manifests/image_manifest.parquet") -> dict:
    cfg = load_yaml(config_path)
    data_cfg = load_yaml("configs/data.yaml")
    df = pd.read_parquet(repo_path(manifest))
    assigned, report = assign_roles(df, cfg, data_cfg["source"]["id"])
    assigned = assigned.sort_values("image_id").reset_index(drop=True)

    processed = repo_path(data_cfg["processed_dir"]) / "images"
    yolo_root = repo_path(cfg["yolo_dev_root"])
    eval_dir = repo_path(cfg["eval_dir"])
    for d in (yolo_root, eval_dir):
        if d.exists():
            shutil.rmtree(d)
    eval_dir.mkdir(parents=True)
    rows_by_role: dict[str, list[dict]] = defaultdict(list)
    for r in assigned.itertuples():
        src = processed / f"{r.original_sha256}.jpg"
        rel = os.path.relpath(src, repo_path("."))
        if r.role in ("dev_train", "dev_val"):
            sub = "train" if r.role == "dev_train" else "val"
            _link(src, yolo_root / sub / r.label / f"{r.image_id}.jpg", cfg["link_mode"])
        rows_by_role[r.role].append({
            "image_id": r.image_id, "label": r.label, "group_id": r.group_id, "role": r.role,
            "original_sha256": r.original_sha256, "path": rel, "source_id": r.source_id,
        })
    for role in ("calibration", "policy", "test"):
        with (eval_dir / f"{role}.jsonl").open("w") as fh:
            for row in rows_by_role[role]:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    # The dev roles are also listed so evaluation/packaging can find them without walking folders.
    for role in ("dev_train", "dev_val"):
        with (eval_dir / f"{role}.jsonl").open("w") as fh:
            for row in rows_by_role[role]:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    assignments = assigned[["image_id", "group_id", "label", "outer_split", "role", "original_sha256", "source_id"]]
    records = assignments.to_dict("records")
    counts = {
        role: {
            "images": int((assigned.role == role).sum()),
            "groups": int(assigned[assigned.role == role].group_id.nunique()),
            "by_label": {k: int(v) for k, v in assigned[assigned.role == role].label.value_counts().sort_index().items()},
        }
        for role in ROLES
    }
    by_source = {
        sid: {role: {k: int(v) for k, v in g[g.role == role].label.value_counts().sort_index().items()}
              for role in ROLES}
        for sid, g in assigned.groupby("source_id")
    }
    total = int(len(assigned))
    manifest_out = {
        "schema_version": 1,
        "config": cfg,
        "source_manifest_sha256": sha256_json(json.loads(repo_path("data/manifests/source_manifest.json").read_text())),
        "counts": counts,
        "counts_by_source": by_source,
        "actual_outer_fractions": {
            o: round(sum(counts[r]["images"] for r in ROLES if OUTER_OF[r] == o) / total, 4) for o in ("train", "val", "test")
        },
        "training_eligible_images": int(assigned.training_eligible.sum()),
        **report,
        "assignments_sha256": sha256_json(records),
    }
    manifest_out["split_sha256"] = sha256_json({k: v for k, v in manifest_out.items() if k != "split_sha256"})
    write_json("data/manifests/split_manifest.json", manifest_out)
    assignments.to_parquet(repo_path("data/manifests/split_assignments.parquet"), index=False)
    return {k: manifest_out[k] for k in ("counts", "actual_outer_fractions", "leakage", "excluded_images", "split_sha256")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="grouped deterministic split")
    ap.add_argument("--config", default="configs/split.yaml")
    args = ap.parse_args(argv)
    print(json.dumps(run(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

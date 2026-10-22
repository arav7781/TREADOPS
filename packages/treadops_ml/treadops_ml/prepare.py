"""Stage `prepare`: audit the D1 archive and build canonical, de-duplicated image records.

Outputs (all deterministic for a given archive + config; no wall-clock timestamps):
    data/processed/images/<original_sha256>.jpg     canonical RGB derivatives (EXIF applied, no metadata)
    data/manifests/image_manifest.parquet            one row per archive image member (incl. quarantined)
    data/manifests/source_manifest.json              source, licence, archive hash, actual counts
    data/manifests/duplicate_review.csv              perceptual / embedding duplicate candidates + decisions
    data/manifests/quality_report.json               quarantine reasons, descriptor summaries, hash checks
    data/manifests/class_mapping.json                semantic labels (model indices are read from model.names)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from .common import load_yaml, read_json, repo_path, sha256_bytes, sha256_file, write_json
from .descriptors import DESCRIPTOR_VERSION, image_descriptors, phash
from .preprocess import (
    ImageRejected,
    Limits,
    PreprocessContract,
    canonical_from_raw,
    load_canonical,
)

ID_NAMESPACE = uuid.UUID("6b6f2d1e-4a1c-5e0b-9a77-7472656164ff")  # TreadOps image-id namespace


def image_id_for(source_id: str, original_sha: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"{source_id}:{original_sha}"))


def _label_from_path(name: str, label_map: dict[str, str]) -> str | None:
    for part in PurePosixPath(name).parts[:-1]:
        if part.lower() in label_map:
            return label_map[part.lower()]
    return None


def _process(args: tuple[str, bytes, dict, dict]) -> dict[str, Any]:
    name, data, limits_d, contract_d = args
    limits = Limits(**limits_d)
    contract = PreprocessContract.from_dict(contract_d)
    rec: dict[str, Any] = {"original_sha256": sha256_bytes(data), "bytes": len(data)}
    try:
        canonical, ing = canonical_from_raw(data, contract, limits)  # same ingest path as the serving API
    except ImageRejected as exc:
        rec.update(status="quarantined", quarantine_reason=exc.code, quarantine_detail=exc.message)
        return rec
    rgb = load_canonical(canonical)
    rec.update(
        status="ok",
        canonical=canonical,
        canonical_sha256=sha256_bytes(canonical),
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        original_width=ing.info["original_width"],
        original_height=ing.info["original_height"],
        exif_orientation=ing.info["exif_orientation"],
        had_gps=ing.info["had_gps"],
        source_format=ing.info["format"],
        phash=f"{phash(rgb):016x}",
        **{f"desc_{k}": v for k, v in image_descriptors(rgb).items()},
    )
    return rec


def _capture_index(member: str) -> int:
    """Publisher file names encode capture order, e.g. 'Defective (412).jpg' -> 412 (-1 if absent)."""
    import re

    m = re.search(r"\((\d+)\)", PurePosixPath(member).name)
    return int(m.group(1)) if m else -1


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _embeddings(paths: list[Path], contract: PreprocessContract, device: str) -> np.ndarray:
    from .descriptors import ReferenceEncoder
    from .preprocess import tensor_from_canonical
    from .weights import pretrained

    enc = ReferenceEncoder(str(pretrained("yolo26n-cls.pt")), device=device)
    out = []
    for i in range(0, len(paths), 64):
        batch = np.concatenate([tensor_from_canonical(p.read_bytes(), contract) for p in paths[i : i + 64]])
        out.append(enc.embed(batch))
    return np.concatenate(out) if out else np.zeros((0, enc.dim), np.float32)


def run(config_path: str = "configs/data.yaml", archive: str | None = None, workers: int = 8) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    archive_path = repo_path(archive or cfg["archive"])
    if not archive_path.exists():
        raise SystemExit(
            f"archive not found: {archive_path}. Download it with `make fetch-data` or from {cfg['source']['url']} "
            "(Download All) and pass ARCHIVE=/path/to/zip."
        )
    contract = PreprocessContract(
        canonical_max_side=cfg["canonical"]["max_side"],
        canonical_resample=cfg["canonical"]["resample"],
        canonical_format=cfg["canonical"]["format"],
        canonical_quality=cfg["canonical"]["quality"],
        canonical_subsampling=cfg["canonical"]["subsampling"],
    )
    limits = Limits(
        max_bytes=cfg["limits"]["max_file_bytes"],
        max_decoded_pixels=cfg["limits"]["max_decoded_pixels"],
        min_side=cfg["limits"]["min_side"],
    )
    source_id = cfg["source"]["id"]
    label_map = {k.lower(): v for k, v in cfg["label_map"].items()}
    allowed = tuple(s.lower() for s in cfg["allowed_suffixes"])
    processed = repo_path(cfg["processed_dir"]) / "images"
    processed.mkdir(parents=True, exist_ok=True)
    manifests = repo_path(cfg["manifest_dir"])
    manifests.mkdir(parents=True, exist_ok=True)

    receipt_path = repo_path(cfg["source"]["receipt"])
    receipt = read_json(receipt_path) if receipt_path.exists() else None
    publisher_files = (receipt or {}).get("publisher", {}).get("files", {})
    archive_sha = sha256_file(archive_path)

    members: list[tuple[str, str | None]] = []
    skipped: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as zf:
        infos = sorted(zf.infolist(), key=lambda i: i.filename)
        for info in infos:
            name = info.filename
            if info.is_dir() or "__MACOSX" in name or PurePosixPath(name).name.startswith("."):
                continue
            if not name.lower().endswith(allowed):
                skipped.append({"member": name, "reason": "unsupported_suffix"})
                continue
            if info.file_size > limits.max_bytes:
                skipped.append({"member": name, "reason": "member_too_large"})
                continue
            members.append((name, _label_from_path(name, label_map)))

        records: list[dict[str, Any]] = []
        results: list[tuple[str, str | None, dict]] = []
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            batch = workers * 4  # bound raw bytes in flight
            for start in range(0, len(members), batch):
                chunk = members[start : start + batch]
                futs = [
                    (name, label, pool.submit(_process, (name, zf.read(name), limits.__dict__, contract.to_dict())))
                    for name, label in chunk
                ]
                results.extend((name, label, fut.result()) for name, label, fut in futs)
            for name, label, rec in results:
                path = PurePosixPath(name)
                folder = path.parent.name
                pub = publisher_files.get(f"{folder}/{path.name}")
                if pub is None:
                    pub_check = "unlisted"
                else:
                    pub_check = "match" if pub["sha256"] == rec["original_sha256"] else "mismatch"
                rec.update(member=name, folder=folder, label=label, publisher_sha_check=pub_check)
                if label is None and rec["status"] == "ok":
                    rec.update(status="quarantined", quarantine_reason="unknown_label", quarantine_detail=folder)
                if pub_check == "mismatch" and rec["status"] == "ok":
                    rec.update(status="quarantined", quarantine_reason="publisher_hash_mismatch")
                records.append(rec)

    # Exact duplicates: identical original bytes collapse to one canonical record.
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_sha[r["original_sha256"]].append(r)
    for _sha, rs in by_sha.items():
        labels = {r["label"] for r in rs if r["status"] == "ok"}
        if len(labels) > 1:
            for r in rs:
                r.update(status="quarantined", quarantine_reason="label_conflict_exact_duplicate")
        for dup in rs[1:]:
            if dup["status"] == "ok":
                dup.update(status="duplicate", duplicate_of_member=rs[0]["member"])

    canon = [r for r in records if r["status"] == "ok"]
    for r in canon:
        r["image_id"] = image_id_for(source_id, r["original_sha256"])
        r["object_key"] = f"images/{r['original_sha256']}.jpg"
        out = processed / f"{r['original_sha256']}.jpg"
        if not out.exists() or sha256_file(out) != r["canonical_sha256"]:
            out.write_bytes(r["canonical"])
    for r in records:
        r.pop("canonical", None)

    # Near-duplicate grouping: pHash (automatic) + reference-embedding cosine (reviewed threshold).
    canon.sort(key=lambda r: r["image_id"])
    ids = [r["image_id"] for r in canon]
    uf = _UnionFind(ids)
    hashes = np.array([int(r["phash"], 16) for r in canon], dtype=np.uint64)
    import os

    emb = _embeddings([processed / f"{r['original_sha256']}.jpg" for r in canon], contract,
                      os.environ.get("PREPARE_DEVICE", "cpu"))
    sims = emb @ emb.T
    ham = np.bitwise_count(hashes[:, None] ^ hashes[None, :]).astype(np.int32)
    dup = cfg["duplicates"]
    seq_num = np.array([_capture_index(r["member"]) for r in canon])
    folder = np.array([r["folder"] for r in canon])
    iu, ju = np.triu_indices(len(canon), k=1)
    h, c = ham[iu, ju], sims[iu, ju]
    same_seq = (folder[iu] == folder[ju]) & (np.abs(seq_num[iu] - seq_num[ju]) <= dup["sequence_index_distance_max"])
    rule_phash = (h <= dup["phash_hamming_max"]) & (c >= dup["phash_requires_cosine_min"])
    rule_emb = c >= dup["embedding_cosine_min"]
    rule_seq = same_seq & (c >= dup["sequence_cosine_min"])
    cand = (h <= dup["phash_hamming_max"] + 4) | (c >= dup["review_floor"]) | rule_seq
    review_rows = []
    for k in np.where(cand)[0].tolist():
        i, j = int(iu[k]), int(ju[k])
        rule = ("embedding" if rule_emb[k] else "phash+cosine" if rule_phash[k] else
                "capture_sequence" if rule_seq[k] else "phash_rejected_low_cosine" if h[k] <= dup["phash_hamming_max"]
                else "below_threshold")
        group = bool(rule_emb[k] or rule_phash[k] or rule_seq[k])
        if group:
            uf.union(ids[i], ids[j])
        review_rows.append({
            "a_image_id": ids[i], "b_image_id": ids[j],
            "a_member": canon[i]["member"], "b_member": canon[j]["member"],
            "a_label": canon[i]["label"], "b_label": canon[j]["label"],
            "phash_hamming": int(h[k]), "embedding_cosine": round(float(c[k]), 4),
            "decision": "group" if group else "separate", "rule": rule,
        })
    members_of: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        members_of[uf.find(i)].append(i)
    group_of = {}
    for _root, ms in members_of.items():
        gid = "grp-" + sha256_bytes(",".join(sorted(ms)).encode())[:16]
        for m in ms:
            group_of[m] = gid
    conflicts = 0
    group_labels: dict[str, set] = defaultdict(set)
    for r in canon:
        r["group_id"] = group_of[r["image_id"]]
        group_labels[r["group_id"]].add(r["label"])
    for r in canon:
        r["group_label_conflict"] = len(group_labels[r["group_id"]]) > 1
        conflicts += int(r["group_label_conflict"])

    df = pd.DataFrame(records)
    for col in ("quarantine_reason", "quarantine_detail", "duplicate_of_member", "image_id", "object_key"):
        if col not in df.columns:
            df[col] = None
    df["source_id"] = source_id
    df["image_origin"] = "public_dataset"
    df["metadata_origin"] = "observed"
    df["label_origin"] = "publisher"
    df["parent_image_id"] = None
    df["preprocess_version"] = contract.version
    df["descriptor_version"] = DESCRIPTOR_VERSION
    gid = {r["image_id"]: r["group_id"] for r in canon}
    conflict = {r["image_id"]: r["group_label_conflict"] for r in canon}
    df["group_id"] = df.get("image_id", pd.Series(dtype=str)).map(gid)
    df["group_label_conflict"] = df.get("image_id", pd.Series(dtype=str)).map(conflict)
    df = df.sort_values("member").reset_index(drop=True)
    df.to_parquet(manifests / "image_manifest.parquet", index=False)
    np.save(repo_path(cfg["processed_dir"]) / "reference_embeddings.npy", emb)
    write_json(repo_path(cfg["processed_dir"]) / "reference_embeddings_index.json", ids)

    with (manifests / "duplicate_review.csv").open("w", newline="") as fh:
        cols = ["a_image_id", "b_image_id", "a_member", "b_member", "a_label", "b_label",
                "phash_hamming", "embedding_cosine", "decision", "rule"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in sorted(review_rows, key=lambda r: (-r["embedding_cosine"], r["a_image_id"])):
            w.writerow(row)

    ok = df[df.status == "ok"]
    counts = {
        "archive_members_images": len(members),
        "skipped_members": len(skipped),
        "canonical_images": int(len(ok)),
        "by_label": {k: int(v) for k, v in ok.label.value_counts().sort_index().items()},
        "quarantined": int((df.status == "quarantined").sum()),
        "exact_duplicates": int((df.status == "duplicate").sum()),
        "groups": int(ok.group_id.nunique()),
        "multi_member_groups": int((ok.group_id.value_counts() > 1).sum()),
        "images_in_label_conflicting_groups": int(conflicts),
    }
    source_manifest = {
        "source_id": source_id,
        "name": cfg["source"]["name"],
        "doi": cfg["source"]["doi"],
        "url": cfg["source"]["url"],
        "version": 1,
        "license": cfg["source"]["license"],
        "attribution": cfg["source"]["attribution"],
        "retrieved_at": (receipt or {}).get("retrieval_finished_at"),
        "retrieval_endpoint": (receipt or {}).get("source_endpoint"),
        "archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "receipt_archive_sha256_matches": (receipt or {}).get("archive_sha256") == archive_sha if receipt else None,
        "publisher_described_images": 1854,
        "actual_counts": counts,
        "preprocess_contract": contract.to_dict(),
        "preprocess_sha256": contract.sha256,
    }
    write_json(manifests / "source_manifest.json", source_manifest)
    write_json(manifests / "class_mapping.json", {"labels": sorted(label_map.values()), "positive_label": "defective"})
    pub_checks = Counter(df.publisher_sha_check)
    desc_cols = [c for c in df.columns if c.startswith("desc_")]
    quality = {
        "quarantine_reasons": {k: int(v) for k, v in df[df.status == "quarantined"].quarantine_reason.value_counts().items()},
        "quarantined_members": df[df.status == "quarantined"][["member", "quarantine_reason"]].to_dict("records"),
        "skipped_members": skipped,
        "publisher_hash_checks": dict(pub_checks),
        "descriptor_summary": {
            c[5:]: {q: float(ok[c].quantile(v)) for q, v in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95))}
            for c in desc_cols
        },
        "source_resolutions": {
            f"{w}x{h}": int(n)
            for (w, h), n in ok.groupby(["original_width", "original_height"]).size().sort_values(ascending=False).head(10).items()
        },
        "exif_orientations": {str(k): int(v) for k, v in ok.exif_orientation.fillna(0).astype(int).value_counts().items()},
    }
    write_json(manifests / "quality_report.json", quality)
    return {"counts": counts, "publisher_hash_checks": dict(pub_checks)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--archive", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    import json

    print(json.dumps(run(args.config, args.archive, args.workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

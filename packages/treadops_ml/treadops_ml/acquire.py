"""D1 acquisition through the publisher's own mechanisms (Mendeley Data public API).

Two publisher-provided paths are used, and nothing else:

* ``/public-api/zip/<id>/download/<version>`` - the "Download All" archive the dataset page links to.
  It 302-redirects to a short-lived signed URL, so each ranged chunk asks for a fresh redirect.
* ``/public-api/datasets/<id>/files?folder_id=...`` - per-file listing with the publisher's SHA-256.
  It returns at most 1000 entries per folder, so it is only used to cross-check archive members.

The archive is fetched with parallel HTTP range requests and resumed chunk by chunk. A local
``--archive PATH`` bypasses the network entirely (see docs/runbooks/data-acquisition.md).

``--extra`` fetches the other classification sources in configs/data.yaml the same way (each publisher's own
download endpoint, checked against its published checksum); ``--damage`` fetches the damage locator's Roboflow
Universe exports in configs/damage.yaml through Roboflow's export API (needs ROBOFLOW_API_KEY).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

DATASET_ID = "bn7ch8tvyp"
VERSION = 1
API = "https://data.mendeley.com/public-api"
ZIP_ENDPOINT = f"{API}/zip/{DATASET_ID}/download/{VERSION}"
CHUNK = 32 * 1024 * 1024


def _signed_url(client: httpx.Client, endpoint: str = ZIP_ENDPOINT) -> str:
    resp = client.get(endpoint, follow_redirects=False)
    if resp.status_code not in (301, 302, 303, 307, 308):
        raise RuntimeError(f"publisher zip endpoint returned {resp.status_code}, expected a redirect")
    return resp.headers["location"]


def _total_size(client: httpx.Client, endpoint: str = ZIP_ENDPOINT) -> int:
    url = _signed_url(client, endpoint)
    resp = client.get(url, headers={"Range": "bytes=0-0"})
    if resp.status_code != 206:
        raise RuntimeError(f"range request not honoured (status {resp.status_code})")
    return int(resp.headers["content-range"].split("/")[-1])


def _fetch_chunk(index: int, start: int, end: int, parts: Path, attempts: int = 6, endpoint: str = ZIP_ENDPOINT) -> Path:
    part = parts / f"{index:05d}.part"
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return part
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0)) as client:
                url = _signed_url(client, endpoint)
                with client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as resp:
                    if resp.status_code != 206:
                        raise RuntimeError(f"chunk {index}: status {resp.status_code}")
                    tmp = part.with_suffix(".tmp")
                    with tmp.open("wb") as fh:
                        for block in resp.iter_bytes(1 << 20):
                            fh.write(block)
            if tmp.stat().st_size != expected:
                raise RuntimeError(f"chunk {index}: got {tmp.stat().st_size} bytes, expected {expected}")
            tmp.replace(part)
            return part
        except Exception as exc:  # network errors are transient here; bounded retries
            if attempt == attempts:
                raise
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"chunk {index} attempt {attempt} failed ({exc}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def download_archive(dest: Path, workers: int = 12, seed_prefix: Path | None = None,
                     endpoint: str = ZIP_ENDPOINT, chunk: int = CHUNK, attempts: int = 6) -> Path:
    """Download a publisher file to ``dest`` using resumable parallel ranged chunks. ``endpoint`` must redirect to a
    short-lived signed URL that honours HTTP ranges (Mendeley Data and Harvard Dataverse both do)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts = dest.parent / f"{dest.name}.parts"
    parts.mkdir(exist_ok=True)
    with httpx.Client(timeout=60.0) as client:
        total = _total_size(client, endpoint)
    ranges = [(i, s, min(s + chunk, total) - 1) for i, s in enumerate(range(0, total, chunk))]

    # A previously interrupted single-stream download is a valid byte prefix: reuse whole chunks of it.
    if seed_prefix is not None and seed_prefix.exists():
        have = seed_prefix.stat().st_size
        with seed_prefix.open("rb") as fh:
            for i, s, e in ranges:
                if e < have and not (parts / f"{i:05d}.part").exists():
                    fh.seek(s)
                    (parts / f"{i:05d}.part").write_bytes(fh.read(e - s + 1))

    print(f"archive size {total} bytes in {len(ranges)} chunks, {workers} workers", file=sys.stderr)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_chunk, i, s, e, parts, attempts, endpoint) for i, s, e in ranges]
        for fut in cf.as_completed(futures):
            fut.result()
            done += 1
            print(f"  {done}/{len(ranges)} chunks", file=sys.stderr)

    tmp = dest.with_suffix(".assembling")
    with tmp.open("wb") as out:
        for i, _, _ in ranges:
            out.write((parts / f"{i:05d}.part").read_bytes())
    if tmp.stat().st_size != total:
        raise RuntimeError("assembled archive size mismatch")
    tmp.replace(dest)
    for p in parts.iterdir():
        p.unlink()
    parts.rmdir()
    return dest


def publisher_file_index() -> dict:
    """Per-file publisher hashes (first 1000 entries per folder; the API does not paginate further)."""
    with httpx.Client(timeout=60.0) as client:
        meta = client.get(f"{API}/datasets/{DATASET_ID}", params={"version": VERSION}).json()
        folders = client.get(f"{API}/datasets/{DATASET_ID}/folders/{VERSION}").json()
        files: dict[str, dict] = {}
        for folder in folders:
            listing = client.get(
                f"{API}/datasets/{DATASET_ID}/files",
                params={"folder_id": folder["id"], "version": VERSION},
            ).json()
            for entry in listing:
                files[f"{folder['name']}/{entry['filename']}"] = {
                    "sha256": entry["content_details"]["sha256_hash"],
                    "size": entry["size"],
                }
    return {
        "dataset": {
            "id": meta["id"],
            "doi": meta["doi"]["id"],
            "name": meta["name"],
            "version": meta["version"],
            "publish_date": meta.get("publish_date"),
            "licence": (meta.get("data_licence") or {}).get("description", "")[:200],
            "contributors": [
                f"{c.get('first_name', '')} {c.get('last_name', '')}".strip() for c in meta["contributors"]
            ],
            "view_url": meta["links"]["view"],
        },
        "folders": [f["name"] for f in folders],
        "files": files,
        "listing_note": "publisher listing returns at most 1000 entries per folder",
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_extra_sources(config: str = "configs/data.yaml", receipts: str = "docs/evidence/data") -> list[dict]:
    """Download every `extra_sources` archive (skipping ones already present and verified) and write a receipt.
    These publishers throttle many parallel connections, so 4 workers and 8 MiB chunks are used."""
    import yaml

    out = []
    for src in yaml.safe_load(Path(config).read_text()).get("extra_sources", []):
        dest = Path(src["archive"])
        started = None
        if not dest.exists():
            started = datetime.now(UTC).isoformat()
            download_archive(dest, workers=4, endpoint=src["endpoint"], chunk=8 * 1024 * 1024, attempts=12)
        sha = sha256_file(dest)
        if src.get("archive_sha256") and sha != src["archive_sha256"]:
            raise RuntimeError(f"{src['id']}: SHA-256 {sha} does not match the publisher's {src['archive_sha256']}")
        if src.get("archive_md5"):
            md5 = hashlib.md5(dest.read_bytes()).hexdigest()  # noqa: S324 - the publisher's checksum is MD5
            if md5 != src["archive_md5"]:
                raise RuntimeError(f"{src['id']}: MD5 {md5} does not match the publisher's {src['archive_md5']}")
        receipt = {"source_id": src["id"], "source_endpoint": src["endpoint"], "archive_path": str(dest),
                   "archive_sha256": sha, "archive_bytes": dest.stat().st_size, "license": src["license"],
                   "doi": src["doi"], "verified_against": "archive_sha256" if src.get("archive_sha256") else "archive_md5",
                   **({"retrieval_started_at": started, "retrieval_finished_at": datetime.now(UTC).isoformat()}
                      if started else {"verified_at": datetime.now(UTC).isoformat(),
                                       "archive_mtime": datetime.fromtimestamp(dest.stat().st_mtime, UTC).isoformat()})}
        path = Path(receipts) / f"download_receipt_{src['id'].split(':')[0].lower()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        out.append(receipt)
    return out


def _box_counts(archive: Path) -> dict[str, int]:
    import collections
    import zipfile

    import yaml

    with zipfile.ZipFile(archive) as zf:
        names = yaml.safe_load(zf.read("data.yaml"))["names"]
        counts: collections.Counter[str] = collections.Counter()
        for n in zf.namelist():
            if n.endswith(".txt") and "/labels/" in n:
                counts.update(names[int(line.split()[0])] for line in zf.read(n).decode().splitlines() if line.strip())
    return dict(sorted(counts.items()))


def fetch_damage_sources(config: str = "configs/damage.yaml", receipts: str = "docs/evidence/data") -> list[dict]:
    """Download the damage locator's Roboflow Universe exports (YOLO format) with ROBOFLOW_API_KEY.
    Archives already present are only verified. The key is sent as a query parameter, so request errors are
    re-raised without the URL."""
    import os

    import yaml

    out = []
    for src in yaml.safe_load(Path(config).read_text())["sources"]:
        dest = Path(src["archive"])
        ws, proj, _, ver = src["url"].removeprefix("https://universe.roboflow.com/").split("/")
        path = Path(receipts) / f"download_receipt_damage_{proj}.json"
        if not dest.exists():
            key = os.environ.get("ROBOFLOW_API_KEY", "")
            if not key:
                raise RuntimeError(f"{src['id']}: set ROBOFLOW_API_KEY in .env to download {src['url']}")
            try:
                with httpx.Client(timeout=120, follow_redirects=True) as client:
                    meta = client.get(f"https://api.roboflow.com/{ws}/{proj}/{ver}/yolov8", params={"api_key": key})
                    meta.raise_for_status()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with client.stream("GET", meta.json()["export"]["link"]) as r:
                        r.raise_for_status()
                        with dest.open("wb") as f:
                            for block in r.iter_bytes(1 << 20):
                                f.write(block)
            except httpx.HTTPStatusError as e:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"{src['id']}: Roboflow answered HTTP {e.response.status_code}") from None
            except httpx.HTTPError as e:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"{src['id']}: download failed ({type(e).__name__})") from None
        sha = sha256_file(dest)
        if sha != src["archive_sha256"]:
            raise RuntimeError(f"{src['id']}: SHA-256 {sha} does not match the recorded {src['archive_sha256']} "
                               "(Roboflow regenerated the export?)")
        if path.exists() and json.loads(path.read_text()).get("archive_sha256") == sha:
            out.append(json.loads(path.read_text()))  # keep the original retrieval record
            continue
        receipt = {"archive_bytes": dest.stat().st_size, "archive_path": str(dest), "archive_sha256": sha,
                   "classes": _box_counts(dest), "format": "yolov8", "license": src["license"],
                   "retrieved_at": datetime.now(UTC).isoformat(), "source": f"roboflow:{ws}/{proj}/{ver}",
                   "url": src["url"]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        out.append(receipt)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch the D1 archive from Mendeley Data")
    ap.add_argument("--dest", default="data/raw/archive/bn7ch8tvyp-1.zip")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed-prefix", default=None, help="partial single-stream download to reuse")
    ap.add_argument("--receipt", default="docs/evidence/data/download_receipt.json")
    ap.add_argument("--extra", action="store_true", help="fetch the extra public sources in configs/data.yaml")
    ap.add_argument("--damage", action="store_true", help="fetch the damage locator exports in configs/damage.yaml")
    args = ap.parse_args(argv)
    if args.extra:
        print(json.dumps([{k: r[k] for k in ("source_id", "archive_sha256", "archive_bytes")}
                          for r in fetch_extra_sources()]))
        return 0
    if args.damage:
        print(json.dumps([{k: r[k] for k in ("source", "archive_sha256", "archive_bytes")}
                          for r in fetch_damage_sources()]))
        return 0

    started = datetime.now(UTC).isoformat()
    dest = download_archive(Path(args.dest), args.workers, Path(args.seed_prefix) if args.seed_prefix else None)
    index = publisher_file_index()
    receipt = {
        "source_endpoint": ZIP_ENDPOINT,
        "retrieval_started_at": started,
        "retrieval_finished_at": datetime.now(UTC).isoformat(),
        "archive_path": str(dest),
        "archive_sha256": sha256_file(dest),
        "archive_bytes": dest.stat().st_size,
        "publisher": index,
    }
    out = Path(args.receipt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({k: receipt[k] for k in ("archive_sha256", "archive_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

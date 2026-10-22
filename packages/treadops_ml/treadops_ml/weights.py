"""Resolve pretrained checkpoints from models/pretrained and refuse any whose SHA-256 is not pinned."""

from __future__ import annotations

from pathlib import Path

from .common import load_yaml, repo_path, sha256_file

LOCK = "configs/weights.lock.yaml"


class WeightsMismatch(RuntimeError):
    pass


def pretrained(name: str) -> Path:
    lock = load_yaml(LOCK)
    entry = lock["weights"].get(name)
    if entry is None:
        raise WeightsMismatch(f"{name} is not listed in {LOCK}")
    path = repo_path(lock["weights_dir"]) / name
    if not path.exists():
        # Ultralytics downloads into the CWD; fetch into the pinned directory, then verify.
        import os

        from ultralytics.utils.downloads import attempt_download_asset

        cwd = os.getcwd()
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chdir(path.parent)
        try:
            attempt_download_asset(name)
        finally:
            os.chdir(cwd)
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise WeightsMismatch(f"{name}: expected sha256 {entry['sha256']}, got {actual}")
    return path


def pinned_sha(name: str) -> str:
    return load_yaml(LOCK)["weights"][name]["sha256"]

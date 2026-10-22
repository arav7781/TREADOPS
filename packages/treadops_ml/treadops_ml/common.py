"""Hashing, config loading and environment capture shared by every pipeline stage."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with repo_path(path).open() as fh:
        return yaml.safe_load(fh)


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with repo_path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> Path:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(repo_path(path).read_text())


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except Exception:
        return None


def git_state() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else None}


def lock_hash() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    return sha256_file(lock) if lock.exists() else None


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception:  # torch is optional for API-only processes
        pass
    if platform.system() == "Darwin":
        try:
            info["cpu_brand"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            info["ram_bytes"] = int(
                subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
            )
        except Exception:
            pass
    return info


def package_versions(names: tuple[str, ...] = ("ultralytics", "torch", "torchvision", "onnxruntime", "numpy")) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def resolve_device(requested: str | int | None) -> str:
    """auto -> TRAIN_DEVICE env, else cuda:0 / mps / cpu."""
    req = str(requested) if requested is not None else "auto"
    if req != "auto":
        return req
    env = os.environ.get("TRAIN_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

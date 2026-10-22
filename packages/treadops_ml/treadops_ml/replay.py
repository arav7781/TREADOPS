"""Deterministic operational replay schedules (simulated metadata over real public images).

Events follow the spec's event contract. Timestamps come from the schedule only (independent of labels);
synthetic IDs and the seed never reach the classifier; labels are preserved even when a corruption makes a
photo unjudgeable. Every event is `training_eligible: false`.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .common import load_yaml, sha256_bytes
from .scenarios import transform_hash

SCENARIOS = ("baseline", "low_light", "camera_blur", "class_prior_shift", "burst_and_outage")


@dataclass(frozen=True)
class Parent:
    image_id: str
    label: str
    group_id: str


def _order(seed: int, key: str) -> str:
    return sha256_bytes(f"{seed}:{key}".encode())


def generate(scenario: str, seed: int, count: int, rate: float, pool: list[Parent],
             start: datetime | None = None, cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    if not pool:
        raise ValueError("empty replay pool")
    cfg = cfg or load_yaml("configs/replay.yaml")
    sc = cfg["scenarios"][scenario]
    rng = random.Random(seed)
    ordered = sorted(pool, key=lambda p: _order(seed, p.image_id))
    defective = [p for p in ordered if p.label == "defective"]
    good = [p for p in ordered if p.label != "defective"]
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    events, t = [], 0.0
    burst_rate = sc.get("burst_rate_per_s", rate)
    for seq in range(1, count + 1):
        if scenario == "class_prior_shift" and seq > sc["start_after"]:
            src = defective if rng.random() < sc["defective_fraction"] else good
            parent = src[(seq * 7919) % len(src)]
        else:
            parent = ordered[(seq - 1) % len(ordered)]
        transform: dict[str, Any] = {}
        if scenario == "low_light" and seq > sc["start_after"] and rng.random() < sc["fraction"]:
            transform = {"brightness_factor": sc["brightness_factor"]}
        if scenario == "camera_blur" and seq > sc["start_after"] and rng.random() < sc["fraction"]:
            transform = {"gaussian_sigma": sc["gaussian_sigma"]}
        in_burst = scenario == "burst_and_outage" and seq <= sc["burst_events"]
        t += 1.0 / (burst_rate if in_burst else rate)
        events.append({
            "event_id": str(uuid.UUID(_order(seed, f"{scenario}:{seq}")[:32])),
            "scenario_id": f"{scenario}:{sc['version']}",
            "seed": seed,
            "sequence": seq,
            "captured_at": (start + timedelta(seconds=t)).isoformat(),
            "offset_s": round(t, 4),
            "image_id": None,  # assigned when the derivative asset is created
            "parent_image_id": parent.image_id,
            "label": parent.label,
            "line_id": rng.choice(cfg["lines"]),
            "camera_id": rng.choice(cfg["cameras"]),
            "metadata_origin": "simulated",
            "transform": transform,
            "transform_hash": transform_hash(transform),
            "training_eligible": False,
            "duplicate_of": None,
        })
    if scenario == "burst_and_outage":
        dup_rng = random.Random(seed + 1)
        for e in list(events):
            if dup_rng.random() < sc["duplicate_fraction"]:
                events.append({**e, "duplicate_of": e["event_id"]})
        events.sort(key=lambda e: (e["sequence"], e["duplicate_of"] is not None))
    return events

"""Replay job: submit simulated events through the real acceptance path at a measured rate.

Duplicate events reuse their original Idempotency-Key, so the API must return the same inspection. Queue
backpressure (429) pauses the replay instead of dropping events. Cancellation stops new events immediately.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from typing import cast

import numpy as np
from sqlalchemy import select

from treadops_api import jobs
from treadops_api.db import must_get, new_session, session_scope
from treadops_api.errors import ApiError
from treadops_api.models import ImageAsset, Inspection, Job
from treadops_api.services import inspections as svc
from treadops_api.storage import get_storage
from treadops_ml.common import load_yaml
from treadops_ml.preprocess import load_canonical
from treadops_ml.replay import Parent, generate
from treadops_ml.scenarios import derivative_bytes

from ..tasks import JobContext, Permanent, handler


@handler("replay")
def run(ctx: JobContext) -> dict:
    p = ctx.payload
    cfg = load_yaml("configs/replay.yaml")
    storage = get_storage()
    db = new_session()
    try:
        parents = db.execute(select(ImageAsset).where(ImageAsset.image_origin == "public_dataset",
                                                      ImageAsset.role.in_(cfg["pool_roles"]))).scalars().all()
        if not parents:
            raise Permanent("empty_pool", "no public images registered in the replay pool roles")
        by_id = {str(a.id): a for a in parents}
        events = generate(p["scenario"], int(p["seed"]), int(p["count"]), float(p["rate"]),
                          [Parent(str(a.id), cast(str, a.label), cast(str, a.group_id)) for a in parents],
                          cfg=cfg)
        canon_cache: dict[uuid.UUID, np.ndarray] = {}
        stats = {"submitted": 0, "accepted": 0, "deduplicated": 0, "rejected": 0, "backpressure_waits": 0,
                 "resumed_skipped": 0}
        accepted_ids = []
        # Resumable: a retried replay skips events whose idempotency keys already produced inspections.
        done_keys = set(db.execute(select(Inspection.idempotency_key).where(Inspection.replay_job_id == ctx.job_id))
                        .scalars())
        pending = [e for e in events if f"replay:{ctx.job_id}:{e['duplicate_of'] or e['event_id']}" not in done_keys
                   or e["duplicate_of"]]
        stats["resumed_skipped"] = len(events) - len(pending)
        t0 = time.monotonic() - (pending[0]["offset_s"] if pending else 0.0)
        for i, ev in enumerate(events):
            if ev not in pending:
                continue
            target = t0 + ev["offset_s"]
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, 5.0))
            if i % 10 == 0:
                ctx.heartbeat(i / len(events), f"event {i}/{len(events)}")  # raises JobCancelled on cancel
            parent = by_id[ev["parent_image_id"]]
            if parent.id not in canon_cache:
                canon_cache[parent.id] = load_canonical(storage.get(cast(str, parent.canonical_key)))
            raw = derivative_bytes(canon_cache[parent.id], ev["transform"])
            key = f"replay:{ctx.job_id}:{ev['duplicate_of'] or ev['event_id']}"
            while True:
                try:
                    insp, created = svc.accept(
                        db, storage, actor=None, actor_id=ctx.created_by, raw=raw, idempotency_key=key,
                        source="replay", metadata_origin="simulated", line_id=ev["line_id"], camera_id=ev["camera_id"],
                        parent_image_id=parent.id, transform=ev["transform"], transform_hash=ev["transform_hash"],
                        scenario_id=ev["scenario_id"], replay_job_id=ctx.job_id, replay_sequence=ev["sequence"])
                    break
                except ApiError as exc:
                    db.rollback()
                    if exc.status == 429:
                        stats["backpressure_waits"] += 1
                        ctx.heartbeat(i / len(events), "backpressure: waiting for queue to drain")
                        time.sleep(exc.retry_after or 5)
                        continue
                    if exc.status == 503 and exc.code == "model_unavailable":
                        raise Permanent("model_unavailable", exc.message) from exc
                    stats["rejected"] += 1
                    insp, created = None, False
                    break
            stats["submitted"] += 1
            if insp is not None:
                ev["image_id"] = str(insp.image_id)
                ev["inspection_id"] = str(insp.id)
                if created:
                    stats["accepted"] += 1
                    accepted_ids.append(str(insp.id))
                else:
                    stats["deduplicated"] += 1
        elapsed = time.monotonic() - t0
    finally:
        db.close()
    buf = io.BytesIO("".join(json.dumps(e, sort_keys=True) + "\n" for e in events).encode())
    events_key = f"replays/{ctx.job_id}/events.jsonl"
    storage.put(events_key, buf.getvalue(), "application/x-ndjson")
    result = {**stats, "scenario": p["scenario"], "seed": p["seed"], "requested_rate_per_s": p["rate"],
              "measured_rate_per_s": round(stats["submitted"] / elapsed, 3) if elapsed else None,
              "elapsed_s": round(elapsed, 1), "events_key": events_key, "metadata_origin": "simulated",
              "operator_drill": "burst_and_outage: pause/restart the inference worker or broker manually during the run"
              if p["scenario"] == "burst_and_outage" else None}
    with session_scope() as s:
        jobs.succeed(s, must_get(s, Job, ctx.job_id, populate_existing=True), result)
    return result

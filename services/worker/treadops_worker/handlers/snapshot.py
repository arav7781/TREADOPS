from __future__ import annotations

from treadops_api import jobs
from treadops_api.db import must_get, session_scope
from treadops_api.models import Job
from treadops_api.services.datasets import build_snapshot
from treadops_api.storage import get_storage

from ..tasks import JobContext, handler


@handler("snapshot")
def run(ctx: JobContext) -> dict:
    with session_scope() as db:
        snap, created = build_snapshot(db, get_storage(), actor_id=ctx.created_by, job_id=ctx.job_id,
                                       description=ctx.payload.get("description"),
                                       include_verified_uploads=ctx.payload.get("include_verified_uploads", True))
        result = {"snapshot_id": str(snap.id), "manifest_sha256": snap.manifest_sha256, "created": created,
                  "counts": snap.counts}
        jobs.succeed(db, must_get(db, Job, ctx.job_id, populate_existing=True), result)
    return result

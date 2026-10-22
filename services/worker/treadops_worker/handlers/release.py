"""Promotion / rollback execution and MLflow alias reconciliation.

1 stage   - fetch the bundle by SHA, verify every file hash, class map, task and gates
2 warm    - load in this ops-worker process (isolated from the serving worker) and replay the golden batch
3 CAS     - lock the pointer row, require expected_generation, write deployment + audit + reconcile job
4 reconcile (separate job) - move the MLflow `champion` alias; retried while MLflow is unavailable
Any failure leaves the previous serving pointer untouched.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime

import numpy as np

from treadops_api import audit, jobs
from treadops_api.db import session_scope
from treadops_api.errors import ApiError
from treadops_api.models import Job, ModelVersion
from treadops_api.services.releases import switch_pointer
from treadops_api.settings import get_settings
from treadops_ml.preprocess import tensor_from_canonical

from ..runtime import WORKER_ID, bundles
from ..tasks import JobContext, Permanent, Transient, handler, on_fail


def warm_up(mv: ModelVersion) -> dict:
    t0 = time.perf_counter()
    bundle = bundles().get(mv.bundle_sha256, mv.bundle_uri, device="cpu")
    load_ms = (time.perf_counter() - t0) * 1000
    if mv.gate_status != "passed" or bundle.manifest["evaluation"]["gate_status"] != "passed":
        raise Permanent("gate_not_passed", "bundle gate report is not passed")
    golden = json.loads((bundle.root / "golden.json").read_text())
    errs, lat, mismatches = [], [], []
    for item in golden["items"]:
        x = tensor_from_canonical((bundle.root / item["file"]).read_bytes(), bundle.preprocess)
        t1 = time.perf_counter()
        raw = bundle.raw_probs(x)[0]
        lat.append((time.perf_counter() - t1) * 1000)
        errs.append(float(np.abs(raw - np.asarray(item["raw_probs"])).max()))
        p = float(bundle.p_defective(raw)[0])
        if bundle.policy.decide(p)[0] != item["triage"]:
            mismatches.append(item["image_id"])
    ok = max(errs) <= golden["tolerance_abs"] and not mismatches
    return {"status": "ready" if ok else "failed", "golden_images": len(golden["items"]),
            "max_abs_error": max(errs), "tolerance": golden["tolerance_abs"], "triage_mismatches": mismatches,
            "load_ms": round(load_ms, 1), "warm_latency_ms": [round(v, 2) for v in lat], "worker": WORKER_ID,
            "at": datetime.now(UTC).isoformat()}


def _release(ctx: JobContext, action: str) -> dict:
    p = ctx.payload
    with session_scope() as db:
        mv = db.get(ModelVersion, uuid.UUID(p["model_version_id"]))
        if mv is None:
            raise Permanent("model_version_missing", "model version not found")
        db.expunge(mv)
    ctx.heartbeat(0.2, "staging + warm-up")
    readiness = warm_up(mv)
    if readiness["status"] != "ready":
        raise Permanent("warmup_failed", f"golden batch mismatch: max error {readiness['max_abs_error']:.2e}, "
                                         f"triage mismatches {readiness['triage_mismatches']}")
    ctx.heartbeat(0.7, "switching pointer")
    with session_scope() as db:
        mv = db.get(ModelVersion, mv.id)
        try:
            dep = switch_pointer(db, mv=mv, expected_generation=int(p["expected_generation"]), action=action,
                                 approved_by=uuid.UUID(p["approved_by"]), reason=p["reason"], readiness=readiness,
                                 correlation_id=ctx.correlation_id)
        except ApiError as exc:
            raise Permanent(exc.code, exc.message) from exc
        job = db.get(Job, ctx.job_id, populate_existing=True)
        result = {"deployment_id": str(dep.id), "generation": dep.generation, "bundle_sha256": dep.bundle_sha256,
                  "readiness": readiness}
        jobs.succeed(db, job, result)
    return result


@handler("promotion")
def promote(ctx: JobContext) -> dict:
    return _release(ctx, "promote")


@handler("rollback")
def rollback(ctx: JobContext) -> dict:
    return _release(ctx, "rollback")


def _release_failed(db, job: Job, status: str) -> None:
    audit.record(db, action=f"release.{job.kind}.failed", resource_type="model_version",
                 resource_id=job.payload.get("model_version_id"), correlation_id=job.correlation_id,
                 outcome="failure", payload={"error_code": job.error_code, "error": job.error_message,
                                             "expected_generation": job.payload.get("expected_generation"),
                                             "pointer": "unchanged"})


on_fail("promotion")(_release_failed)
on_fail("rollback")(_release_failed)


@handler("registry_reconcile")
def reconcile(ctx: JobContext) -> dict:
    p = ctx.payload
    uri = get_settings().mlflow_tracking_uri
    if not p.get("registry_version") or not uri:
        result = {"status": "skipped", "reason": "model version not registered in MLflow"}
    else:
        try:
            from mlflow import MlflowClient

            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
            c = MlflowClient(tracking_uri=uri)
            c.set_registered_model_alias(p["registry_name"], "champion", p["registry_version"])
            c.set_model_version_tag(p["registry_name"], p["registry_version"], "deployment_generation",
                                    str(p["generation"]))
            result = {"status": "reconciled", "alias": "champion", "version": p["registry_version"]}
        except Exception as exc:
            raise Transient("registry_unavailable", f"{exc.__class__.__name__}: {str(exc)[:200]}") from exc
    with session_scope() as db:
        jobs.succeed(db, db.get(Job, ctx.job_id, populate_existing=True), result)
    return result

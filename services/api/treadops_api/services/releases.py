"""ReleaseService: bundle import, promotion/rollback requests and the compare-and-swap pointer switch.

The API validates and enqueues; the ops worker stages, warms and switches (see treadops_worker.handlers.release).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from treadops_ml.bundle import verify_bundle

from .. import audit, jobs
from ..errors import ApiError
from ..models import Deployment, EnvironmentPointer, ModelVersion
from ..storage import Storage
from .inspections import ENV, active_deployment

JOB_KIND = {"promote": "promotion", "rollback": "rollback"}
STAGE_FOR_GATE = {"passed": "validated", "failed": "rejected", "insufficient_evidence": "candidate"}


def import_bundle(db: Session, storage: Storage, bundle_dir: Path, *, training_run_id: uuid.UUID | None = None,
                  actor=None) -> ModelVersion:
    """Upload a verified bundle to bundles/<sha>/ and register it (idempotent by bundle SHA)."""
    manifest = verify_bundle(bundle_dir)
    sha = manifest["bundle_sha256"]
    existing = db.execute(select(ModelVersion).where(ModelVersion.bundle_sha256 == sha)).scalar_one_or_none()
    if existing is not None:
        return existing
    prefix = f"bundles/{sha}"
    for rel in [*manifest["files"], "manifest.json"]:
        storage.put(f"{prefix}/{rel}", (bundle_dir / rel).read_bytes())
    gate = json.loads((bundle_dir / "gate_report.json").read_text())
    metrics = json.loads((bundle_dir / "metrics.json").read_text())
    mv = ModelVersion(
        registry_name=manifest["registry"]["name"], registry_version=manifest["registry"].get("version"),
        bundle_sha256=sha, bundle_uri=f"s3://{storage.bucket}/{prefix}", model_family=manifest["model_family"],
        stage=STAGE_FOR_GATE.get(gate["status"], "candidate"), gate_status=gate["status"], gate_report=gate,
        metrics=metrics, manifest=manifest, training_run_id=training_run_id,
        mlflow_run_id=manifest["training"].get("mlflow_run_id"),
    )
    db.add(mv)
    db.flush()
    audit.record(db, actor=actor, action="model.import", resource_type="model_version", resource_id=mv.id,
                 payload={"bundle_sha256": sha, "gate_status": gate["status"],
                          "registry_version": mv.registry_version})
    return mv


def _version(db: Session, model_version_id: uuid.UUID) -> ModelVersion:
    mv = db.get(ModelVersion, model_version_id)
    if mv is None:
        raise ApiError(404, "not_found", "Model version not found.")
    return mv


def request_release(db: Session, *, actor, action: str, model_version_id: uuid.UUID, expected_generation: int,
                    reason: str, correlation_id: str) -> dict[str, Any]:
    if not reason or len(reason.strip()) < 8:
        raise ApiError(422, "reason_required", "An approval reason of at least 8 characters is required.")
    mv = _version(db, model_version_id)
    ptr, dep = active_deployment(db)
    if ptr.generation != expected_generation:
        raise ApiError(409, "generation_conflict", "The serving deployment changed; reload and retry.",
                       {"expected_generation": expected_generation, "current_generation": ptr.generation})
    if dep is not None and dep.model_version_id == mv.id:
        raise ApiError(409, "already_active", "This model version is already serving.")
    if action == "promote":
        if mv.gate_status != "passed":
            audit.record(db, actor=actor, action="release.blocked", resource_type="model_version", resource_id=mv.id,
                         correlation_id=correlation_id, outcome="denied",
                         payload={"gate_status": mv.gate_status, "reason": reason})
            db.commit()
            raise ApiError(409, "gate_not_passed", f"Gates are {mv.gate_status}; a failing or insufficient bundle "
                           "cannot be promoted.", {"gate_status": mv.gate_status})
    else:  # rollback: only to a version that has served before and still passes its gates
        served = db.execute(select(Deployment.id).where(Deployment.model_version_id == mv.id)).first()
        if served is None:
            raise ApiError(409, "not_previously_deployed", "Rollback target must be a previously deployed version.")
        if mv.gate_status != "passed":
            raise ApiError(409, "gate_not_passed", "Rollback target no longer passes its gates.")
    job = jobs.create_job(
        db, kind=JOB_KIND[action], dedupe_key=f"{action}:{mv.id}:gen{expected_generation}:{uuid.uuid4().hex[:8]}",
        payload={"model_version_id": str(mv.id), "expected_generation": expected_generation, "reason": reason.strip(),
                 "approved_by": str(actor.id)},
        actor_id=actor.id, correlation_id=correlation_id, resource_type="model_version", resource_id=str(mv.id),
        max_attempts=1,
    )
    audit.record(db, actor=actor, action=f"release.{action}.requested", resource_type="model_version",
                 resource_id=mv.id, correlation_id=correlation_id,
                 payload={"expected_generation": expected_generation, "reason": reason.strip(), "job_id": str(job.id),
                          "bundle_sha256": mv.bundle_sha256})
    db.commit()
    return {"job_id": str(job.id), "status": "queued", "action": action, "model_version_id": str(mv.id),
            "expected_generation": expected_generation}


def switch_pointer(db: Session, *, mv: ModelVersion, expected_generation: int, action: str, approved_by,
                   reason: str, readiness: dict, correlation_id: str | None) -> Deployment:
    """Compare-and-swap inside the caller's transaction. Raises ApiError(409) when the generation moved."""
    ptr = db.execute(select(EnvironmentPointer).where(EnvironmentPointer.env == ENV).with_for_update()).scalar_one()
    if ptr.generation != expected_generation:
        raise ApiError(409, "generation_conflict", "Serving deployment changed during the release.",
                       {"expected_generation": expected_generation, "current_generation": ptr.generation})
    previous = db.get(Deployment, ptr.active_deployment_id) if ptr.active_deployment_id else None
    if previous is not None:
        previous.state = "superseded"
        prev_mv = db.get(ModelVersion, previous.model_version_id)
        if prev_mv.id != mv.id:
            prev_mv.stage = "rolled_back" if action == "rollback" else "retired"
    manifest = mv.manifest
    dep = Deployment(
        model_version_id=mv.id, bundle_sha256=mv.bundle_sha256,
        preprocess_version=manifest["serving"]["preprocess_version"],
        calibration_version=manifest["serving"]["calibration_version"],
        policy_version=manifest["serving"]["policy_version"], generation=expected_generation + 1,
        action="bootstrap" if previous is None else action, state="active",
        previous_deployment_id=previous.id if previous else None, approved_by=approved_by, reason=reason,
        readiness=readiness,
    )
    db.add(dep)
    db.flush()
    ptr.active_deployment_id = dep.id
    ptr.generation = dep.generation
    mv.stage = "active"
    audit.record(db, actor=None, action=f"release.{action}.activated", resource_type="deployment",
                 resource_id=dep.id, correlation_id=correlation_id,
                 payload={"generation_before": expected_generation, "generation_after": dep.generation,
                          "bundle_sha256": mv.bundle_sha256, "previous_deployment_id": str(previous.id)
                          if previous else None, "approved_by": str(approved_by), "reason": reason})
    jobs.create_job(db, kind="registry_reconcile", dedupe_key=f"registry:champion:gen{dep.generation}",
                    payload={"model_version_id": str(mv.id), "registry_name": mv.registry_name,
                             "registry_version": mv.registry_version, "generation": dep.generation},
                    correlation_id=correlation_id, resource_type="deployment", resource_id=str(dep.id),
                    max_attempts=8)
    return dep


def model_view(mv: ModelVersion, active_id: uuid.UUID | None = None, deployments: list | None = None) -> dict:
    m = mv.metrics or {}
    gate = mv.gate_report or {}
    test = (m.get("locked_test") or {}).get("binary", {})
    return {
        "id": str(mv.id), "registry_name": mv.registry_name, "registry_version": mv.registry_version,
        "bundle_sha256": mv.bundle_sha256, "model_family": mv.model_family, "stage": mv.stage,
        "gate_status": mv.gate_status, "is_active": active_id == mv.id, "created_at": mv.created_at.isoformat(),
        "mlflow_run_id": mv.mlflow_run_id,
        "headline": {
            "gate_recall": next((t.get("value") for t in gate.get("quality_targets", []) if t["id"] == "defect_recall"), None),
            "gate_specificity": next((t.get("value") for t in gate.get("quality_targets", []) if t["id"] == "specificity"), None),
            "test_recall": (test.get("defect_recall") or {}).get("value"),
            "test_specificity": (test.get("specificity") or {}).get("value"),
            "model_p95_ms": ((m.get("latency") or {}).get("model_only") or {}).get("p95_ms"),
        },
        "training": m.get("training"),
        "runtimes": (mv.manifest or {}).get("serving", {}).get("runtimes"),
        "deployments": deployments or [],
    }

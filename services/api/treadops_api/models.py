"""Relational schema. Large bytes (images, reports, embeddings, bundles) live in object storage, never here."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def uid() -> uuid.UUID:
    return uuid.uuid4()


UUIDPK = UUID(as_uuid=True)
TS = DateTime(timezone=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    display_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(300))
    roles: Mapped[list[str]] = mapped_column(ARRAY(String(32)))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_fixture: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class UserSession(Base):
    __tablename__ = "sessions"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(TS)
    revoked_at: Mapped[datetime | None] = mapped_column(TS)


class ImageAsset(Base):
    __tablename__ = "image_assets"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    original_sha256: Mapped[str] = mapped_column(String(64))
    sanitized_sha256: Mapped[str | None] = mapped_column(String(64))
    canonical_sha256: Mapped[str | None] = mapped_column(String(64))
    parent_image_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("image_assets.id"))
    transform: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    transform_hash: Mapped[str | None] = mapped_column(String(64))
    group_id: Mapped[str | None] = mapped_column(String(80))
    source_id: Mapped[str] = mapped_column(String(80))
    label: Mapped[str | None] = mapped_column(String(32))
    label_origin: Mapped[str] = mapped_column(String(32), default="pending")
    outer_split: Mapped[str | None] = mapped_column(String(16))
    role: Mapped[str | None] = mapped_column(String(32))
    object_key: Mapped[str] = mapped_column(String(300))
    canonical_key: Mapped[str | None] = mapped_column(String(300))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    image_origin: Mapped[str] = mapped_column(String(32))
    metadata_origin: Mapped[str] = mapped_column(String(32))
    training_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    ingest_info: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (
        CheckConstraint("image_origin in ('public_dataset','user_upload','synthetic_derivative')", name="ck_image_origin"),
        CheckConstraint("metadata_origin in ('simulated','user_provided','observed')", name="ck_metadata_origin"),
        CheckConstraint("label_origin in ('publisher','inspector_verified','pending')", name="ck_label_origin"),
        # canonical originals are unique by content; derivatives are unique per (parent, transform)
        Index("uq_image_original", "original_sha256", unique=True, postgresql_where=text("parent_image_id IS NULL")),
        Index("uq_image_derivative", "parent_image_id", "transform_hash", unique=True,
              postgresql_where=text("parent_image_id IS NOT NULL")),
        Index("ix_image_group", "group_id"),
        Index("ix_image_parent", "parent_image_id"),
        Index("ix_image_role", "role"),
    )


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    kind: Mapped[str] = mapped_column(String(32))
    queue: Mapped[str] = mapped_column(String(32))
    dedupe_key: Mapped[str] = mapped_column(String(200), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    progress_message: Mapped[str | None] = mapped_column(String(300))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    retryable: Mapped[bool | None] = mapped_column(Boolean)
    lease_owner: Mapped[str | None] = mapped_column(String(200))
    lease_expires_at: Mapped[datetime | None] = mapped_column(TS)
    next_retry_at: Mapped[datetime | None] = mapped_column(TS)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    correlation_id: Mapped[str | None] = mapped_column(String(64))
    resource_type: Mapped[str | None] = mapped_column(String(40))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(TS)
    finished_at: Mapped[datetime | None] = mapped_column(TS)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        CheckConstraint("status in ('queued','running','succeeded','failed','retry_wait',"
                        "'cancellation_requested','cancelled')", name="ck_job_status"),
        Index("ix_jobs_dispatchable", "status", "next_retry_at"),
        Index("ix_jobs_stale_lease", "status", "lease_expires_at"),
        Index("ix_jobs_kind_status", "kind", "status"),
    )


class JobEvent(Base):
    __tablename__ = "job_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), index=True)
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str] = mapped_column(String(32))
    attempt: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text)
    correlation_id: Mapped[str | None] = mapped_column(String(64))
    at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class OutboxEvent(Base):
    __tablename__ = "outbox_events"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    topic: Mapped[str] = mapped_column(String(64))
    event_version: Mapped[int] = mapped_column(Integer, default=1)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    dispatch_state: Mapped[str] = mapped_column(String(16), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    dispatched_at: Mapped[datetime | None] = mapped_column(TS)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (Index("ix_outbox_dispatchable", "dispatch_state", "available_at"),)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    registry_name: Mapped[str] = mapped_column(String(120))
    registry_version: Mapped[str | None] = mapped_column(String(32))
    bundle_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    bundle_uri: Mapped[str] = mapped_column(String(300))
    model_family: Mapped[str] = mapped_column(String(64))
    stage: Mapped[str] = mapped_column(String(24), default="candidate")
    gate_status: Mapped[str] = mapped_column(String(32))
    gate_report: Mapped[dict[str, Any]] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB)
    training_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("training_runs.id"))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (
        CheckConstraint("stage in ('candidate','validated','shadow','approved','active','rolled_back',"
                        "'rejected','retired')", name="ck_model_stage"),
    )


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    model_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_versions.id"))
    bundle_sha256: Mapped[str] = mapped_column(String(64))
    preprocess_version: Mapped[str] = mapped_column(String(16))
    calibration_version: Mapped[str] = mapped_column(String(16))
    policy_version: Mapped[str] = mapped_column(String(16))
    generation: Mapped[int] = mapped_column(Integer, unique=True)
    action: Mapped[str] = mapped_column(String(16))
    state: Mapped[str] = mapped_column(String(16), default="active")
    previous_deployment_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("deployments.id"))
    approved_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    reason: Mapped[str] = mapped_column(Text)
    readiness: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    model_version: Mapped[ModelVersion] = relationship(lazy="joined")

    __table_args__ = (
        CheckConstraint("action in ('bootstrap','promote','rollback')", name="ck_deploy_action"),
        CheckConstraint("state in ('active','superseded')", name="ck_deploy_state"),
    )


class EnvironmentPointer(Base):
    """Singleton row per environment: the authoritative serving pointer."""

    __tablename__ = "environment_pointer"
    env: Mapped[str] = mapped_column(String(32), primary_key=True)
    active_deployment_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("deployments.id"))
    generation: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)


class Inspection(Base):
    __tablename__ = "inspections"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    image_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("image_assets.id"))
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"))
    deployment_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("deployments.id"))
    deployment_generation: Mapped[int] = mapped_column(Integer)
    model_tuple: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    source: Mapped[str] = mapped_column(String(24))
    line_id: Mapped[str | None] = mapped_column(String(40))
    camera_id: Mapped[str | None] = mapped_column(String(40))
    scenario_id: Mapped[str | None] = mapped_column(String(60))
    replay_job_id: Mapped[uuid.UUID | None] = mapped_column(UUIDPK)
    replay_sequence: Mapped[int | None] = mapped_column(Integer)
    metadata_origin: Mapped[str] = mapped_column(String(32))
    tyre_category: Mapped[str | None] = mapped_column(String(32))
    notes: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String(200))
    request_hash: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(TS)

    image: Mapped[ImageAsset] = relationship(lazy="joined")

    __table_args__ = (
        UniqueConstraint("created_by", "idempotency_key", name="uq_inspection_idempotency"),
        Index("ix_inspections_created", "created_at", "id"),
        Index("ix_inspections_status_created", "status", "created_at"),
        Index("ix_inspections_image", "image_id"),
    )


class InspectionStage(Base):
    __tablename__ = "inspection_stages"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    inspection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspections.id"), index=True)
    stage_index: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(120))
    object_key: Mapped[str | None] = mapped_column(String(300))
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    log: Mapped[list[str]] = mapped_column(JSONB, default=list)
    duration_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (UniqueConstraint("inspection_id", "stage_index", name="uq_stage_index"),)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    inspection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspections.id"), unique=True)
    raw_probs: Mapped[dict[str, float]] = mapped_column(JSONB)
    calibrated_probs: Mapped[dict[str, float]] = mapped_column(JSONB)
    p_defective: Mapped[float] = mapped_column(Float)
    triage: Mapped[str] = mapped_column(String(24))
    triage_reason: Mapped[str] = mapped_column(Text)
    binary_defective: Mapped[bool] = mapped_column(Boolean)
    quality_status: Mapped[str] = mapped_column(String(24))
    quality: Mapped[dict[str, Any]] = mapped_column(JSONB)
    timing: Mapped[dict[str, float]] = mapped_column(JSONB)
    lineage: Mapped[dict[str, Any]] = mapped_column(JSONB)
    attribution: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    embedding_key: Mapped[str | None] = mapped_column(String(300))
    second_opinion: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # vision model re-check of auto-clears
    circularity: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (
        CheckConstraint("triage in ('good_candidate','manual_review','defect_flagged')", name="ck_triage"),
        Index("ix_predictions_created", "created_at"),
    )


class ReviewLabel(Base):
    __tablename__ = "review_labels"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    inspection_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inspections.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(24))
    disposition: Mapped[str | None] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text)
    reviewer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(16), default="final")
    verified_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (
        UniqueConstraint("inspection_id", "revision", name="uq_review_revision"),
        CheckConstraint("label in ('good','defective','unjudgeable')", name="ck_review_label"),
        CheckConstraint("disposition is null or disposition in ('retread','repair','material_recovery','reinspect')",
                        name="ck_review_disposition"),
    )


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    manifest_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    dvc_rev: Mapped[str | None] = mapped_column(String(64))
    source_versions: Mapped[dict[str, Any]] = mapped_column(JSONB)
    split_policy: Mapped[dict[str, Any]] = mapped_column(JSONB)
    counts: Mapped[dict[str, Any]] = mapped_column(JSONB)
    artifact_uri: Mapped[str] = mapped_column(String(300))
    parent_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    license_refs: Mapped[list[str]] = mapped_column(JSONB, default=list)
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("jobs.id"))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    snapshot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("dataset_snapshots.id"))
    config_id: Mapped[str] = mapped_column(String(40))
    config_sha256: Mapped[str] = mapped_column(String(64))
    recipe: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    epoch_metrics: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    artifact_uri: Mapped[str | None] = mapped_column(String(300))
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64))
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(UUIDPK)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("jobs.id"))
    error: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(TS)
    finished_at: Mapped[datetime | None] = mapped_column(TS)


class ShadowResult(Base):
    __tablename__ = "shadow_results"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), index=True)
    challenger_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_versions.id"))
    champion_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_versions.id"))
    parent_image_id: Mapped[uuid.UUID] = mapped_column(UUIDPK)
    image_id: Mapped[uuid.UUID] = mapped_column(UUIDPK)
    scenario: Mapped[str] = mapped_column(String(40))
    label: Mapped[str | None] = mapped_column(String(24))
    champion_output: Mapped[dict[str, Any]] = mapped_column(JSONB)
    challenger_output: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (UniqueConstraint("job_id", "image_id", name="uq_shadow_job_image"),)


class MonitoringWindow(Base):
    __tablename__ = "monitoring_windows"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    window_key: Mapped[str] = mapped_column(String(120), unique=True)
    deployment_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("deployments.id"))
    reference_version: Mapped[str] = mapped_column(String(64))
    cohort_sha256: Mapped[str] = mapped_column(String(64))
    n_inspections: Mapped[int] = mapped_column(Integer)
    n_distinct_parents: Mapped[int] = mapped_column(Integer)
    first_at: Mapped[datetime | None] = mapped_column(TS)
    last_at: Mapped[datetime | None] = mapped_column(TS)
    status: Mapped[str] = mapped_column(String(24))
    descriptor_psi: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    prediction_shift: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    embedding_drift: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    labelled_quality: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    system_health: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    breakdown: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    report_key: Mapped[str | None] = mapped_column(String(300))
    evidently: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class DriftAlert(Base):
    __tablename__ = "drift_alerts"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    alert_key: Mapped[str] = mapped_column(String(160))
    state: Mapped[str] = mapped_column(String(24), default="open")
    severity: Mapped[str] = mapped_column(String(16))
    features: Mapped[list[str]] = mapped_column(JSONB)
    reference_version: Mapped[str] = mapped_column(String(64))
    cohort_sha256: Mapped[str] = mapped_column(String(64))
    window_ids: Mapped[list[str]] = mapped_column(JSONB)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB)
    recommendation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        CheckConstraint("state in ('open','acknowledged','investigating','resolved')", name="ck_alert_state"),
        Index("uq_alert_open", "alert_key", unique=True, postgresql_where=text("state <> 'resolved'")),
    )


class DriftAlertReview(Base):
    __tablename__ = "drift_alert_reviews"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    alert_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("drift_alerts.id"), index=True)
    from_state: Mapped[str] = mapped_column(String(24))
    to_state: Mapped[str] = mapped_column(String(24))
    reason: Mapped[str] = mapped_column(Text)
    actor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_id: Mapped[uuid.UUID | None] = mapped_column(UUIDPK)
    actor_email: Mapped[str | None] = mapped_column(String(320))
    action: Mapped[str] = mapped_column(String(64))
    resource_type: Mapped[str] = mapped_column(String(40))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    correlation_id: Mapped[str | None] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16), default="success")
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (Index("ix_audit_at", "at"), Index("ix_audit_resource", "resource_type", "resource_id"))


class Lease(Base):
    """Distributed leases: the single GPU/training slot, the scheduler, the monitoring runner."""

    __tablename__ = "leases"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(200))
    acquired_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    heartbeat_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(TS)


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"
    worker_name: Mapped[str] = mapped_column(String(200), primary_key=True)
    queue: Mapped[str] = mapped_column(String(32))
    last_seen_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    ready_bundles: Mapped[list[str]] = mapped_column(JSONB, default=list)
    paused: Mapped[bool] = mapped_column(Boolean, default=False)
    info: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


# ---- Copilot / LLMOps -------------------------------------------------------------------------------------


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    inspection_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("inspections.id"))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    attachments: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    suggestions: Mapped[list[str]] = mapped_column(JSONB, default=list)
    trace_id: Mapped[uuid.UUID | None] = mapped_column(UUIDPK)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class LLMTrace(Base):
    __tablename__ = "llm_traces"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(24))
    llm_mode: Mapped[str] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(120))
    prompt_version: Mapped[str] = mapped_column(String(32))
    prompt_sha256: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    n_llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    n_tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    tools_used: Mapped[list[str]] = mapped_column(JSONB, default=list)
    guardrail_flags: Mapped[list[str]] = mapped_column(JSONB, default=list)
    grounding: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow, index=True)


class LLMSpan(Base):
    __tablename__ = "llm_spans"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    trace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("llm_traces.id"), index=True)
    parent_span_id: Mapped[uuid.UUID | None] = mapped_column(UUIDPK)
    name: Mapped[str] = mapped_column(String(80))
    kind: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="ok")
    start_offset_ms: Mapped[float] = mapped_column(Float)
    duration_ms: Mapped[float] = mapped_column(Float)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class GuardrailEvent(Base):
    __tablename__ = "guardrail_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("llm_traces.id"), index=True)
    stage: Mapped[str] = mapped_column(String(16))
    rule: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(16))
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class ChatFeedback(Base):
    __tablename__ = "chat_feedback"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_messages.id"))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    score: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)

    __table_args__ = (
        UniqueConstraint("message_id", "user_id", name="uq_feedback_once"),
        CheckConstraint("score in (-1, 1)", name="ck_feedback_score"),
    )


class Camera(Base):
    """A line camera (RTSP or other FFmpeg-readable stream). The stream URL can carry credentials, so it is stored
    encrypted and only ever returned masked."""

    __tablename__ = "cameras"
    id: Mapped[uuid.UUID] = mapped_column(UUIDPK, primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    code: Mapped[str] = mapped_column(String(64), unique=True)        # written to inspections.camera_id
    line_id: Mapped[str | None] = mapped_column(String(64))
    stream_url_enc: Mapped[str] = mapped_column(Text)
    stream_url_masked: Mapped[str] = mapped_column(String(512))
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)   # test streams label inspections "simulated"
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    capture_mode: Mapped[str] = mapped_column(String(16), default="on_change")
    interval_s: Mapped[float] = mapped_column(Float, default=5.0)
    change_threshold: Mapped[float] = mapped_column(Float, default=0.04)
    min_sharpness: Mapped[float] = mapped_column(Float, default=15.0)
    state: Mapped[str] = mapped_column(String(16), default="idle")
    last_error: Mapped[str | None] = mapped_column(Text)
    last_frame_at: Mapped[datetime | None] = mapped_column(TS)
    last_capture_at: Mapped[datetime | None] = mapped_column(TS)
    fps: Mapped[float | None] = mapped_column(Float)
    frame_width: Mapped[int | None] = mapped_column(Integer)
    frame_height: Mapped[int | None] = mapped_column(Integer)
    frames_read: Mapped[int] = mapped_column(BigInteger, default=0)
    captures: Mapped[int] = mapped_column(Integer, default=0)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUIDPK, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        CheckConstraint("capture_mode in ('interval','on_change')", name="ck_camera_capture_mode"),
        CheckConstraint("state in ('idle','connecting','streaming','error','disabled')", name="ck_camera_state"),
        CheckConstraint("interval_s >= 1 and interval_s <= 3600", name="ck_camera_interval"),
    )


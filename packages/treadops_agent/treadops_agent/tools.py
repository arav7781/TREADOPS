"""Copilot tools. Each is an OpenAI function schema plus an implementation backed by the real services.

`classify_tyre_image` never classifies inside the agent: it submits (or waits for) a real inspection job, so
the YOLO bundle pinned by the active deployment produces the decision and full lineage is stored.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select

from treadops_api import circularity
from treadops_api.db import new_session
from treadops_api.models import (
    DriftAlert,
    Inspection,
    InspectionStage,
    MonitoringWindow,
    Prediction,
    ReviewLabel,
)
from treadops_api.services.inspections import active_deployment

from . import kb
from .guardrails import config

TRIAGE_LABEL = {"good_candidate": "Retread candidate", "manual_review": "Expert review",
                "defect_flagged": "Defect flagged"}


@dataclass
class ToolContext:
    actor: Any
    inspection_id: uuid.UUID | None = None          # set once classify_tyre_image has submitted the photo
    image_bytes: bytes | None = None                # the user's photo for this turn (in memory, never persisted raw)
    idempotency_key: str | None = None              # one inspection per turn, even if the tool is called twice
    emit: Callable[[str, dict[str, Any]], None] = lambda *_: None
    results: list[dict[str, Any]] = field(default_factory=list)


def _schema(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": required or [], "additionalProperties": False}}}


SCHEMAS = [
    _schema("classify_tyre_image",
            "Run the deployed YOLO tyre-casing classifier on the photo attached to the user's current message. "
            "Returns triage, calibrated p(defective), image quality, the preprocessing stages and the "
            "recommended circular pathway. Only valid when the user attached a photo."),
    _schema("get_inspection", "Fetch the stored result and lineage of an earlier inspection by ID.",
            {"inspection_id": {"type": "string", "description": "inspection UUID"}}, ["inspection_id"]),
    _schema("get_preprocessing_trace", "Explain each preprocessing stage (ingest, canonical, letterbox, tensor, "
            "quality, inference, attribution) of an inspection, with timings.",
            {"inspection_id": {"type": "string"}}, ["inspection_id"]),
    _schema("get_model_status", "Active model version, gate status and measured evidence (gate cohort and "
            "locked test metrics, latency)."),
    _schema("get_drift_status", "Latest monitoring window (descriptor PSI, embedding drift, labelled quality) "
            "and open drift alerts."),
    _schema("estimate_circularity_impact", "Estimate raw material and CO2e avoided for a number of verified "
            "retreads using Michelin's published truck/bus factors.",
            {"verified_retreads": {"type": "integer", "minimum": 1, "maximum": 100000}}, ["verified_retreads"]),
    _schema("search_knowledge", "Search the TreadOps knowledge base (triage policy, visual defect cues, "
            "retreading, photo capture, model/data, monitoring). Returns passages with [KB:id] citations.",
            {"query": {"type": "string"}}, ["query"]),
    _schema("get_review_queue", "How many completed inspections await a human label, and the oldest wait."),
]


def _compact(insp: Inspection, pred: Prediction | None, stages: list[InspectionStage]) -> dict[str, Any]:
    out: dict[str, Any] = {"inspection_id": str(insp.id), "status": insp.status,
                           "model": {"family": insp.model_tuple.get("model_family"),
                                     "registry_version": insp.model_tuple.get("registry_version"),
                                     "bundle_sha256": insp.model_tuple.get("bundle_sha256", "")[:12],
                                     "deployment_generation": insp.deployment_generation},
                           "metadata_origin": insp.metadata_origin}
    if pred is not None:
        circ = pred.circularity or {}
        out.update({
            "triage": pred.triage, "triage_label": TRIAGE_LABEL[pred.triage], "triage_reason": pred.triage_reason,
            "p_defective": round(pred.p_defective, 4),
            "raw_probs": {k: round(v, 4) for k, v in pred.raw_probs.items()},
            "quality_status": pred.quality_status, "quality_reasons": pred.quality.get("reasons", []),
            "descriptors": {k: round(v, 3) for k, v in pred.quality.get("descriptors", {}).items()
                            if k in ("brightness", "contrast", "blur_variance")},
            "recommended_pathway": {"label": circ.get("label"), "action": circ.get("action"),
                                    "estimate_if_verified": circ.get("estimate_if_verified")},
            "attribution": {"method": (pred.attribution or {}).get("method"),
                            "note": "attribution heatmap, not a defect location"},
            "timing_ms": {k: round(v, 1) for k, v in pred.timing.items()},
        })
    out["stages"] = [{"index": s.stage_index, "name": s.name, "title": s.title,
                      "duration_ms": round(s.duration_ms, 1)} for s in stages]
    return out


def _load(db, inspection_id: uuid.UUID) -> dict[str, Any] | None:
    insp = db.get(Inspection, inspection_id)
    if insp is None:
        return None
    pred = db.execute(select(Prediction).where(Prediction.inspection_id == insp.id)).scalar_one_or_none()
    stages = db.execute(select(InspectionStage).where(InspectionStage.inspection_id == insp.id)
                        .order_by(InspectionStage.stage_index)).scalars().all()
    return _compact(insp, pred, stages)


def _submit_photo(ctx: ToolContext) -> dict[str, Any] | None:
    """Create the inspection for this turn's photo through the normal acceptance path (pinned deployment)."""
    from treadops_api.errors import ApiError
    from treadops_api.services.inspections import accept
    from treadops_api.storage import get_storage

    ctx.emit("tool_progress", {"tool": "classify_tyre_image",
                               "message": "submitting the photo to the YOLO inspection pipeline"})
    db = new_session()
    try:
        insp, _ = accept(db, get_storage(), actor=ctx.actor, actor_id=ctx.actor.id, raw=ctx.image_bytes or b"",
                         idempotency_key=ctx.idempotency_key or f"copilot:{uuid.uuid4()}", source="copilot",
                         metadata_origin="user_provided")
        tup = dict(insp.model_tuple)
    except ApiError as exc:
        return {"error": exc.code, "message": exc.message}
    finally:
        db.close()
    ctx.inspection_id = insp.id
    ctx.emit("inspection", {"inspection_id": str(insp.id), "deployment_generation": tup.get("generation"),
                            "model": f"{tup.get('model_family')} v{tup.get('registry_version')}",
                            "bundle_sha256": str(tup.get("bundle_sha256", ""))[:12]})
    return None


def classify_tyre_image(ctx: ToolContext) -> dict[str, Any]:
    if ctx.inspection_id is None:
        if not ctx.image_bytes:
            return {"error": "no_attachment", "message": "No photo is attached to this message. Ask the user to "
                                                          "upload a tyre photo."}
        failure = _submit_photo(ctx)
        if failure is not None:
            return failure
    inspection_id = ctx.inspection_id
    assert inspection_id is not None  # set by _submit_photo above
    wait = float(config()["classify_wait_s"])
    t0 = time.monotonic()
    seen_stages = 0
    ctx.emit("tool_progress", {"tool": "classify_tyre_image",
                               "message": "YOLO inspection job queued; waiting for the worker"})
    while True:
        db = new_session()
        try:
            res = _load(db, inspection_id)
        finally:
            db.close()
        if res is None:
            return {"error": "not_found", "inspection_id": str(ctx.inspection_id)}
        if len(res["stages"]) > seen_stages:
            for st in res["stages"][seen_stages:]:
                ctx.emit("stage", {"tool": "classify_tyre_image", "stage": st["index"], "name": st["name"],
                                   "title": st["title"], "duration_ms": st["duration_ms"],
                                   "message": f"stage {st['index']}: {st['title']} ({st['duration_ms']} ms)"})
            seen_stages = len(res["stages"])
        if res["status"] in ("succeeded", "failed", "cancelled"):
            if res["status"] != "succeeded":
                return {"error": "inspection_failed", "inspection_id": res["inspection_id"],
                        "status": res["status"]}
            return res
        if time.monotonic() - t0 > wait:
            return {"error": "timeout", "inspection_id": res["inspection_id"], "status": res["status"],
                    "message": "The inspection is still queued; results will appear on the inspection page."}
        time.sleep(0.5)


def get_inspection(ctx: ToolContext, inspection_id: str) -> dict[str, Any]:
    try:
        iid = uuid.UUID(inspection_id)
    except ValueError:
        return {"error": "invalid_id"}
    db = new_session()
    try:
        return _load(db, iid) or {"error": "not_found"}
    finally:
        db.close()


def get_preprocessing_trace(ctx: ToolContext, inspection_id: str) -> dict[str, Any]:
    try:
        iid = uuid.UUID(inspection_id)
    except ValueError:
        return {"error": "invalid_id"}
    db = new_session()
    try:
        stages = db.execute(select(InspectionStage).where(InspectionStage.inspection_id == iid)
                            .order_by(InspectionStage.stage_index)).scalars().all()
        return {"inspection_id": inspection_id, "stages": [
            {"index": s.stage_index, "title": s.title, "duration_ms": round(s.duration_ms, 1), "log": s.log}
            for s in stages]} if stages else {"error": "not_found_or_not_processed"}
    finally:
        db.close()


def get_model_status(ctx: ToolContext) -> dict[str, Any]:
    db = new_session()
    try:
        ptr, dep = active_deployment(db)
        db.commit()
        if dep is None:
            return {"state": "untrained", "message": "No trained model available"}
        mv = dep.model_version
        m = mv.metrics or {}
        gate = mv.gate_report or {}
        test = (m.get("locked_test") or {}).get("binary", {})

        def val(x):
            return None if not x else {k: x.get(k) for k in ("value", "numerator", "denominator", "ci95")}

        return {
            "state": "ready", "model_family": mv.model_family, "registry_version": mv.registry_version,
            "bundle_sha256": mv.bundle_sha256[:12], "generation": ptr.generation, "gate_status": mv.gate_status,
            "gate_evidence": gate.get("evidence_label"),
            "quality_targets": [{k: t.get(k) for k in ("id", "target", "value", "support", "status")}
                                for t in gate.get("quality_targets", [])],
            "hard_gates": {g["id"]: g["status"] for g in gate.get("hard_gates", [])},
            "locked_test": {"defect_recall": val(test.get("defect_recall")),
                            "specificity": val(test.get("specificity")), "threshold": test.get("threshold")},
            "latency_model_p95_ms": ((m.get("latency") or {}).get("model_only") or {}).get("p95_ms"),
        }
    finally:
        db.close()


def get_drift_status(ctx: ToolContext) -> dict[str, Any]:
    db = new_session()
    try:
        w = db.execute(select(MonitoringWindow).order_by(MonitoringWindow.created_at.desc()).limit(1)).scalar_one_or_none()
        alerts = db.execute(select(DriftAlert).where(DriftAlert.state != "resolved")).scalars().all()
        return {
            "latest_window": None if w is None else {
                "status": w.status, "n_inspections": w.n_inspections, "distinct_parents": w.n_distinct_parents,
                "descriptor_psi": {k: round(v["psi"], 3) for k, v in w.descriptor_psi.items()},
                "embedding_auc": (w.embedding_drift or {}).get("auc"),
                "labelled_coverage": (w.labelled_quality or {}).get("coverage")},
            "open_alerts": [{"features": a.features, "severity": a.severity, "state": a.state} for a in alerts],
            "note": "drift is an investigation signal; it never retrains or promotes automatically",
        }
    finally:
        db.close()


def estimate_circularity_impact(ctx: ToolContext, verified_retreads: int) -> dict[str, Any]:
    n = max(1, min(int(verified_retreads), 100000))
    led = circularity.ledger({"retread": n}, {})
    return {"verified_retreads": n, **led["estimates"], "factors": led["factors"], "scope": led["scope"],
            "label": led["label"], "source": led["sources"][0]["url"], "kb": "kb-003#2"}


def search_knowledge(ctx: ToolContext, query: str) -> dict[str, Any]:
    return {"query": query, "passages": kb.search(query, 3)}


def get_review_queue(ctx: ToolContext) -> dict[str, Any]:
    db = new_session()
    try:
        has = select(ReviewLabel.id).where(ReviewLabel.inspection_id == Inspection.id).exists()
        n, oldest = db.execute(select(func.count(), func.min(Inspection.created_at))
                               .where(Inspection.status == "succeeded", ~has)).one()
        return {"awaiting_review": n, "oldest_waiting_at": oldest.isoformat() if oldest else None}
    finally:
        db.close()


IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "classify_tyre_image": classify_tyre_image,
    "get_inspection": get_inspection,
    "get_preprocessing_trace": get_preprocessing_trace,
    "get_model_status": get_model_status,
    "get_drift_status": get_drift_status,
    "estimate_circularity_impact": estimate_circularity_impact,
    "search_knowledge": search_knowledge,
    "get_review_queue": get_review_queue,
}


def execute(ctx: ToolContext, name: str, arguments: str | dict | None) -> dict[str, Any]:
    if name not in IMPLS:
        return {"error": "unknown_tool", "name": name}
    try:
        args: Any = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else arguments
    except json.JSONDecodeError:
        return {"error": "invalid_arguments"}
    if args in (None, "") or (isinstance(args, str) and not args.strip()):
        args = {}
    if not isinstance(args, dict):  # LLMs occasionally send a list, number or bare string
        return {"error": "invalid_arguments", "message": "tool arguments must be a JSON object"}
    allowed = set(next(s for s in SCHEMAS if s["function"]["name"] == name)["function"]["parameters"]["properties"])
    args = {k: v for k, v in args.items() if k in allowed}  # never pass unexpected arguments through
    try:
        result = IMPLS[name](ctx, **args)
    except TypeError as exc:
        result = {"error": "bad_arguments", "message": str(exc)[:200]}
    ctx.results.append({"tool": name, "args": args, "result": result})
    return result

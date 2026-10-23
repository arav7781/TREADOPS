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


VERDICT_LABEL = {"damaged": "Damaged", "not_damaged": "Not damaged", "uncertain": "Uncertain (expert check)"}
TRIAGE_ORDER = {"good_candidate": 0, "manual_review": 1, "defect_flagged": 2}


def verdict_of(binary_defective: bool, triage: str, quality_status: str) -> str:
    """Damaged if the policy says so; 'not damaged' only for a clean auto-clear; anything else is uncertain."""
    if binary_defective:
        return "damaged"
    return "not_damaged" if triage == "good_candidate" and quality_status != "low_quality" else "uncertain"


@dataclass
class ToolContext:
    actor: Any
    inspection_id: uuid.UUID | None = None          # set once classify_tyre_image has submitted the photo
    image_bytes: bytes | None = None                # the user's photo for this turn (in memory, never persisted raw)
    idempotency_key: str | None = None              # one inspection per turn, even if the tool is called twice
    frames: list[bytes] = field(default_factory=list)       # video turns: sampled frames of one tyre, in order
    frame_times: list[float] = field(default_factory=list)  # seconds into the clip for each frame
    video_source: str | None = None                          # "clip" or "camera"
    frame_inspections: list[uuid.UUID] = field(default_factory=list)
    session_inspections: list[dict[str, Any]] = field(default_factory=list)  # earlier in this chat, newest first
    emit: Callable[[str, dict[str, Any]], None] = lambda *_: None
    results: list[dict[str, Any]] = field(default_factory=list)


def _schema(name: str, description: str, properties: dict | None = None, required: list | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": required or [], "additionalProperties": False}}}


SCHEMAS = [
    _schema("classify_tyre_image",
            "Run the deployed YOLO tyre quality classifier on the photo attached to the user's current message. "
            "Returns the damage verdict (damaged / not damaged), whether an expert check is needed, triage, "
            "calibrated p(defective), image quality, the preprocessing stages and the recommended pathway. Only "
            "valid when the user attached a photo."),
    _schema("classify_tyre_video",
            "Inspect one tyre from the video frames attached to the user's current message (sampled around its "
            "circumference). Classifies every frame with the deployed YOLO model and returns a tyre-level verdict: "
            "damaged if any frame is damaged, the decisive (worst) frame, and per-frame results. Only valid when "
            "the user attached a video."),
    _schema("get_inspection", "Fetch the stored result and lineage of an earlier inspection by ID.",
            {"inspection_id": {"type": "string", "description": "inspection UUID"}}, ["inspection_id"]),
    _schema("get_preprocessing_trace", "Explain each preprocessing stage (ingest, canonical, letterbox, tensor, "
            "quality, inference, attribution) of an inspection, with timings.",
            {"inspection_id": {"type": "string"}}, ["inspection_id"]),
    _schema("show_attribution_heatmap", "Show an inspection's occlusion-attribution heatmap in the chat (the user "
            "sees the heatmap image, the pipeline and the result) and return what it measured. Use when the user "
            "asks to see the heatmap, the attribution or which region drove the decision.",
            {"inspection_id": {"type": "string", "description": "inspection UUID"}}, ["inspection_id"]),
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
    _schema("list_cameras", "Line cameras connected to TreadOps: name, production line, live status, frame rate, "
            "and last-hour damaged / not damaged / expert-check counts."),
    _schema("camera_quality_summary", "Damaged / not damaged / expert-check counts and damage rate per line camera "
            "over the last N hours (camera inspections only).",
            {"hours": {"type": "integer", "minimum": 1, "maximum": 168},
             "camera": {"type": "string", "description": "camera name or code; omit for all cameras"}}),
    _schema("inspect_camera", "Inspect the tyre currently in front of a line camera: takes its live frame and runs the "
            "deployed YOLO model on it as a normal pinned inspection. Use when the user asks what a camera sees now.",
            {"camera": {"type": "string", "description": "camera name or code"}}, ["camera"]),
]


def _compact(insp: Inspection, pred: Prediction | None, stages: list[InspectionStage]) -> dict[str, Any]:
    out: dict[str, Any] = {"inspection_id": str(insp.id), "status": insp.status, "source": insp.source,
                           "created_at": insp.created_at.isoformat() if insp.created_at else None,
                           "model": {"family": insp.model_tuple.get("model_family"),
                                     "registry_version": insp.model_tuple.get("registry_version"),
                                     "bundle_sha256": insp.model_tuple.get("bundle_sha256", "")[:12],
                                     "deployment_generation": insp.deployment_generation},
                           "metadata_origin": insp.metadata_origin}
    if pred is not None:
        circ = pred.circularity or {}
        verdict = verdict_of(pred.binary_defective, pred.triage, pred.quality_status)
        so = pred.second_opinion or {}
        out.update({
            "verdict": verdict, "verdict_label": VERDICT_LABEL[verdict],
            "needs_expert_check": pred.triage == "manual_review" or pred.quality_status == "low_quality",
            "second_opinion": {k: so.get(k) for k in ("model", "decision", "damage_visible", "confidence", "findings",
                                                      "location")} if so else None,
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
            earlier = [i["inspection_id"] for i in ctx.session_inspections]
            return {"error": "no_attachment", "message": "No photo is attached to this message. "
                    + ("Earlier inspections in this conversation can be used with get_inspection, "
                       "get_preprocessing_trace or show_attribution_heatmap." if earlier
                       else "Ask the user to upload a tyre photo."), "earlier_inspections": earlier}
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


def _find_camera(db, name: str | None):
    from treadops_api.models import Camera

    cams = db.execute(select(Camera).order_by(Camera.name)).scalars().all()
    if not name:
        return cams
    want = name.strip().lower()
    exact = [c for c in cams if want in (c.name.lower(), c.code)]
    return exact or [c for c in cams if want in c.name.lower() or want in c.code]


def list_cameras(ctx: ToolContext) -> dict[str, Any]:
    from treadops_api import cameras as cam_svc

    db = new_session()
    try:
        cams = _find_camera(db, None)
        st = cam_svc.stats(db, [c.code for c in cams])
        items = [{k: v for k, v in cam_svc.view(c, st.get(c.code)).items()
                  if k in ("name", "code", "line_id", "state", "live", "fps", "resolution", "simulated", "enabled",
                           "last_capture_at", "last_hour", "last_error")} for c in cams]
    finally:
        db.close()
    return {"cameras": items, "count": len(items),
            "note": "simulated cameras replay dataset photos; their results are labelled simulated"}


def camera_quality_summary(ctx: ToolContext, hours: int = 1, camera: str | None = None) -> dict[str, Any]:
    from treadops_api import cameras as cam_svc

    hours = max(1, min(int(hours), 168))
    db = new_session()
    try:
        cams = _find_camera(db, camera)
        if camera and not cams:
            return {"error": "camera_not_found", "message": f"No camera matches '{camera}'."}
        st = cam_svc.stats(db, [c.code for c in cams], hours=hours)
    finally:
        db.close()
    rows = []
    for c in cams:
        s = st.get(c.code, {})
        n = s.get("inspected", 0)
        rows.append({"camera": c.name, "line_id": c.line_id, "simulated": c.simulated, **s,
                     "damage_rate": round(s.get("damaged", 0) / n, 4) if n else None})
    return {"hours": hours, "cameras": rows}


def inspect_camera(ctx: ToolContext, camera: str) -> dict[str, Any]:
    from treadops_api.errors import ApiError
    from treadops_api.routers.cameras import inspect_now

    db = new_session()
    try:
        cams = _find_camera(db, camera)
        if len(cams) != 1:
            return {"error": "camera_not_found" if not cams else "ambiguous_camera",
                    "message": f"Cameras matching '{camera}': {[c.name for c in cams]}"}
        cam = cams[0]
        info = {"camera": cam.name, "line_id": cam.line_id, "simulated": cam.simulated}  # read before commit expires it
        body = inspect_now(db, cam, ctx.actor, f"{ctx.idempotency_key or uuid.uuid4()}:camera:{cam.code}")
    except ApiError as exc:
        return {"error": exc.code, "message": exc.message}
    finally:
        db.close()
    ctx.inspection_id = uuid.UUID(body["inspection_id"])
    ctx.emit("inspection", {"inspection_id": body["inspection_id"], "camera": info["camera"],
                            "deployment_generation": body.get("deployment_generation")})
    res = classify_tyre_image(ctx)  # waits for the worker exactly like an attached photo
    return {**res, **info}


def aggregate_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """One decision per tyre from its frames. Conservative: any damaged frame makes the tyre damaged, and the
    worst frame (triage first, then p(defective)) decides the triage and is shown as the evidence."""
    done = [f for f in frames if f.get("triage")]
    if not done:
        return {"error": "no_frame_classified", "frames": frames}
    decisive = max(done, key=lambda f: (TRIAGE_ORDER[f["triage"]], f["p_defective"]))
    verdicts = {f["verdict"] for f in done}
    verdict = "damaged" if "damaged" in verdicts else "uncertain" if "uncertain" in verdicts else "not_damaged"
    low_q = sum(1 for f in done if f.get("quality_status") == "low_quality")
    return {
        "tyre_level": True,
        "rule": "any damaged frame makes the tyre damaged, else any uncertain frame makes it uncertain; the worst "
                "frame decides the triage",
        "verdict": verdict, "verdict_label": VERDICT_LABEL[verdict],
        "needs_expert_check": decisive["triage"] == "manual_review" or low_q > 0,
        "triage": decisive["triage"], "triage_label": TRIAGE_LABEL[decisive["triage"]],
        "p_defective": decisive["p_defective"], "p_defective_min": min(f["p_defective"] for f in done),
        "decisive_frame": decisive["frame"], "decisive_time_s": decisive.get("time_s"),
        "decisive_inspection_id": decisive["inspection_id"],
        "frames_submitted": len(frames), "frames_classified": len(done), "frames_low_quality": low_q,
        "frames": frames,
    }


def classify_tyre_video(ctx: ToolContext) -> dict[str, Any]:
    """Each sampled frame becomes its own pinned inspection; the tool waits for all, then aggregates per tyre."""
    from treadops_api.errors import ApiError
    from treadops_api.services.inspections import accept
    from treadops_api.storage import get_storage

    if not ctx.frames:
        return {"error": "no_video", "message": "No video is attached to this message."}
    base = ctx.idempotency_key or f"copilot:{uuid.uuid4()}"
    if not ctx.frame_inspections:
        ctx.emit("tool_progress", {"tool": "classify_tyre_video",
                                   "message": f"submitting {len(ctx.frames)} frames to the YOLO inspection pipeline"})
        for i, raw in enumerate(ctx.frames):
            t = ctx.frame_times[i] if i < len(ctx.frame_times) else None
            db = new_session()
            try:
                insp, _ = accept(db, get_storage(), actor=ctx.actor, actor_id=ctx.actor.id, raw=raw,
                                 idempotency_key=f"{base}:f{i}", source="copilot", metadata_origin="user_provided",
                                 notes=f"video frame {i + 1}/{len(ctx.frames)}"
                                       + (f" at {t:.1f} s" if t is not None else "") + f" ({ctx.video_source or 'clip'})",
                                 enforce_backpressure=False)
            except ApiError as exc:
                return {"error": exc.code, "message": exc.message, "frame": i}
            finally:
                db.close()
            ctx.frame_inspections.append(insp.id)
            ctx.emit("inspection", {"inspection_id": str(insp.id), "frame": i, "time_s": t,
                                    "deployment_generation": insp.deployment_generation})
    wait = float(config()["classify_wait_s"]) * 2
    t0 = time.monotonic()
    reported: set[int] = set()
    while True:
        db = new_session()
        try:
            loaded = [_load(db, iid) for iid in ctx.frame_inspections]
        finally:
            db.close()
        frames = []
        for i, res in enumerate(loaded):
            t = ctx.frame_times[i] if i < len(ctx.frame_times) else None
            if res is None:
                frames.append({"frame": i, "time_s": t, "status": "missing"})
                continue
            f = {"frame": i, "time_s": t, "inspection_id": res["inspection_id"], "status": res["status"]}
            if res.get("triage"):
                f.update({k: res[k] for k in ("verdict", "triage", "p_defective", "quality_status")})
                if i not in reported:
                    reported.add(i)
                    ctx.emit("frame_result", {"tool": "classify_tyre_video", "frame": i, "time_s": t,
                                              "inspection_id": res["inspection_id"], "verdict": res["verdict"],
                                              "triage": res["triage"], "p_defective": res["p_defective"],
                                              "message": f"frame {i + 1}: {res['verdict_label']} "
                                                         f"(p(defective) {res['p_defective']:.3f})"})
            frames.append(f)
        if all(f["status"] in ("succeeded", "failed", "cancelled", "missing") for f in frames):
            out = aggregate_frames(frames)
            if not out.get("error"):
                decisive = loaded[out["decisive_frame"]] or {}
                out.update({"model": decisive.get("model"), "recommended_pathway": decisive.get("recommended_pathway")})
            return out
        if time.monotonic() - t0 > wait:
            return {"error": "timeout", "message": "Some frames are still queued; results will appear in Inspections.",
                    "frames": frames}
        time.sleep(0.5)


def _show(ctx: ToolContext, inspection_id: uuid.UUID, focus: str | None) -> None:
    """Put an existing inspection on screen in this answer (its pipeline and result), opened at `focus`."""
    ctx.emit("inspection_ref", {"inspection_id": str(inspection_id), "focus": focus,
                                "message": f"showing inspection {str(inspection_id)[:8]}"
                                           + (f" at its {focus} stage" if focus else "")})


def get_inspection(ctx: ToolContext, inspection_id: str) -> dict[str, Any]:
    try:
        iid = uuid.UUID(inspection_id)
    except ValueError:
        return {"error": "invalid_id"}
    db = new_session()
    try:
        res = _load(db, iid)
    finally:
        db.close()
    if res is None:
        return {"error": "not_found"}
    _show(ctx, iid, "classified")
    return res


def get_preprocessing_trace(ctx: ToolContext, inspection_id: str) -> dict[str, Any]:
    try:
        iid = uuid.UUID(inspection_id)
    except ValueError:
        return {"error": "invalid_id"}
    db = new_session()
    try:
        stages = db.execute(select(InspectionStage).where(InspectionStage.inspection_id == iid)
                            .order_by(InspectionStage.stage_index)).scalars().all()
    finally:
        db.close()
    if not stages:
        return {"error": "not_found_or_not_processed"}
    _show(ctx, iid, None)
    return {"inspection_id": inspection_id, "stages": [
        {"index": s.stage_index, "title": s.title, "duration_ms": round(s.duration_ms, 1), "log": s.log}
        for s in stages]}


def show_attribution_heatmap(ctx: ToolContext, inspection_id: str) -> dict[str, Any]:
    try:
        iid = uuid.UUID(inspection_id)
    except ValueError:
        return {"error": "invalid_id"}
    db = new_session()
    try:
        pred = db.execute(select(Prediction).where(Prediction.inspection_id == iid)).scalar_one_or_none()
        stage = db.execute(select(InspectionStage).where(InspectionStage.inspection_id == iid,
                                                         InspectionStage.name == "attribution")).scalar_one_or_none()
    finally:
        db.close()
    if pred is None or stage is None or not pred.attribution:
        return {"error": "no_heatmap", "message": "This inspection has no attribution heatmap (not found or not "
                                                  "processed yet)."}
    _show(ctx, iid, "attribution")
    a = pred.attribution
    cell = a.get("strongest_cell")
    return {"inspection_id": str(iid), "shown_in_chat": True, "method": a.get("method"), "grid": a.get("grid"),
            "strongest_cell": {"row": cell[0] + 1, "column": cell[1] + 1} if cell else None,
            "max_delta": round(float(a.get("max_delta", 0.0)), 3),
            "min_delta": round(float(a.get("min_delta", 0.0)), 3),
            "p_defective": round(pred.p_defective, 4), "triage_label": TRIAGE_LABEL[pred.triage],
            "verdict_label": VERDICT_LABEL[verdict_of(pred.binary_defective, pred.triage, pred.quality_status)],
            "how_to_read": "Each cell of the grid over the model input was hidden in turn; delta is the drop in the "
                           "defective-vs-good logit margin. Red = regions supporting 'defective' (max_delta is the "
                           "strongest), blue = regions supporting 'good' (negative delta). Rows and columns count "
                           "from the top left.",
            "note": "attribution heatmap, not a defect location or segmentation: the strongest cell shows what the "
                    "classifier weighed (it can be background or the rim) and must not be presented as where the "
                    "damage is, or as agreeing with a visible defect"}


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
    "classify_tyre_video": classify_tyre_video,
    "get_inspection": get_inspection,
    "get_preprocessing_trace": get_preprocessing_trace,
    "show_attribution_heatmap": show_attribution_heatmap,
    "get_model_status": get_model_status,
    "get_drift_status": get_drift_status,
    "estimate_circularity_impact": estimate_circularity_impact,
    "search_knowledge": search_knowledge,
    "get_review_queue": get_review_queue,
    "list_cameras": list_cameras,
    "camera_quality_summary": camera_quality_summary,
    "inspect_camera": inspect_camera,
}


def parse_arguments(raw: str | dict | None) -> Any:
    """Tool-call arguments as JSON. Some OpenAI-compatible gateways append stream artefacts after the object
    (for example `{}""`); the first JSON value is used when everything after it is only quotes or whitespace."""
    if raw is None or isinstance(raw, dict):
        return raw or {}
    text = raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        value, end = json.JSONDecoder().raw_decode(text)  # raises if there is no leading JSON value at all
        if text[end:].strip().strip("\"'`").strip():
            raise
        return value


def execute(ctx: ToolContext, name: str, arguments: str | dict | None) -> dict[str, Any]:
    if name not in IMPLS:
        return {"error": "unknown_tool", "name": name}
    try:
        args: Any = parse_arguments(arguments)
    except json.JSONDecodeError:
        return {"error": "invalid_arguments", "message": "tool arguments are not valid JSON"}
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

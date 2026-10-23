"""Shadow evaluation: champion and challenger see the identical prepared tensor for paired policy-role parents.

Nothing here creates inspections, so the challenger can never influence user-visible triage.
"""

from __future__ import annotations

import uuid
from typing import cast

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from treadops_api import audit, jobs
from treadops_api.db import must_get, new_session, session_scope
from treadops_api.models import ImageAsset, Job, ModelVersion, ShadowResult
from treadops_api.storage import get_storage
from treadops_ml.common import load_yaml, sha256_bytes
from treadops_ml.descriptors import image_descriptors, quality_status
from treadops_ml.metrics import binary_metrics, paired_bootstrap
from treadops_ml.preprocess import canonical_from_raw, letterbox, load_canonical, to_tensor
from treadops_ml.scenarios import derivative_bytes, transform_hash

from ..runtime import bundles
from ..tasks import JobContext, Permanent, handler


def _decide(bundle, p: float, rgb) -> str:
    q, _ = quality_status(image_descriptors(rgb), bundle.policy.quality)
    return bundle.policy.decide(p, q)[0]


@handler("shadow")
def run(ctx: JobContext) -> dict:
    p = ctx.payload
    replay_cfg = load_yaml("configs/replay.yaml")["scenarios"]
    transforms = {"baseline": {}, "low_light": {"brightness_factor": replay_cfg["low_light"]["brightness_factor"]},
                  "camera_blur": {"gaussian_sigma": replay_cfg["camera_blur"]["gaussian_sigma"]}}
    db = new_session()
    try:
        challenger = db.get(ModelVersion, uuid.UUID(p["challenger_id"]))
        champion = db.get(ModelVersion, uuid.UUID(p["champion_id"]))
        if challenger is None or champion is None:
            raise Permanent("model_version_missing", "champion or challenger not found")
        b_ch = bundles().get(champion.bundle_sha256, champion.bundle_uri)
        b_cl = bundles().get(challenger.bundle_sha256, challenger.bundle_uri)
        scen = [s for s in p["scenarios"] if s in transforms]
        pool = db.execute(select(ImageAsset).where(ImageAsset.image_origin == "public_dataset",
                                                   ImageAsset.role == "policy")).scalars().all()
        pool = sorted(pool, key=lambda a: sha256_bytes(f"shadow:{a.id}".encode()))[: max(1, p["max_images"] // len(scen))]
        storage = get_storage()
        out: dict[str, dict] = {}
        for si, sname in enumerate(scen):
            pc, pl, tc, tl, y, groups = [], [], [], [], [], []
            for i, a in enumerate(pool):
                if i % 10 == 0:
                    ctx.heartbeat((si + i / len(pool)) / len(scen), f"{sname} {i}/{len(pool)}")
                rgb0 = load_canonical(storage.get(cast(str, a.canonical_key)))
                canonical, _ = canonical_from_raw(derivative_bytes(rgb0, transforms[sname]), b_ch.preprocess)
                rgb = load_canonical(canonical)
                x = to_tensor(letterbox(rgb, b_ch.preprocess)[0], b_ch.preprocess)
                if b_cl.preprocess.sha256 != b_ch.preprocess.sha256:
                    x_cl = to_tensor(letterbox(rgb, b_cl.preprocess)[0], b_cl.preprocess)
                else:
                    x_cl = x
                p_ch = float(b_ch.p_defective(b_ch.raw_probs(x))[0])
                p_cl = float(b_cl.p_defective(b_cl.raw_probs(x_cl))[0])
                t_ch, t_cl = _decide(b_ch, p_ch, rgb), _decide(b_cl, p_cl, rgb)
                pc.append(p_ch)
                pl.append(p_cl)
                tc.append(t_ch)
                tl.append(t_cl)
                y.append(a.label == "defective")
                groups.append(a.group_id)
                with session_scope() as s:
                    s.execute(insert(ShadowResult).values(
                        job_id=ctx.job_id, challenger_version_id=challenger.id, champion_version_id=champion.id,
                        parent_image_id=a.id, image_id=uuid.uuid5(a.id, f"{sname}:{transform_hash(transforms[sname])}"),
                        scenario=sname, label=a.label,
                        champion_output={"p_defective": p_ch, "triage": t_ch},
                        challenger_output={"p_defective": p_cl, "triage": t_cl},
                    ).on_conflict_do_nothing(constraint="uq_shadow_job_image"))
            y_a = np.array(y)
            pred_ch = np.array(pc) >= b_ch.policy.binary_threshold
            pred_cl = np.array(pl) >= b_cl.policy.binary_threshold
            changes: dict[str, int] = {}
            for a_, b_ in zip(tc, tl, strict=True):
                if a_ != b_:
                    changes[f"{a_}->{b_}"] = changes.get(f"{a_}->{b_}", 0) + 1
            out[sname] = {
                "n": len(y), "triage_agreement": float(np.mean(np.array(tc) == np.array(tl))),
                "triage_changes": changes, "mean_abs_p_delta": float(np.mean(np.abs(np.array(pc) - np.array(pl)))),
                "champion": binary_metrics(pc, y_a.tolist(), b_ch.policy.binary_threshold),
                "challenger": binary_metrics(pl, y_a.tolist(), b_cl.policy.binary_threshold),
                "recall_difference": paired_bootstrap(cast(list[str], groups), y_a.tolist(),
                                                      pred_ch.tolist(), pred_cl.tolist(), "recall"),
                "specificity_difference": paired_bootstrap(cast(list[str], groups), y_a.tolist(),
                                                           pred_ch.tolist(), pred_cl.tolist(), "specificity"),
            }
    finally:
        db.close()
    report = {"cohort": "policy role (development evidence), paired parents per scenario",
              "champion_bundle": champion.bundle_sha256, "challenger_bundle": challenger.bundle_sha256,
              "scenarios": out, "user_visible_effect": "none (shadow)"}
    with session_scope() as s:
        mv = must_get(s, ModelVersion, challenger.id)
        if mv.stage == "validated":
            mv.stage = "shadow"
        audit.record(s, action="model.shadow.completed", resource_type="model_version", resource_id=mv.id,
                     correlation_id=ctx.correlation_id,
                     payload={k: {"agreement": v["triage_agreement"]} for k, v in out.items()})
        jobs.succeed(s, must_get(s, Job, ctx.job_id, populate_existing=True), report)
    return report

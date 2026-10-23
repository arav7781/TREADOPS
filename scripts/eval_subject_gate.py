"""Measure the subject gate on tyre photos and non-tyre probes, with the production code path.

1. Unusual-image score (subject_gate.residual) for every dev_train, validation (calibration + policy) and locked-test
   tyre, and for generated non-tyre probes (logos, documents, charts, flat colours, noise) plus the sample photos
   that ship with scikit-learn and matplotlib. Threshold: unusual_above = p99 of dev_train scores, rounded up to 0.01
   (training tyres; no evidence role is used to choose it).
2. With VISION_CHECK_MODEL set, the vision model's "is a tyre the subject?" answer wherever production would ask it
   (YOLO auto-clear or unusual image). Production rejects an image only when both agree: unusual score AND the
   vision model says no tyre; a "no tyre" on a typical-looking image sends it to an expert instead.

Writes docs/evidence/subject_gate/<timestamp>.json.  Usage: python scripts/eval_subject_gate.py [--no-vision]
"""

from __future__ import annotations

import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import select

from treadops_api.db import new_session
from treadops_api.models import Deployment, ModelVersion
from treadops_api.settings import get_settings
from treadops_ml.preprocess import PreprocessContract, load_canonical, tensor_from_canonical
from treadops_worker import subject_gate, vision_check
from treadops_worker.runtime import bundles, reference_encoder

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def probes() -> list[tuple[str, bytes]]:
    """Non-tyre images, generated here or shipped inside installed packages (nothing is committed)."""
    import matplotlib.pyplot as plt
    import sklearn

    rng = np.random.default_rng(0)
    out = []
    font = ImageFont.load_default(size=56)
    for i, (bg, fg, text) in enumerate([((255, 255, 255), (20, 60, 160), "HEALTH"), ((250, 250, 250), (200, 30, 40), "ACME"),
                                        ((10, 20, 60), (255, 220, 0), "MediCare"), ((255, 255, 255), (0, 120, 80), "Leaf"),
                                        ((235, 90, 20), (255, 255, 255), "BRAND"), ((0, 0, 0), (255, 255, 255), "LOGO")]):
        for j, shape in enumerate(("ellipse", "rectangle")):
            im = Image.new("RGB", (512, 512), bg)
            d = ImageDraw.Draw(im)
            getattr(d, shape)((156, 60, 356, 260), fill=fg)
            d.text((70, 330), text, fill=fg, font=font)
            out.append((f"logo_{i}_{j}", _jpeg(im)))
    for i in range(6):
        im = Image.new("RGB", (600, 800), (255, 255, 255))
        d = ImageDraw.Draw(im)
        for y in range(60, 760, 28):
            d.line((50, y, 50 + int(rng.integers(200, 500)), y), fill=(60, 60, 60), width=6)
        out.append((f"document_{i}", _jpeg(im)))
    for i in range(6):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rng.normal(size=50).cumsum())
        ax.bar(range(10), rng.random(10))
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        out.append((f"chart_{i}", _jpeg(Image.open(buf))))
    for i, colour in enumerate([(0, 0, 0), (255, 255, 255), (128, 128, 128), (200, 50, 50)]):
        out.append((f"flat_{i}", _jpeg(Image.new("RGB", (512, 512), colour))))
    out.append(("noise", _jpeg(Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)))))
    shipped = [Path(sklearn.__file__).parent / "datasets/images" / n for n in ("china.jpg", "flower.jpg")]
    shipped += [Path(matplotlib.get_data_path()) / "sample_data" / n
                for n in ("grace_hopper.jpg", "Minduka_Present_Blue_Pack.png", "logo2.png")]
    out += [(f"photo_{p.stem}", _jpeg(Image.open(p))) for p in shipped if p.exists()]
    return out


def main(use_vision: bool) -> int:
    db = new_session()
    dep = db.execute(select(Deployment).where(Deployment.state == "active")).scalar_one()
    mv = db.get(ModelVersion, dep.model_version_id)
    db.close()
    bundle = bundles().get(mv.bundle_sha256, mv.bundle_uri)
    bundle_dir = bundles().local_dir(mv.bundle_sha256, mv.bundle_uri)
    contract, enc = PreprocessContract(), reference_encoder()
    mean, components = subject_gate._reference(str(bundle_dir))
    split = pd.read_parquet(ROOT / "data/manifests/split_assignments.parquet")

    def measure(items: list[tuple[str, bytes]], label: str) -> list[dict]:
        recs = []
        for i in range(0, len(items), 32):
            chunk = items[i : i + 32]
            tensors = [tensor_from_canonical(d, contract) for _, d in chunk]
            scores = subject_gate.residual(enc.embed(np.concatenate(tensors)), mean, components)
            for (name, data), t, s in zip(chunk, tensors, scores, strict=True):
                rgb = np.array(load_canonical(data))
                p = bundle.predict_canonical_rgb(rgb, t)
                recs.append({"id": name, "set": label, "score": round(float(s), 4), "policy_triage": p.triage,
                             "rgb": rgb})
        return recs

    def role(*names: str) -> list[tuple[str, bytes]]:
        rows = split[split.role.isin(names)]
        return [(r.image_id, (ROOT / f"data/processed/images/{r.original_sha256}.jpg").read_bytes())
                for r in rows.itertuples()]

    recs = (measure(role("dev_train"), "dev_train") + measure(role("calibration", "policy"), "validation")
            + measure(role("test"), "test") + measure(probes(), "non_tyre"))
    train = np.array([r["score"] for r in recs if r["set"] == "dev_train"])
    threshold = math.ceil(float(np.quantile(train, 0.99)) * 100) / 100
    for r in recs:
        r["unusual"] = r["score"] > threshold
        r["asked"] = r["set"] != "dev_train" and (r["unusual"] or r["policy_triage"] == "good_candidate")

    vision = use_vision and vision_check.enabled()
    if vision:
        asked = [r for r in recs if r["asked"]]
        with ThreadPoolExecutor(8) as ex:
            for r, o in zip(asked, ex.map(lambda r: vision_check.second_opinion(r["rgb"]), asked), strict=True):
                r["tyre_visible"], r["subject"] = o.get("tyre_visible"), o.get("subject")

    def summary(name: str) -> dict:
        g = [r for r in recs if r["set"] == name]
        s = np.array([r["score"] for r in g])
        out = {"images": len(g), "score_p50": round(float(np.median(s)), 4),
               "score_p99": round(float(np.quantile(s, 0.99)), 4), "score_min": round(float(s.min()), 4),
               "score_max": round(float(s.max()), 4), "unusual": int(sum(r["unusual"] for r in g))}
        if vision and name != "dev_train":
            asked = [r for r in g if r["asked"]]
            no_tyre = [r for r in asked if r.get("tyre_visible") is False]
            out |= {"vision_asked": len(asked), "vision_no_tyre": len(no_tyre),
                    "vision_no_answer": sum(r.get("tyre_visible") is None for r in asked),
                    "rejected": sum(r["unusual"] for r in no_tyre),
                    "no_tyre_but_typical_to_expert": sum(not r["unusual"] for r in no_tyre)}
        return out

    template_sha = vision_check.prompt()[2] if vision else None
    report = {
        "at": datetime.now(UTC).isoformat(), "bundle_sha256": bundle.sha, "model_version": mv.registry_version,
        "method": subject_gate.config()["method"], "unusual_above": threshold,
        "threshold_rule": "p99 of dev_train scores, rounded up to 0.01",
        "vision_model": get_settings().vision_check_model if vision else None,
        "vision_prompt_version": vision_check.prompt()[1] if vision else None, "vision_prompt_sha256": template_sha,
        "sets": {n: summary(n) for n in ("dev_train", "validation", "test", "non_tyre")},
        "non_tyre_probes": [{k: r.get(k) for k in ("id", "score", "unusual", "policy_triage", "tyre_visible", "subject")}
                            for r in recs if r["set"] == "non_tyre"],
        "tyres_vision_said_no_tyre": [{k: r.get(k) for k in ("id", "set", "score", "unusual", "subject")}
                                      for r in recs if r["set"] in ("validation", "test")
                                      and r.get("tyre_visible") is False],
    }
    out = ROOT / "docs/evidence/subject_gate" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "non_tyre_probes"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(use_vision="--no-vision" not in sys.argv))

"""Immutable model bundle: verification, loading and the prediction adapter used by every serving path.

A bundle directory contains manifest.json whose `files` map lists every file with its SHA-256. Loading fails
closed: a missing file, a hash mismatch or a class map that disagrees with model.names rejects the bundle.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibrate import TemperatureCalibrator
from .common import sha256_file, sha256_json
from .descriptors import image_descriptors, quality_status
from .policy import Policy
from .preprocess import Limits, PreprocessContract, ingest, run_stages, tensor_from_canonical

REQUIRED_FILES = ("best.pt", "class_mapping.json", "preprocess.json", "calibration.json", "policy.json",
                  "metrics.json", "gate_report.json", "model_card.md")
# Optional: the damage locator (a YOLO26 detector) and its settings; packaged only when its own gate passed.
LOCATOR_FILES = ("damage_locator.pt", "damage.json")


_PREDICT_LOCK = threading.Lock()


def tensor_input_only(model):
    """Serve prepared tensors only. Checkpoints trained with FullImageTrainer pickle our EvalTransform as
    model.transforms; Ultralytics' classify predictor introspects it as a torchvision Compose. Tensor inputs never
    use those transforms, so they are removed rather than re-applied (no default centre crop can sneak in)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "transforms"):
        inner.transforms = None
    return model


def native_raw_probs(model, x: np.ndarray, device: str, imgsz: int, batch: int = 32) -> np.ndarray:
    """Run an Ultralytics classify model on prepared (N,3,S,S) tensors; no Ultralytics transforms applied."""
    import torch

    out: list[np.ndarray] = []
    for i in range(0, len(x), batch):
        chunk = torch.from_numpy(np.ascontiguousarray(x[i : i + batch]))
        with _PREDICT_LOCK:
            results = model.predict(source=chunk, verbose=False, device=device, imgsz=imgsz)
        out.extend(r.probs.data.detach().cpu().numpy().astype(np.float64) for r in results)
    return np.stack(out)


class BundleInvalid(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def bundle_sha_from_files(files: dict[str, str]) -> str:
    return sha256_json(dict(sorted(files.items())))


def verify_bundle(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise BundleInvalid("missing_manifest", f"{root} has no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    files: dict[str, str] = manifest.get("files", {})
    for req in REQUIRED_FILES:
        if req not in files:
            raise BundleInvalid("missing_file", f"manifest does not list {req}")
    for rel, expected in files.items():
        p = root / rel
        if not p.exists():
            raise BundleInvalid("missing_file", f"{rel} is listed but absent")
        actual = sha256_file(p)
        if actual != expected:
            raise BundleInvalid("hash_mismatch", f"{rel}: expected {expected}, got {actual}")
    if bundle_sha_from_files(files) != manifest.get("bundle_sha256"):
        raise BundleInvalid("bundle_sha_mismatch", "bundle_sha256 does not match the file map")
    if manifest.get("task") != "classify":
        raise BundleInvalid("wrong_task", f"task={manifest.get('task')}")
    return manifest


@dataclass
class Prediction:
    raw_probs: dict[str, float]
    calibrated_probs: dict[str, float]
    p_defective: float
    triage: str
    triage_reason: str
    binary_defective: bool
    quality_status: str
    quality_reasons: list[str]
    descriptors: dict[str, float]
    timing_ms: dict[str, float]


class Bundle:
    def __init__(self, root: str | Path, device: str = "cpu", verify: bool = True, runtime: str | None = None):
        self.root = Path(root)
        self.manifest = verify_bundle(self.root) if verify else json.loads((self.root / "manifest.json").read_text())
        self.sha = self.manifest["bundle_sha256"]
        self.device = device
        # Default: what the bundle was packaged to serve with (ONNX for torchvision models, parity-checked).
        self.runtime = runtime or self.manifest.get("serving", {}).get("preferred_runtime", "native")
        self._tv = False
        self.class_mapping: dict[str, int] = json.loads((self.root / "class_mapping.json").read_text())["mapping"]
        self.preprocess = PreprocessContract.from_dict(json.loads((self.root / "preprocess.json").read_text()))
        self.calibrator = TemperatureCalibrator.from_dict(json.loads((self.root / "calibration.json").read_text()))
        self.policy = Policy.from_dict(json.loads((self.root / "policy.json").read_text()))
        self._model: Any = None
        self._ort: Any = None
        self._locator: Any = None
        files = self.manifest.get("files", {})
        self.damage_cfg = (json.loads((self.root / "damage.json").read_text())
                           if all(f in files for f in LOCATOR_FILES) else None)
        self.load()

    # -- runtimes ----------------------------------------------------------------------------------
    def load(self) -> None:
        if self.runtime == "onnx":
            import onnxruntime as ort

            if not (self.root / "model.onnx").exists():
                raise BundleInvalid("missing_file", "model.onnx not present in this bundle")
            self._ort = ort.InferenceSession(str(self.root / "model.onnx"), providers=["CPUExecutionProvider"])
            names = self.manifest["serving"]["class_names"]
        elif self.manifest.get("model_framework") == "torchvision":
            from . import tvmodel

            self._model, tv_names = tvmodel.load(self.root / "best.pt")
            self._tv = True
            names = {str(k): v for k, v in tv_names.items()}
        else:
            from ultralytics import YOLO

            self._model = tensor_input_only(YOLO(str(self.root / "best.pt"), task="classify"))
            names = {str(k): v for k, v in self._model.names.items()}
        self._verify_names({int(k): v for k, v in names.items()})

    def _verify_names(self, names: dict[int, str]) -> None:
        expected = {v: k for k, v in names.items()}
        if expected != self.class_mapping:
            raise BundleInvalid("class_map_mismatch", f"model.names {names} != class_mapping {self.class_mapping}")
        if "defective" not in self.class_mapping:
            raise BundleInvalid("class_map_mismatch", "class mapping lacks 'defective'")

    def raw_probs(self, x: np.ndarray) -> np.ndarray:
        """(N,3,S,S) float32 tensor batch -> (N,C) probabilities in model-index order."""
        if self._ort is not None:
            inp = self._ort.get_inputs()[0].name
            return np.asarray(self._ort.run(None, {inp: x.astype(np.float32)})[0], dtype=np.float64)
        if self._tv:
            from . import tvmodel

            with _PREDICT_LOCK:
                return tvmodel.raw_probs(self._model, x)
        return native_raw_probs(self._model, x, self.device, self.preprocess.imgsz)

    def locate_damage(self, rgb: np.ndarray) -> dict[str, Any] | None:
        """Boxes around local damage on the canonical image, strongest first; None when the bundle has no locator.
        `escalate` is true when the strongest box reaches the threshold chosen on the policy role's good tyres and
        the locator passed its own gate (authority "escalate"); a display-only locator never escalates."""
        if self.damage_cfg is None:
            return None
        if self._locator is None:
            from ultralytics import YOLO

            self._locator = YOLO(str(self.root / "damage_locator.pt"), task="detect")
        cfg = self.damage_cfg
        can_escalate = cfg.get("authority", "escalate") == "escalate"  # bundles before authority: gate had passed
        t0 = time.perf_counter()
        with _PREDICT_LOCK:
            res = self._locator.predict(source=[rgb], imgsz=cfg["imgsz"], conf=cfg["display_confidence"],
                                        verbose=False, device="cpu")[0]
        boxes = sorted(({"class": res.names[int(c)], "confidence": round(float(p), 4),
                         "box": [round(float(v), 4) for v in xyxyn],
                         "escalates": can_escalate and float(p) >= cfg["escalation_threshold"]}
                        for c, p, xyxyn in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxyn, strict=True)),
                       key=lambda b: -b["confidence"])[:10]
        return {"version": cfg["version"], "threshold": cfg["escalation_threshold"], "boxes": boxes,
                "authority": cfg.get("authority", "escalate"), "gate_status": cfg.get("gate_status", "passed"),
                "max_confidence": boxes[0]["confidence"] if boxes else 0.0,
                "escalate": bool(boxes and boxes[0]["escalates"]), "ms": round((time.perf_counter() - t0) * 1000, 1)}

    # -- semantics ---------------------------------------------------------------------------------
    @property
    def defective_index(self) -> int:
        return self.class_mapping["defective"]

    def semantic(self, vec: np.ndarray) -> dict[str, float]:
        return {label: float(vec[idx]) for label, idx in sorted(self.class_mapping.items())}

    def p_defective(self, raw: np.ndarray) -> np.ndarray:
        return np.atleast_2d(self.calibrator.apply(raw))[:, self.defective_index]

    def predict_canonical_rgb(self, rgb: np.ndarray, x: np.ndarray | None = None) -> Prediction:
        t0 = time.perf_counter()
        if x is None:
            from .preprocess import letterbox, to_tensor

            x = to_tensor(letterbox(rgb, self.preprocess)[0], self.preprocess)
        t1 = time.perf_counter()
        raw = self.raw_probs(x)[0]
        t2 = time.perf_counter()
        if not np.all(np.isfinite(raw)):
            raise BundleInvalid("non_finite_output", "model produced non-finite probabilities")
        cal = self.calibrator.apply(raw)
        p = float(cal[self.defective_index])
        desc = image_descriptors(rgb)
        q_status, q_reasons = quality_status(desc, self.policy.quality)
        triage, reason = self.policy.decide(p, q_status)
        t3 = time.perf_counter()
        return Prediction(
            raw_probs=self.semantic(raw),
            calibrated_probs=self.semantic(cal),
            p_defective=p,
            triage=triage,
            triage_reason=reason,
            binary_defective=self.policy.binary(p),
            quality_status=q_status,
            quality_reasons=q_reasons,
            descriptors=desc,
            timing_ms={"tensor_ms": (t1 - t0) * 1000, "inference_ms": (t2 - t1) * 1000,
                       "postprocess_ms": (t3 - t2) * 1000},
        )

    def predict_bytes(self, image_bytes: bytes, limits: Limits = Limits()) -> tuple[Prediction, dict[str, Any]]:
        ing = ingest(image_bytes, limits, self.preprocess)
        staged = run_stages(ing.sanitized, ing.info, self.preprocess)
        pred = self.predict_canonical_rgb(staged["canonical_rgb"], staged["tensor"])
        return pred, staged

    def predict_canonical_bytes(self, canonical: bytes) -> Prediction:
        from .preprocess import load_canonical

        return self.predict_canonical_rgb(load_canonical(canonical), tensor_from_canonical(canonical, self.preprocess))

    def lineage(self) -> dict[str, Any]:
        m = self.manifest
        return {
            "model_version": m.get("registry", {}).get("version"),
            "bundle_sha256": self.sha,
            "dataset_snapshot_id": m["lineage"].get("dataset_snapshot_id"),
            "split_sha256": m["lineage"].get("split_sha256"),
            "preprocess_version": self.preprocess.version,
            "preprocess_sha256": self.preprocess.sha256,
            "calibration_version": self.calibrator.version,
            "policy_version": self.policy.version,
            "git_sha": m["lineage"].get("git_sha"),
            "training_run": m["lineage"].get("training_run_name"),
            "mlflow_run_id": m["lineage"].get("mlflow_run_id"),
        }

"""MLflow pyfunc wrapper: one loadable model = preprocessing + YOLO + calibration + policy (the whole bundle)."""

from __future__ import annotations

import base64

import mlflow.pyfunc
import pandas as pd


class TreadOpsBundleModel(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with `image_b64` (raw image bytes, base64). Output: calibrated decision per row."""

    def load_context(self, context):
        from treadops_ml.bundle import Bundle

        self.bundle = Bundle(context.artifacts["bundle"], device="cpu")

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        rows = []
        for b64 in model_input["image_b64"]:
            pred, _ = self.bundle.predict_bytes(base64.b64decode(b64))
            rows.append({
                "p_defective": pred.p_defective,
                "triage": pred.triage,
                "quality_status": pred.quality_status,
                "raw_defective": pred.raw_probs["defective"],
                "bundle_sha256": self.bundle.sha,
            })
        return pd.DataFrame(rows)

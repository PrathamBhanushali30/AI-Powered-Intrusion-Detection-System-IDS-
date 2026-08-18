from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

from .utils import load_joblib
from .config import DEFAULT_TARGET_COL

ART = Path("artifacts")

app = FastAPI(title="IDS Inference API", version="1.0.0")

class PredictRequest(BaseModel):
    samples: List[Dict]  # list of JSON feature dicts
    threshold: Optional[float] = None  # for AE

@app.on_event("startup")
def load_artifacts():
    global preproc, le, model, ae, mode
    preproc = load_joblib(ART / "preprocessor.joblib")
    le_path = ART / "label_encoder.joblib"
    ae_path = ART / "autoencoder.keras"

    mode = None
    if le_path.exists():
        le = load_joblib(le_path)
        import joblib
        if (ART / "model_rf.joblib").exists():
            model = joblib.load(ART / "model_rf.joblib")
        elif (ART / "model_mlp.joblib").exists():
            model = joblib.load(ART / "model_mlp.joblib")
        else:
            raise RuntimeError("No supervised model artifact found in artifacts/")
        mode = "supervised"
    elif ae_path.exists():
        ae = tf.keras.models.load_model(ae_path)
        mode = "autoencoder"
    else:
        raise RuntimeError("No recognizable model artifacts found. Train first.")

@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.samples)
    if DEFAULT_TARGET_COL in df.columns:
        df = df.drop(columns=[DEFAULT_TARGET_COL])
    Xt = preproc.transform(df)

    if mode == "supervised":
        y_pred_enc = model.predict(Xt)
        y_pred_enc = np.asarray(y_pred_enc).astype(int)
        labels = load_joblib(ART / "label_encoder.joblib").inverse_transform(y_pred_enc)
        return {"mode": mode, "predictions": labels.tolist()}
    else:
        recon = ae.predict(Xt, verbose=0)
        mse = np.mean((Xt - recon) ** 2, axis=1)
        thr = float(req.threshold) if req.threshold else float(np.percentile(mse, 95))
        y_pred = np.where(mse > thr, "ANOMALY", "BENIGN").tolist()
        return {"mode": mode, "predictions": y_pred, "scores": mse.tolist(), "threshold": thr}

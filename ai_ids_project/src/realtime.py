from __future__ import annotations
import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf

from .utils import load_joblib
from .config import DEFAULT_TARGET_COL

def predict_supervised(preproc, model, le, df_batch):
    Xt = preproc.transform(df_batch)
    y_pred_enc = model.predict(Xt)
    y_pred_enc = np.asarray(y_pred_enc).astype(int)
    return le.inverse_transform(y_pred_enc)

def predict_autoencoder(preproc, ae, df_batch, threshold: float | None = None):
    Xt = preproc.transform(df_batch)
    recon = ae.predict(Xt, verbose=0)
    mse = np.mean((Xt - recon) ** 2, axis=1)
    if threshold is None:
        threshold = float(np.percentile(mse, 95))
    y_pred = np.where(mse > threshold, "ANOMALY", "BENIGN")
    return y_pred, mse, threshold

def main():
    ap = argparse.ArgumentParser(description="Simulate real-time detection from a CSV stream.")
    ap.add_argument("--stream_csv", required=True, help="Path to CSV that we will stream row by row.")
    ap.add_argument("--artifacts_dir", required=True, help="Path to artifacts/ from training.")
    ap.add_argument("--target_col", default=DEFAULT_TARGET_COL, help="Column to drop if present during inference.")
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between rows to simulate stream.")
    ap.add_argument("--out", default="predictions.csv", help="Where to append predictions.")
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    preproc = load_joblib(art / "preprocessor.joblib")
    le_path = art / "label_encoder.joblib"
    ae_path = art / "autoencoder.keras"

    mode = None
    model = None
    le = None
    if le_path.exists():
        # supervised mode
        le = load_joblib(le_path)
        import joblib
        if (art / "model_rf.joblib").exists():
            model = joblib.load(art / "model_rf.joblib")
        elif (art / "model_mlp.joblib").exists():
            model = joblib.load(art / "model_mlp.joblib")
        else:
            raise RuntimeError("No supervised model artifact found in artifacts_dir.")
        mode = "supervised"
    elif ae_path.exists():
        ae = tf.keras.models.load_model(ae_path)
        mode = "autoencoder"
    else:
        raise RuntimeError("No recognizable model artifacts found (label_encoder or autoencoder).")

    # Prepare output file
    out_path = Path(args.out)
    if not out_path.exists():
        out_path.write_text("timestamp,mode,prediction,score(optional)
")

    # Stream the CSV
    for chunk in pd.read_csv(args.stream_csv, chunksize=1):
        row = chunk.copy()
        if args.target_col in row.columns:
            row = row.drop(columns=[args.target_col])

        if mode == "supervised":
            pred = predict_supervised(preproc, model, le, row)[0]
            line = f"{time.time():.3f},{mode},{pred},\n"
        else:
            y_pred, mse, thr = predict_autoencoder(preproc, ae, row, threshold=None)
            line = f"{time.time():.3f},{mode},{y_pred[0]},{mse[0]:.6f}\n"

        with open(out_path, "a") as f:
            f.write(line)

        print(line.strip())
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse
from pathlib import Path
import json

from sklearn.preprocessing import LabelEncoder

from .config import ARTIFACT_DIR, DEFAULT_TARGET_COL
from .data import load_dataset, split_preprocess
from .models import train_supervised, train_autoencoder
from .utils import save_joblib, save_json

def main():
    ap = argparse.ArgumentParser(description="Train IDS models (supervised or autoencoder).")
    ap.add_argument("--data_csv", required=True, help="Path to CSV with a 'label' column.")
    ap.add_argument("--dataset", default=None, help="Dataset hint: nsl-kdd or None.")
    ap.add_argument("--task", choices=["supervised", "autoencoder"], default="supervised")
    ap.add_argument("--model", choices=["rf", "mlp"], default="rf", help="Supervised model type if task=supervised.")
    ap.add_argument("--target_col", default=DEFAULT_TARGET_COL)
    ap.add_argument("--output_dir", default=str(ARTIFACT_DIR))
    ap.add_argument("--smote", action="store_true", help="Apply SMOTE on train split (supervised).")

    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_csv, dataset=args.dataset)

    if args.task == "supervised":
        X_train, X_test, y_train, y_test, preproc, feat_names = split_preprocess(
            df, target_col=args.target_col, smote=args.smote, dataset=args.dataset
        )

        # Encode labels to integers for models if needed
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        res = train_supervised(X_train, y_train_enc, X_test, y_test_enc, model_type=args.model)

        # Save artifacts
        save_joblib(outdir / "preprocessor.joblib", preproc)
        save_joblib(outdir / "label_encoder.joblib", le)
        from joblib import dump
        dump(res.model, outdir / f"model_{args.model}.joblib")

        save_json(outdir / "metrics.json", res.metrics)

        print("=== Training complete ===")
        print(f"Artifacts saved at: {outdir.resolve()}")
        print(json.dumps(res.metrics, indent=2))

    else:  # autoencoder
        X_train, X_test, y_train, y_test, preproc, feat_names = split_preprocess(
            df, target_col=args.target_col, smote=False, dataset=args.dataset
        )
        res = train_autoencoder(X_train, X_test, y_test)

        # Save artifacts
        from joblib import dump
        preproc_path = outdir / "preprocessor.joblib"
        save_joblib(preproc_path, preproc)

        ae_path = outdir / "autoencoder.keras"
        res.model.save(ae_path)

        save_json(outdir / "metrics.json", res.metrics)

        print("=== Autoencoder training complete ===")
        print(f"Artifacts saved at: {outdir.resolve()}")
        print(json.dumps(res.metrics, indent=2))


if __name__ == "__main__":
    main()

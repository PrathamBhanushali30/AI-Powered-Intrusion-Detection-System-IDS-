from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from .config import RANDOM_STATE, TEST_SIZE, NSL_KDD_CATEGORICAL, DEFAULT_TARGET_COL


def load_dataset(csv_path: str, dataset: str | None = None) -> pd.DataFrame:
    """
    Load a CSV as DataFrame. If dataset == 'nsl-kdd', attempt light normalization of column names and label field.
    Otherwise, just read CSV.
    """
    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    if dataset and dataset.lower() == "nsl-kdd":
        # NSL-KDD variants often have 'class' or 'label' with values like 'normal' vs 'neptune' etc.
        # Ensure 'label' exists.
        if "label" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "label"})
        if "label" not in df.columns:
            raise ValueError("NSL-KDD mode expects a 'label' column in the CSV.")

        # Convert known 'normal' to BENIGN for consistency with CIC style (optional)
        df["label"] = df["label"].astype(str).str.strip()
        df["label"] = df["label"].replace({"normal": "BENIGN", "Normal": "BENIGN"})
    else:
        if "label" not in df.columns:
            raise ValueError("CSV must contain a 'label' column (binary or multiclass).")

    return df


def split_preprocess(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    smote: bool = False,
    dataset: str | None = None
):
    """
    Splits data into train/test. Builds a sklearn Pipeline that encodes categoricals and scales numerics.
    Returns: (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # Identify categorical columns
    if dataset and dataset.lower() == "nsl-kdd":
        cat_cols = [c for c in NSL_KDD_CATEGORICAL if c in X.columns]
    else:
        # Heuristic: object dtype as categorical
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    num_cols = [c for c in X.columns if c not in cat_cols]

    # Build preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    if smote:
        # Fit transform on train to get numeric matrix for SMOTE
        Xt = preprocessor.fit_transform(X_train)
        sm = SMOTE(random_state=RANDOM_STATE)
        Xt_res, y_train = sm.fit_resample(Xt, y_train)
        # Keep preprocessor fitted. We'll use Xt_res as X_train inputs for model fitting (pipeline-less fit).
        feature_names = _feature_names(preprocessor, num_cols)
        return Xt_res, preprocessor.transform(X_test), y_train, y_test, preprocessor, feature_names
    else:
        # Return transformed arrays (fit on train only)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        feature_names = _feature_names(preprocessor, num_cols)
        return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names


def _feature_names(preprocessor, num_cols):
    cat_ohe = preprocessor.named_transformers_["cat"]
    cat_names = []
    if hasattr(cat_ohe, "get_feature_names_out"):
        cat_names = list(cat_ohe.get_feature_names_out())
    return cat_names + num_cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--dataset", default=None, help="nsl-kdd or None")
    args = ap.parse_args()
    df = load_dataset(args.data_csv, args.dataset)
    print(df.head())
    print(df["label"].value_counts())

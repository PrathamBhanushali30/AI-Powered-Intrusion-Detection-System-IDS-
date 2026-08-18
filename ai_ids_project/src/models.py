from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

import tensorflow as tf
from tensorflow.keras import layers, models as kmodels, losses, optimizers, callbacks


@dataclass
class TrainResult:
    model: object
    history: Optional[dict]
    metrics: dict


def train_supervised(
    X_train, y_train, X_test, y_test, model_type: Literal["rf", "mlp"] = "rf"
) -> TrainResult:
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        )
    elif model_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=30,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.1
        )
    else:
        raise ValueError("model_type must be 'rf' or 'mlp'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilities if available
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm,
    }

    # Macro AUC (if binary or one-vs-rest)
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
        try:
            # derive macro-average ROC AUC
            if len(np.unique(y_test)) > 2:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=np.unique(y_test))
                auc = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            metrics["roc_auc"] = float(auc)
        except Exception:
            pass

    return TrainResult(model=model, history=None, metrics=metrics)


def build_autoencoder(input_dim: int) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    bottleneck = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(bottleneck)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(input_dim, activation=None)(x)

    model = kmodels.Model(inp, out, name="autoencoder")
    model.compile(optimizer=optimizers.Adam(1e-3), loss=losses.MeanSquaredError())
    return model


def train_autoencoder(
    X_train, X_test, y_test, epochs: int = 30, batch_size: int = 256
) -> TrainResult:
    input_dim = X_train.shape[1]
    ae = build_autoencoder(input_dim)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = ae.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[es]
    )
    # Reconstruction error as anomaly score
    recon = ae.predict(X_test, verbose=0)
    mse = np.mean(np.square(X_test - recon), axis=1)

    # Heuristic threshold: 95th percentile of train reconstruction error
    recon_tr = ae.predict(X_train, verbose=0)
    mse_tr = np.mean(np.square(X_train - recon_tr), axis=1)
    thr = float(np.percentile(mse_tr, 95))

    y_pred = (mse > thr).astype(int)  # 1=anomaly
    # Map labels to 0/1 for eval: assume 'BENIGN' or 'normal' as 0
    y_is_attack = np.array([0 if str(y).upper() in ("BENIGN", "NORMAL") else 1 for y in y_test])

    report = classification_report(y_is_attack, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_is_attack, y_pred).tolist()
    try:
        ap = average_precision_score(y_is_attack, mse)
    except Exception:
        ap = None

    metrics = {
        "threshold": thr,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    if ap is not None:
        metrics["average_precision"] = float(ap)

    return TrainResult(model=ae, history=hist.history, metrics=metrics)

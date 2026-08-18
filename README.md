# AI-Powered Intrusion Detection System (IDS)

End-to-end IDS using classical ML and Deep Learning (Autoencoder) for anomaly detection on tabular flow data
(e.g., NSL-KDD/CIC-IDS CSVs). Includes: preprocessing pipeline, model training, evaluation reports, 
real-time(ish) simulation, and an optional Streamlit dashboard.

## Features
- Clean **train → evaluate → save** pipeline
- **Two modes**: Supervised (RandomForest/MLP) and Unsupervised (Autoencoder)
- **Consistent preprocessing** (fit-on-train; saved as artifact)
- **Imbalance handling** (optional SMOTE)
- **Metrics & plots**: confusion matrix, PR/ROC, classification report
- **Real-time simulation** from CSV "stream" and optional **Scapy** sniffer (demo)
- **FastAPI** service for inference (optional)
- **Streamlit dashboard** for live monitoring (optional)

## Folder Structure
```
ai_ids_project/
├── README.md
├── requirements.txt
├── streamlit_app.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── realtime.py
│   ├── serve.py
│   └── utils.py
└── artifacts/ (auto-created)
```

## Quickstart

1) Create a venv and install deps:
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Prepare a CSV dataset (e.g., NSL-KDD or CICIDS2017). Put it anywhere, then run training.
Assume your CSV has a **label** column (binary or multiclass). For NSL-KDD format, see `data.py` note.

**Supervised (RandomForest example):**
```bash
python -m src.train   --data_csv /path/to/your.csv   --task supervised   --model rf   --target_col label   --output_dir artifacts   --smote
```

**Deep Learning Autoencoder (unsupervised anomaly detection):**
```bash
python -m src.train   --data_csv /path/to/your.csv   --task autoencoder   --target_col label   --output_dir artifacts
```

3) Simulate streaming detection from a CSV (tail-like read) and write live predictions to a file:
```bash
python -m src.realtime   --stream_csv /path/to/your.csv   --artifacts_dir artifacts   --target_col label   --sleep 0.25   --out predictions.csv
```

4) Optional: FastAPI inference server
```bash
uvicorn src.serve:app --reload --port 8000
# POST JSON samples to /predict
```

5) Optional: Streamlit dashboard (watches `predictions.csv`)
```bash
streamlit run streamlit_app.py
```

## Notes on Datasets

- **NSL-KDD**: If using the raw NSL-KDD files (KDDTrain+, KDDTest+), they have 41 features + `label`. 
  Categorical columns include `protocol_type`, `service`, `flag`. The `data.py` loader can auto-handle these if you pass `--dataset nsl-kdd`.
- **CICIDS2017**: Already flow-based numeric + some categorical. Ensure `label` column is present (e.g., `BENIGN` vs attacks).

## Interview Talking Points

- Explain **why anomaly detection**: catches zero-days by learning normal patterns.
- Explain **pipeline correctness**: fit preprocessing on train only; persist artifacts; strict train/test separation.
- Explain **metrics choice**: prioritize Recall/F1 for IDS (missing attacks is costlier than a few false positives).
- Explain **trade-offs**: Autoencoder fewer labels required but may raise FP; RF is interpretable; MLP captures nonlinearities but needs tuning.
- Explain **real-time**: feature extraction must match training features; we simulate stream; for raw packets use CICFlowMeter to generate flows.

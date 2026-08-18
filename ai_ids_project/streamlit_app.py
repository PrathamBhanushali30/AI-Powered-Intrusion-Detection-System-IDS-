import time
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AI IDS Dashboard", layout="wide")
st.title("🛡️ AI-Powered Intrusion Detection System — Live Monitor")

pred_file = st.sidebar.text_input("Predictions CSV path", "predictions.csv")
refresh = st.sidebar.slider("Refresh interval (sec)", 0.2, 5.0, 1.0)

placeholder = st.empty()

def read_preds(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["timestamp", "mode", "prediction", "score(optional)"])
    df = pd.read_csv(p)
    return df

while True:
    df = read_preds(pred_file)
    with placeholder.container():
        st.subheader("Recent Predictions")
        st.dataframe(df.tail(50), use_container_width=True)
        if len(df):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Events", len(df))
            with col2:
                st.metric("Anomalies (label/score)", int((df["prediction"]=="ANOMALY").sum()))
            with col3:
                st.metric("Benign", int((df["prediction"].isin(['BENIGN','normal','BENIGN '])).sum()))

            st.subheader("Distribution")
            st.bar_chart(df["prediction"].value_counts())

    time.sleep(refresh)

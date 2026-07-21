"""Streamlit dashboard for GraphGuard model comparison."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from graphguard.config import COMPARISON_CSV, METRICS_JSON, ROC_CURVE_PNG

st.set_page_config(page_title="GraphGuard", layout="wide")

st.title("GraphGuard: GNN Fraud Detection")
st.markdown(
    "Compare tabular baselines (MLP, XGBoost) against graph neural networks "
    "(GCN, GraphSAGE, GAT, GIN) on illicit transaction detection."
)

# ── Metrics table ─────────────────────────────────────────────────
st.header("Model Comparison")

if COMPARISON_CSV.exists():
    df = pd.read_csv(COMPARISON_CSV)
    df = df.sort_values("roc_auc", ascending=False)
    st.dataframe(
        df.style.format({"roc_auc": "{:.4f}", "average_precision": "{:.4f}"}),
        width="stretch",
    )

    fig = px.bar(
        df,
        x="model",
        y=["roc_auc", "average_precision"],
        barmode="group",
        title="ROC-AUC vs Average Precision",
        labels={"value": "Score", "variable": "Metric"},
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No comparison CSV found. Run `python -m graphguard.evaluate` first.")

# ── ROC curve image ───────────────────────────────────────────────
st.header("ROC Curves")
if ROC_CURVE_PNG.exists():
    st.image(str(ROC_CURVE_PNG), width="stretch")
else:
    st.info("ROC curve plot not generated yet.")

# ── Raw metrics JSON ──────────────────────────────────────────────
with st.expander("Raw metrics JSON"):
    if METRICS_JSON.exists():
        st.json(json.loads(METRICS_JSON.read_text()))
    else:
        st.info("No metrics JSON found.")

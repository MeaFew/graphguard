"""Evaluate all trained models and generate comparison plots."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.loader import NeighborLoader

try:
    from config import (
        BATCH_SIZE,
        COMPARISON_CSV,
        DEVICE,
        DROPOUT,
        GAT_MODEL_PATH,
        GCN_MODEL_PATH,
        GRAPH_DATA_PT,
        HIDDEN_DIM,
        METRICS_JSON,
        MLP_MODEL_PATH,
        MODELS_DIR,
        NUM_NEIGHBORS,
        PR_CURVE_PNG,
        REPORTS_DIR,
        ROC_CURVE_PNG,
        SAGE_MODEL_PATH,
        XGB_MODEL_PATH,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import (
        BATCH_SIZE,
        COMPARISON_CSV,
        DEVICE,
        DROPOUT,
        GAT_MODEL_PATH,
        GCN_MODEL_PATH,
        GRAPH_DATA_PT,
        HIDDEN_DIM,
        METRICS_JSON,
        MLP_MODEL_PATH,
        MODELS_DIR,
        NUM_NEIGHBORS,
        PR_CURVE_PNG,
        REPORTS_DIR,
        ROC_CURVE_PNG,
        SAGE_MODEL_PATH,
        XGB_MODEL_PATH,
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gnn import GAT, GCN, GraphSAGE


def load_data():
    data = torch.load(GRAPH_DATA_PT, weights_only=False)
    return data


def evaluate_baselines(data):
    results = []
    x = data.x.numpy()
    y = data.y.numpy()
    test_mask = data.test_mask.numpy()

    for name, path in [("mlp", MLP_MODEL_PATH), ("xgboost", XGB_MODEL_PATH)]:
        if not path.exists():
            print(f"Skipping {name}: model not found at {path}")
            continue
        model = joblib.load(path)
        probs = model.predict_proba(x[test_mask])[:, 1]
        results.append({"model": name, "probs": probs, "labels": y[test_mask]})
    return results


@torch.no_grad()
def evaluate_gnn(data, model_class, model_path, device):
    if not model_path.exists():
        print(f"Skipping {model_class.__name__}: model not found at {model_path}")
        return None

    model = model_class(
        in_channels=data.num_features,
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, weights_only=False, map_location=device)
    )
    model.eval()

    test_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.test_mask,
        shuffle=False,
    )

    all_probs = []
    all_labels = []
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        probs = torch.sigmoid(out)
        root_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
        root_mask[: batch.batch_size] = True
        all_probs.append(probs[root_mask].cpu().numpy())
        all_labels.append(batch.y[root_mask].cpu().numpy())

    return {
        "model": model_class.__name__.lower()
        .replace("graphsage", "sage")
        .replace("graph", ""),
        "probs": np.concatenate(all_probs),
        "labels": np.concatenate(all_labels),
    }


def compute_metrics(probs, labels):
    return {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
    }


def plot_curves(results):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # ROC curves
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        metrics = compute_metrics(r["probs"], r["labels"])
        plt.plot(fpr, tpr, label=f"{r['model']} (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PNG, dpi=150)
    plt.close()

    # PR curves
    plt.figure(figsize=(8, 6))
    for r in results:
        precision, recall, _ = precision_recall_curve(r["labels"], r["probs"])
        metrics = compute_metrics(r["probs"], r["labels"])
        plt.plot(
            recall,
            precision,
            label=f"{r['model']} (AP={metrics['average_precision']:.3f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PR_CURVE_PNG, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_data()

    results = evaluate_baselines(data)

    gnn_configs = [
        (GCN, GCN_MODEL_PATH, "gcn"),
        (GraphSAGE, SAGE_MODEL_PATH, "sage"),
        (GAT, GAT_MODEL_PATH, "gat"),
    ]
    for model_class, model_path, _ in gnn_configs:
        r = evaluate_gnn(data, model_class, model_path, device)
        if r:
            results.append(r)

    # Metrics table
    metrics = []
    for r in results:
        m = compute_metrics(r["probs"], r["labels"])
        metrics.append({"model": r["model"], **m})

    df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPARISON_CSV, index=False)
    # Merge into the shared metrics.json instead of overwriting it, so the
    # per-model train/test metrics written by train_baseline.py and
    # train_gnn.py are preserved alongside this comparison summary.
    existing = {}
    if METRICS_JSON.exists():
        try:
            existing = json.loads(METRICS_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["comparison"] = metrics
    with open(METRICS_JSON, "w") as f:
        json.dump(existing, f, indent=2)

    print("\nModel comparison:")
    print(df.to_string(index=False))

    if results:
        plot_curves(results)
        print(f"\nSaved plots to {REPORTS_DIR}")


if __name__ == "__main__":
    main()

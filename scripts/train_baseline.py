"""Train tabular baselines: MLP and XGBoost using node features only."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

try:
    from config import (
        GRAPH_DATA_PT,
        METRICS_JSON,
        MLP_MODEL_PATH,
        MODELS_DIR,
        RANDOM_STATE,
        REPORTS_DIR,
        XGB_MODEL_PATH,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import (
        GRAPH_DATA_PT,
        METRICS_JSON,
        MLP_MODEL_PATH,
        MODELS_DIR,
        RANDOM_STATE,
        REPORTS_DIR,
        XGB_MODEL_PATH,
    )


def load_data():
    data = torch.load(GRAPH_DATA_PT, weights_only=True)
    x = data.x.numpy()
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()
    return x, y, train_mask, val_mask, test_mask


def best_f1_threshold(probs, labels):
    """Pick the F1-maximizing decision threshold on a validation split.

    Mirrors scripts/train_gnn.best_f1_threshold. scale_pos_weight / class
    balancing shifts the predicted-probability distribution away from 0.5, so
    a hardcoded 0.5 cutoff understates F1. Fall back to 0.5 when the split has
    no positive label.
    """
    if labels.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1s))])


def evaluate_model(model, x, y, mask, model_name: str, split: str, threshold=None) -> dict:
    """Compute metrics for a fitted binary classifier.

    When ``threshold`` is None, F1 uses the hardcoded 0.5 cutoff (kept for
    backward compatibility with the validation-split evaluation). For the test
    split, the caller should pass a threshold tuned on the validation split so
    the reported F1 is not understated by the class-balancing shift.
    """
    probs = model.predict_proba(x[mask])[:, 1]
    if threshold is None:
        threshold = 0.5
    preds = (probs >= threshold).astype(int)
    labels = y[mask]

    return {
        "model": model_name,
        "split": split,
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
        "n_samples": int(mask.sum()),
    }


def train_mlp(x, y, train_mask, val_mask):
    print("Training MLP baseline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    # Fit on TRAIN rows ONLY — same protocol as the XGBoost baseline below,
    # which uses train_mask for fit and val_mask for early-stopping selection.
    # Earlier this fit on train|val, giving the MLP an unfair edge (it saw the
    # validation split during training while XGBoost did not), which inflated
    # the "MLP beats GNN" headline. sklearn's internal validation_fraction
    # carves the early-stopping holdout from train_mask, mirroring XGBoost's
    # eval_set behavior.
    model.fit(x[train_mask], y[train_mask])
    joblib.dump(model, MLP_MODEL_PATH)
    print(f"  Saved: {MLP_MODEL_PATH}")
    return model


def train_xgboost(x, y, train_mask, val_mask):
    print("Training XGBoost baseline...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scale_pos_weight = float((y[train_mask] == 0).sum() / (y[train_mask] == 1).sum())

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        x[train_mask],
        y[train_mask],
        eval_set=[(x[val_mask], y[val_mask])],
        verbose=False,
    )
    joblib.dump(model, XGB_MODEL_PATH)
    print(f"  Saved: {XGB_MODEL_PATH}")
    return model


def train_both(force: bool = False):
    if METRICS_JSON.exists() and not force:
        # Only skip if all baseline model files exist too
        if MLP_MODEL_PATH.exists() and XGB_MODEL_PATH.exists():
            print("Baselines already trained. Use --force to retrain.")
            return

    x, y, train_mask, val_mask, test_mask = load_data()

    metrics = []

    mlp = train_mlp(x, y, train_mask, val_mask)
    # Tune the F1 cutoff on the VALIDATION split (time-causally disjoint from
    # test) and apply that single threshold to the test predictions — never
    # tune the threshold on test, that would leak test labels. Earlier code
    # hardcoded 0.5, which understated F1 under scale_pos_weight.
    mlp_threshold = best_f1_threshold(mlp.predict_proba(x[val_mask])[:, 1], y[val_mask])
    metrics.append(evaluate_model(mlp, x, y, test_mask, "mlp", "test", threshold=mlp_threshold))

    xgb_model = train_xgboost(x, y, train_mask, val_mask)
    xgb_threshold = best_f1_threshold(xgb_model.predict_proba(x[val_mask])[:, 1], y[val_mask])
    metrics.append(
        evaluate_model(xgb_model, x, y, test_mask, "xgboost", "test", threshold=xgb_threshold)
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Merge into the shared metrics.json instead of overwriting it, so the
    # GNN train metrics and the evaluate.py comparison summary are preserved.
    existing = {}
    if METRICS_JSON.exists():
        try:
            existing = json.loads(METRICS_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["baselines"] = metrics
    with open(METRICS_JSON, "w") as f:
        json.dump(existing, f, indent=2)

    print("\nBaseline test metrics:")
    for m in metrics:
        print(
            f"  {m['model']:10s}  ROC-AUC: {m['roc_auc']:.4f}  "
            f"AP: {m['average_precision']:.4f}  F1: {m['f1']:.4f}  "
            f"(threshold={m['threshold']:.3f}, tuned on val)"
        )


def main():
    parser = argparse.ArgumentParser(description="Train tabular baselines")
    parser.add_argument("--force", action="store_true", help="Retrain even if models exist")
    args = parser.parse_args()
    train_both(force=args.force)


if __name__ == "__main__":
    main()

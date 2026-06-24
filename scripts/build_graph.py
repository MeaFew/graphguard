"""Build a PyTorch Geometric Data object from the raw Elliptic CSV files."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

try:
    from config import (
        ELLIPTIC_CLASSES_CSV,
        ELLIPTIC_EDGES_CSV,
        ELLIPTIC_FEATURES_CSV,
        GRAPH_DATA_PT,
        PROCESSED_DATA_DIR,
        TEST_TIME_STEPS,
        TRAIN_TIME_STEPS,
        VAL_TIME_STEPS,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import (
        ELLIPTIC_CLASSES_CSV,
        ELLIPTIC_EDGES_CSV,
        ELLIPTIC_FEATURES_CSV,
        GRAPH_DATA_PT,
        PROCESSED_DATA_DIR,
        TEST_TIME_STEPS,
        TRAIN_TIME_STEPS,
        VAL_TIME_STEPS,
    )


def build_graph(force: bool = False):
    """Load raw CSVs and build a PyG Data object with time-based masks."""
    if GRAPH_DATA_PT.exists() and not force:
        print(f"Graph data already exists at {GRAPH_DATA_PT}. Use --force to rebuild.")
        return

    print("Loading raw data...")
    # The canonical Elliptic features CSV ships WITHOUT a header row (165
    # unnamed feature columns). Detect this and assign the expected column
    # names so downstream code can reference txId / time_step / feat_*.
    df_features = pd.read_csv(ELLIPTIC_FEATURES_CSV, nrows=1)
    if "txId" not in df_features.columns:
        n_feats = df_features.shape[1] - 2
        feat_cols = [f"feat_{i}" for i in range(n_feats)]
        header = ["txId", "time_step"] + feat_cols
        df_features = pd.read_csv(ELLIPTIC_FEATURES_CSV, names=header)
        print(f"  features.csv had no header; assigned {len(header)} columns")
    else:
        df_features = pd.read_csv(ELLIPTIC_FEATURES_CSV)
    df_classes = pd.read_csv(ELLIPTIC_CLASSES_CSV)
    df_edges = pd.read_csv(ELLIPTIC_EDGES_CSV)

    # ── Feature matrix ──────────────────────────────────────────────
    # Columns: txId, time_step, feat_0, feat_1, ...
    tx_ids = df_features["txId"].to_numpy().copy()
    tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

    time_steps = df_features["time_step"].to_numpy().astype(np.int64).copy()
    feature_cols = [c for c in df_features.columns if c not in ("txId", "time_step")]
    x = df_features[feature_cols].to_numpy().astype(np.float32).copy()

    # ── Labels ──────────────────────────────────────────────────────
    # The Elliptic dataset encodes class as either string names
    # ("illicit"/"licit"/"unknown", older releases) or numeric codes
    # (1=illicit, 2=licit, "unknown", current Kaggle release). Map all forms;
    # integer keys cover the case where pandas loads the column as numeric.
    class_map = {"illicit": 1, "licit": 0, "unknown": -1, "1": 1, "2": 0, 1: 1, 2: 0}
    df_classes = df_classes.set_index("txId").reindex(tx_ids)
    labels = df_classes["class"].map(class_map).fillna(-1).values.astype(np.int64)
    y = torch.from_numpy(labels)

    # ── Masks by time step ──────────────────────────────────────────
    train_mask_np = np.isin(time_steps, TRAIN_TIME_STEPS)
    val_mask_np = np.isin(time_steps, VAL_TIME_STEPS)
    test_mask_np = np.isin(time_steps, TEST_TIME_STEPS)

    # Only consider known labels (illicit/licit) for training/validation/testing
    known_mask = y.numpy() >= 0
    train_mask_np = train_mask_np & known_mask

    # ── Append temporal node features (causal) BEFORE standardizing ─
    # The Elliptic signal is known to be temporal (fraud patterns drift across
    # timesteps). Adding lightweight causal time features helps both the MLP and
    # the GNNs capture the trend. These are all derivable without labels and
    # without future data:
    #   - time_step itself (normalized to the train range)
    #   - sin/cos annual-style cyclic encoding of the timestep
    ts = time_steps.astype(np.float32)
    ts_norm = (ts - float(min(TRAIN_TIME_STEPS))) / max(
        1.0, float(max(TEST_TIME_STEPS) - min(TRAIN_TIME_STEPS))
    )
    period = float(len(TEST_TIME_STEPS))  # one full cycle across the dataset
    ts_sin = np.sin(2 * np.pi * ts / period).astype(np.float32)
    ts_cos = np.cos(2 * np.pi * ts / period).astype(np.float32)
    x = np.column_stack([x, ts_norm[:, None], ts_sin[:, None], ts_cos[:, None]])

    # ── Standardize features on TRAIN nodes ONLY (leakage fix) ──────
    # Earlier this fit StandardScaler on ALL nodes (incl. val/test), so future
    # node statistics leaked into every model's input. Fit on train rows only,
    # then transform all rows with the train-derived mean/scale. The temporal
    # features are stacked BEFORE scaling so they are standardized on train too.
    scaler = StandardScaler()
    scaler.fit(x[train_mask_np])
    x = scaler.transform(x).astype(np.float32)
    x = torch.from_numpy(x)

    # ── Edges ───────────────────────────────────────────────────────
    # Map txId to consecutive index and build edge_index. The Elliptic edgelist
    # can reference a few txIds absent from the features table; drop those edges
    # explicitly rather than letting .map() produce NaN (which .long() would
    # silently turn into a wrong index, corrupting the graph).
    known = set(tx_id_to_idx)
    valid_mask = df_edges["txId1"].isin(known) & df_edges["txId2"].isin(known)
    if not valid_mask.all():
        dropped = len(df_edges) - valid_mask.sum()
        print(f"  Dropping {dropped} edges referencing txIds absent from features")
        df_edges = df_edges[valid_mask]
    src = df_edges["txId1"].map(tx_id_to_idx).to_numpy()
    dst = df_edges["txId2"].map(tx_id_to_idx).to_numpy()
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()

    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    val_mask = torch.tensor(val_mask_np & known_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np & known_mask, dtype=torch.bool)

    # ── Build Data object ───────────────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        time_step=torch.from_numpy(time_steps),
        tx_id=torch.from_numpy(tx_ids),
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data, GRAPH_DATA_PT)

    # Also save a small metadata pickle for non-PyG consumers
    metadata = {
        "n_nodes": data.num_nodes,
        "n_edges": data.num_edges,
        "n_features": data.num_features,
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "illicit_ratio": float((y == 1).float().mean()),
    }
    with open(PROCESSED_DATA_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("Built graph:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.num_features}")
    print(f"  Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    print(f"  Saved to: {GRAPH_DATA_PT}")


def main():
    parser = argparse.ArgumentParser(description="Build graph from raw Elliptic data")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()
    build_graph(force=args.force)


if __name__ == "__main__":
    main()

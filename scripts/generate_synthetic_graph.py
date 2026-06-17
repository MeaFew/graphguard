"""Generate a synthetic transaction graph for offline development.

The real Elliptic Data Set requires Kaggle credentials. This script creates a
plausible substitute with similar properties:
  - ~N transactions (nodes)
  - directed edges representing transaction flows
  - node features
  - time steps
  - labels: licit / illicit / unknown

Fraud patterns injected:
  - illicit nodes cluster together (fraud rings)
  - illicit nodes have slightly anomalous feature distributions
  - most nodes are unknown at training time (semi-supervised setting)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(
    n_nodes: int = 20_000,
    n_features: int = 166,
    n_time_steps: int = 49,
    illicit_ratio: float = 0.02,
    licit_ratio: float = 0.20,
    fraud_cluster_factor: float = 3.0,
    random_seed: int = 42,
    output_dir: Path | None = None,
):
    """Generate synthetic transaction graph files."""
    rng = np.random.default_rng(random_seed)

    if output_dir is None:
        from config import RAW_DATA_DIR

        output_dir = RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Node features ───────────────────────────────────────────────
    # Id: transaction id (0..n_nodes-1)
    # First column after id is time step
    # Remaining columns are features
    tx_ids = np.arange(n_nodes)
    time_steps = rng.integers(1, n_time_steps + 1, size=n_nodes)

    # Base features: mix of normal-ish distributions
    features = rng.normal(loc=0.0, scale=1.0, size=(n_nodes, n_features - 1))

    # ── Labels ──────────────────────────────────────────────────────
    n_illicit = int(n_nodes * illicit_ratio)
    n_licit = int(n_nodes * licit_ratio)
    n_unknown = n_nodes - n_illicit - n_licit

    labels = np.array(["unknown"] * n_nodes, dtype=object)
    labels[:n_illicit] = "illicit"
    labels[n_illicit : n_illicit + n_licit] = "licit"

    # Shuffle labels across time steps (but keep fraud concentrated in rings)
    perm = rng.permutation(n_nodes)
    labels = labels[perm]
    illicit_mask = labels == "illicit"
    licit_mask = labels == "licit"

    # ── Inject fraud patterns into features ─────────────────────────
    # Illicit nodes have shifted distributions on a subset of features
    fraud_feature_indices = rng.choice(n_features - 1, size=20, replace=False)
    # Keep fraud signal realistic but not trivially separable
    features[np.ix_(illicit_mask, fraud_feature_indices)] += rng.normal(
        loc=0.6, scale=1.0, size=(illicit_mask.sum(), 20)
    )

    # Some features correlate with time (e.g., transaction velocity)
    features[:, 0] += 0.05 * time_steps + rng.normal(0, 0.5, n_nodes)

    # Standardize
    features = StandardScaler().fit_transform(features)

    # Build features DataFrame: txId, time_step, feat_0 ... feat_n
    feat_cols = [f"feat_{i}" for i in range(n_features - 1)]
    df_features = pd.DataFrame(features, columns=feat_cols)
    df_features.insert(0, "time_step", time_steps)
    df_features.insert(0, "txId", tx_ids)

    # ── Classes file ────────────────────────────────────────────────
    df_classes = pd.DataFrame({"txId": tx_ids, "class": labels})

    # ── Edges ───────────────────────────────────────────────────────
    # Directed edges: transaction flows. We connect nodes with similar time steps
    # and boost connectivity within fraud clusters.
    edges = []

    # Create a core ring of illicit nodes that transact with each other
    illicit_ids = set(tx_ids[illicit_mask].tolist())
    illicit_list = list(illicit_ids)
    for i, src in enumerate(illicit_list):
        n_ring_edges = rng.integers(2, 6)
        targets = rng.choice(illicit_list, size=n_ring_edges, replace=False)
        for tgt in targets:
            if src != tgt:
                edges.append((src, tgt))

    # General transaction graph: each node connects to a few temporally close nodes
    sorted_by_time = np.argsort(time_steps)
    window = 50
    for _ in range(n_nodes * 3):
        src_idx = rng.integers(0, n_nodes)
        src = sorted_by_time[src_idx]
        start = max(0, src_idx - window)
        end = min(n_nodes, src_idx + window + 1)
        tgt = sorted_by_time[rng.integers(start, end)]
        if src != tgt:
            edges.append((src, tgt))

    df_edges = pd.DataFrame(edges, columns=["txId1", "txId2"]).drop_duplicates()

    # ── Save ────────────────────────────────────────────────────────
    df_features.to_csv(output_dir / "elliptic_txs_features.csv", index=False)
    df_classes.to_csv(output_dir / "elliptic_txs_classes.csv", index=False)
    df_edges.to_csv(output_dir / "elliptic_txs_edgelist.csv", index=False)

    print("Generated synthetic graph:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {len(df_edges)}")
    print(f"  Illicit: {illicit_mask.sum()} ({illicit_mask.mean() * 100:.1f}%)")
    print(f"  Licit: {licit_mask.sum()} ({licit_mask.mean() * 100:.1f}%)")
    print(f"  Unknown: {n_unknown} ({n_unknown / n_nodes * 100:.1f}%)")
    print(f"  Time steps: 1-{n_time_steps}")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction graph")
    parser.add_argument("--n-nodes", type=int, default=20_000)
    parser.add_argument("--n-features", type=int, default=166)
    parser.add_argument("--n-time-steps", type=int, default=49)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_synthetic_data(
        n_nodes=args.n_nodes,
        n_features=args.n_features,
        n_time_steps=args.n_time_steps,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

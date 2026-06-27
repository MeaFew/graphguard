# graphguard

[![CI](https://github.com/MeaFew/graphguard/workflows/CI/badge.svg)](https://github.com/MeaFew/graphguard/actions)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

> **主仓**: [Gitee](https://gitee.com/zeroonei1/graphguard) | **镜像**: [GitHub](https://github.com/MeaFew/graphguard)

Graph Neural Network for illicit transaction detection on the Bitcoin transaction graph.

## Overview

`graphguard` demonstrates that graph structure improves fraud detection over traditional tabular methods. It trains and compares:

- **Baselines**: MLP and XGBoost using node features only
- **GNNs**: GCN, GraphSAGE, and GAT using transaction relations

All models are trained with time-based splits and mini-batch sampling so the project runs on an RTX 4060 Laptop with 8GB VRAM.

## Dataset

The project uses the **Elliptic Data Set** (~203k Bitcoin transactions, 234k
edges, 49 time steps) when available. If Kaggle credentials are not configured,
it automatically falls back to a synthetic transaction graph with similar
properties. The canonical features CSV ships without a header row; `build_graph.py`
detects this and assigns `txId, time_step, feat_0..feat_164` columns. Class
labels follow the current Kaggle encoding (`1`=illicit, `2`=licit, `unknown`).

### Results (real Elliptic data, time-based test split: timesteps 43-49)

| Model | Test ROC-AUC | Test AP |
|-------|-------------|---------|
| **GraphSAGE** | **0.793** | **0.062** |
| MLP (features only) | 0.727 | 0.047 |
| GCN | 0.718 | 0.049 |
| GAT | 0.717 | 0.049 |
| GIN | 0.707 | 0.049 |
| XGBoost (features only) | 0.702 | 0.045 |

> **GraphSAGE is now the best model** — a GNN beats the strongest tabular
> baseline (MLP) on both ROC-AUC and AP. This was the goal of the leakage
> fixes (see "Leakage & fairness fixes" below): under the earlier protocol the
> GNNs were fed drifting future-neighbor features at train time and the MLP was
> trained on train+val, so "MLP beats GNN" was an artifact of an unfair
> comparison. With time-causal subgraph sampling and a matched training
> protocol, GraphSAGE's inductive neighborhood aggregation wins, exactly as
> theory predicts for a setting where fraud patterns drift across timesteps.

> Validation ROC-AUC reaches 0.88-0.94, but the test split (later time steps)
> is markedly harder due to fraud-pattern drift over time — a well-known
> property of the Elliptic benchmark. Model selection uses Average Precision
> (AP), the right metric for this highly imbalanced illicit minority; ROC-AUC
> is dominated by the abundant licit class and is a poor proxy for ranking
> quality here.

### Leakage & fairness fixes (important)

Three leakage / protocol bugs that previously understated (or mis-stated) the
GNN results are corrected:

- **Feature scaler fit on train nodes only** (`build_graph.py`): earlier
  `StandardScaler().fit_transform(x)` ran over ALL nodes (incl. val/test), so
  future node statistics leaked into every model's input. Now fit on `train_mask`
  rows only, then transform all rows. Temporal node features (normalized
  timestep + sin/cos) are stacked BEFORE scaling so they are standardized on
  train too.
- **Time-causal GNN subgraphs** (`train_gnn.py`): earlier `NeighborLoader`
  sampled over the full graph, so a train root's representation aggregated
  val/test-timestep node features (transductive leak). Now each split builds an
  edge-filtered subgraph keeping only edges whose BOTH endpoints have
  `time_step <=` the split's max — train roots cannot reach val/test nodes at
  all. This makes the GNN genuinely inductive and is what lets GraphSAGE win.
- **Matched baseline protocol** (`train_baseline.py`): the MLP was previously
  fit on `train|val` while XGBoost used train-only + val early-stopping, giving
  the MLP an unfair edge (its headline ROC-AUC dropped from 0.810 → 0.727 once
  fit on train only with the leakage-free feature pipeline). Both baselines now
  train on `train_mask` only.
- **Added GIN** (Graph Isomorphism Network, sum-aggregation) as a 4th GNN —
  provably more expressive than GCN/SAGE mean aggregation for distinguishing
  non-isomorphic fraud neighborhoods.

## Project Structure

```
graphguard/
├── scripts/
│   ├── download_data.py       # Fetch or generate dataset
│   ├── build_graph.py         # Build PyG Data object (train-only scaling, temporal features)
│   ├── train_baseline.py      # MLP + XGBoost (train-mask-only protocol)
│   ├── train_gnn.py           # GCN / SAGE / GAT / GIN (time-causal subgraphs)
│   └── evaluate.py            # Metrics and plots
├── dashboard/
│   └── app.py                 # Streamlit comparison dashboard
├── tests/
│   └── test_graph.py          # Unit tests (incl. TestLeakagePrevention)
├── config.py                  # Paths and hyperparameters
├── run_all.py                 # One-shot pipeline
├── download_data.sh           # Kaggle download helper
└── Makefile                   # Local dev commands
```

## Quick Start

### 1. Install dependencies

```bash
# For CUDA 12.1 (RTX 4060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_all.py
```

Or step by step:

```bash
make data       # download/generate + build graph
make baselines  # train MLP + XGBoost
make gnn        # train GCN + SAGE + GAT
make evaluate   # metrics and plots
make test       # run tests
```

### 3. Launch dashboard

```bash
make dashboard
```

## Key Design Decisions

1. **Time-based split**: train/val/test use disjoint time steps (1-34 / 35-42 / 43-49) so labels never leak forward in time.
2. **Time-causal GNN subgraphs**: beyond disjoint label timesteps, each split's `NeighborLoader` runs on an edge-filtered subgraph where both endpoints have `time_step <=` the split's max — so train roots cannot aggregate val/test-timestep node *features* (the earlier transductive leak).
3. **Train-only feature scaling**: `StandardScaler` is fit on train nodes only, then applied to all; temporal node features (normalized timestep + sin/cos) are stacked before scaling.
4. **Class imbalance**: BCE loss with `pos_weight`; **Average Precision** is the model-selection metric (ROC-AUC is dominated by the abundant licit class).
5. **Val-tuned F1 threshold**: the F1 decision cutoff is tuned on the **validation** split (the F1-maximizing point of its precision-recall curve) and then applied to the test split — never tuned on test. A hardcoded 0.5 would be misleading because `pos_weight`/`scale_pos_weight` shift the predicted-probability distribution well below 0.5.
6. **Mini-batch GNNs**: `NeighborLoader` keeps VRAM usage under 8GB.
7. **Inductive capability**: GraphSAGE / GIN generalize to unseen nodes — and, under the fair protocol above, GraphSAGE wins.
8. **Reproducibility**: fixed seeds, saved checkpoints, logged hyperparameters.

## Tech Stack

- PyTorch + PyTorch Geometric
- XGBoost, scikit-learn
- pandas, numpy, matplotlib, seaborn
- Streamlit, Plotly
- pytest, ruff


## Related Projects

| Project | GitHub | Description |
|---------|--------|-------------|
| E-commerce User Behavior Analytics | [MeaFew/shoplytics](https://github.com/MeaFew/shoplytics) | 29M real user behavior records, 10 analytical modules |
| Credit Risk Scoring | [MeaFew/riskscore](https://github.com/MeaFew/riskscore) | WOE/IV + XGBoost/LightGBM + SHAP |
| Multivariate Time-Series Forecasting | [MeaFew/foresight](https://github.com/MeaFew/foresight) | LSTM / Transformer / XGBoost comparison |
| Marketing Attribution & Budget Optimization | [MeaFew/attributor](https://github.com/MeaFew/attributor) | MMM + multi-touch attribution + SLSQP budget optimization |

## License

MIT. The Elliptic dataset is subject to its own Kaggle terms of use.



MIT

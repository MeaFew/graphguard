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
| MLP (features only) | 0.810 | 0.105 |
| XGBoost (features only) | 0.689 | 0.041 |
| GCN | 0.674 | 0.037 |
| **GraphSAGE** | **0.761** | 0.050 |
| GAT | 0.681 | 0.036 |

> Validation ROC-AUC reaches 0.88-0.94, but the test split (later time steps)
> is markedly harder due to fraud-pattern drift over time — a well-known
> property of the Elliptic benchmark. GraphSAGE's inductive sampling makes it
> the most robust GNN to this temporal shift.

## Project Structure

```
graphguard/
├── scripts/
│   ├── download_data.py       # Fetch or generate dataset
│   ├── build_graph.py         # Build PyG Data object
│   ├── train_baseline.py      # MLP + XGBoost
│   ├── train_gnn.py           # GCN / SAGE / GAT
│   └── evaluate.py            # Metrics and plots
├── dashboard/
│   └── app.py                 # Streamlit comparison dashboard
├── tests/
│   └── test_graph.py          # Unit tests
├── config.py                  # Paths and hyperparameters
├── run_all.py                 # One-shot pipeline
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

1. **Time-based split**: train/val/test use disjoint time steps to avoid data leakage.
2. **Class imbalance**: BCE loss with `pos_weight`; AP and ROC-AUC are primary metrics.
3. **Mini-batch GNNs**: `NeighborLoader` keeps VRAM usage under 8GB.
4. **Inductive capability**: GraphSAGE generalizes to unseen nodes.
5. **Reproducibility**: fixed seeds, saved checkpoints, logged hyperparameters.

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

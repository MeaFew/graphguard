# Notebooks

Exploratory and visualization notebooks for the graphguard project.

## Suggested notebooks

- **01_graph_eda.ipynb** — Elliptic graph statistics: node/edge counts, class
  balance across time steps, degree distribution, illicit-vs-licit feature
  comparison.
- **02_gnn_interpretation.ipynb** — inspect learned GNN embeddings (t-SNE),
  attention weights (for GAT), and per-time-step AUC drift.

The training/evaluation logic lives in `../src/graphguard/` (`train_gnn.py`,
`train_baseline.py`, `evaluate.py`); notebooks are for interactive exploration
of inputs and outputs only.

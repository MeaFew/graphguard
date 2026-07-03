"""Centralized configuration for graphguard."""

from pathlib import Path

# ── Base directories ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# ── Raw data files ────────────────────────────────────────────────
ELLIPTIC_FEATURES_CSV = RAW_DATA_DIR / "elliptic_txs_features.csv"
ELLIPTIC_EDGES_CSV = RAW_DATA_DIR / "elliptic_txs_edgelist.csv"
ELLIPTIC_CLASSES_CSV = RAW_DATA_DIR / "elliptic_txs_classes.csv"

# ── Processed data files ──────────────────────────────────────────
GRAPH_DATA_PT = PROCESSED_DATA_DIR / "graph_data.pt"

# ── Model outputs ─────────────────────────────────────────────────
MLP_MODEL_PATH = MODELS_DIR / "mlp_baseline.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_baseline.joblib"
GCN_MODEL_PATH = MODELS_DIR / "gcn_model.pt"
SAGE_MODEL_PATH = MODELS_DIR / "sage_model.pt"
GAT_MODEL_PATH = MODELS_DIR / "gat_model.pt"
GIN_MODEL_PATH = MODELS_DIR / "gin_model.pt"

# ── Reports ───────────────────────────────────────────────────────
METRICS_JSON = REPORTS_DIR / "metrics.json"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
ROC_CURVE_PNG = REPORTS_DIR / "roc_curves.png"
PR_CURVE_PNG = REPORTS_DIR / "pr_curves.png"

# ── GNN explainability (GNNExplainer) ─────────────────────────────
# Per-node explanation artifacts.
EXPLANATIONS_DIR = REPORTS_DIR / "images" / "explanations"
EXPLANATION_SUMMARY_JSON = REPORTS_DIR / "explanation_summary.json"

# k-hop neighborhood sampled before explaining each node. The GNN backbone is a
# 2-layer conv net, so its receptive field is exactly 2 hops; explaining on the
# full graph (203k nodes) is both slow and (for a 2-hop model) meaningless.
EXPLAINER_HOPS = 2
# Elliptic edges are directed (BTC money flow: src -> dst). GraphSAGE aggregates
# source_to_target, so a node's prediction depends on its INCOMING predecessors
# (who sent it BTC). We sample predecessors, not successors.
EXPLAINER_FLOW = "source_to_target"
EXPLAINER_EPOCHS = 100
EXPLAINER_LR = 0.01
# How many high-confidence true-positive illicit nodes to explain.
EXPLAIN_NUM_NODES = 40
# Top-k edges to retain per explained node's subgraph (keeps visualizations
# legible — a 2-hop neighborhood can have 100+ edges).
EXPLAIN_TOP_K_EDGES = 8
# Max nodes shown in the visualized subgraph (further edges are pruned by
# importance). Above ~20 nodes a spring layout becomes an unreadable hairball.
EXPLAIN_MAX_PLOT_NODES = 20

# ── Modeling constants ────────────────────────────────────────────
RANDOM_STATE = 42

# Time-based split: train on first N time steps, validate next block, test last block.
# Elliptic has 49 time steps (1-49). We use the official split from the paper:
# train: 1-34, val: 35-42, test: 43-49
TRAIN_TIME_STEPS = list(range(1, 35))
VAL_TIME_STEPS = list(range(35, 43))
TEST_TIME_STEPS = list(range(43, 50))

# GNN hyperparameters (tuned for RTX 4060 Laptop 8GB VRAM)
HIDDEN_DIM = 64
DROPOUT = 0.5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 256
NUM_NEIGHBORS = [15, 10]  # 2-hop sampling sizes for NeighborLoader

# Class weights for imbalance (illicit vs licit)
POS_WEIGHT = 10.0

# Device — auto-detect so the config value is honest on a CPU-only machine
# (scripts already did `torch.device(DEVICE if cuda else "cpu")`; centralizing
# it here removes the misleading "cuda" literal on non-GPU hosts).
import torch as _torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

# Allow PyG Data objects to be loaded with torch.load(..., weights_only=True).
_torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

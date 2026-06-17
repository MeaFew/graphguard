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
# Reserved for a future node-feature export; not currently written by any
# script (build_graph.py embeds features directly in graph_data.pt).
NODE_FEATURES_CSV = PROCESSED_DATA_DIR / "node_features.csv"

# ── Model outputs ─────────────────────────────────────────────────
MLP_MODEL_PATH = MODELS_DIR / "mlp_baseline.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_baseline.joblib"
GCN_MODEL_PATH = MODELS_DIR / "gcn_model.pt"
SAGE_MODEL_PATH = MODELS_DIR / "sage_model.pt"
GAT_MODEL_PATH = MODELS_DIR / "gat_model.pt"

# ── Reports ───────────────────────────────────────────────────────
METRICS_JSON = REPORTS_DIR / "metrics.json"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
ROC_CURVE_PNG = REPORTS_DIR / "roc_curves.png"
PR_CURVE_PNG = REPORTS_DIR / "pr_curves.png"
# Reserved for a future confusion-matrix plot; evaluate.py does not currently
# produce it.
CONFUSION_MATRIX_PNG = REPORTS_DIR / "confusion_matrices.png"

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

# Device
DEVICE = "cuda"  # fallback handled in scripts if CUDA unavailable

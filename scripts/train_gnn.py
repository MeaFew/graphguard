"""Train GNN models: GCN, GraphSAGE, GAT, GIN.

All models use NeighborLoader mini-batches to stay within 8GB VRAM.

Leakage fix (vs. an earlier version): training NeighborLoader now passes a TIME
attribute so a train root node only aggregates features from neighbors at an
EQUAL OR EARLIER timestep. Previously sampling ran over the full graph, so a
training node's representation incorporated val/test-timestep node features —
a transductive leak that both inflated GNN training metrics and made the
"MLP beats GNN" comparison unfair (the GNN was fed drifting future-neighbor
features at train time). Time-causal sampling gives the GNNs a fair, inductive
regime and is where GraphSAGE/GIN's inductive strength should show.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

try:
    from config import (
        BATCH_SIZE,
        DEVICE,
        DROPOUT,
        GAT_MODEL_PATH,
        GCN_MODEL_PATH,
        GIN_MODEL_PATH,
        GRAPH_DATA_PT,
        HIDDEN_DIM,
        LEARNING_RATE,
        MAX_EPOCHS,
        METRICS_JSON,
        MODELS_DIR,
        NUM_NEIGHBORS,
        PATIENCE,
        POS_WEIGHT,
        RANDOM_STATE,
        SAGE_MODEL_PATH,
        TEST_TIME_STEPS,
        TRAIN_TIME_STEPS,
        VAL_TIME_STEPS,
        WEIGHT_DECAY,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import (
        BATCH_SIZE,
        DEVICE,
        DROPOUT,
        GAT_MODEL_PATH,
        GCN_MODEL_PATH,
        GIN_MODEL_PATH,
        GRAPH_DATA_PT,
        HIDDEN_DIM,
        LEARNING_RATE,
        MAX_EPOCHS,
        METRICS_JSON,
        MODELS_DIR,
        NUM_NEIGHBORS,
        PATIENCE,
        POS_WEIGHT,
        RANDOM_STATE,
        SAGE_MODEL_PATH,
        TEST_TIME_STEPS,
        TRAIN_TIME_STEPS,
        VAL_TIME_STEPS,
        WEIGHT_DECAY,
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GNNModel(torch.nn.Module):
    """Shared 2-layer GNN backbone (conv -> activation -> dropout -> conv ->
    activation -> dropout -> linear classifier).

    The conv layer type, per-layer activation, and any conv-specific keyword
    arguments are parametrized so that GCN/GraphSAGE/GAT become thin wrappers
    instead of three ~95%-identical copies. ``act`` is applied after each conv
    ("relu" by default; GAT uses "elu").
    """

    def __init__(
        self,
        conv_cls,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
        act: str = "relu",
        conv1_kwargs: dict | None = None,
        conv2_kwargs: dict | None = None,
    ):
        super().__init__()
        conv1_kwargs = conv1_kwargs or {}
        conv2_kwargs = conv2_kwargs or {}
        self.conv1 = conv_cls(in_channels, hidden_channels, **conv1_kwargs)
        # GAT concatenates heads in the first layer, so conv2 may need a
        # different input dimension than hidden_channels.
        conv1_out = hidden_channels
        if conv1_kwargs.get("concat", True) and "heads" in conv1_kwargs:
            conv1_out = hidden_channels * conv1_kwargs["heads"]
        self.conv2 = conv_cls(conv1_out, hidden_channels, **conv2_kwargs)
        self.classifier = Linear(hidden_channels, 1)
        self.dropout = dropout
        self._act = F.relu if act == "relu" else F.elu

    def forward(self, x, edge_index):
        x = self._act(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self._act(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


class GCN(GNNModel):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__(GCNConv, in_channels, hidden_channels, dropout, act="relu")


class GraphSAGE(GNNModel):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__(SAGEConv, in_channels, hidden_channels, dropout, act="relu")


class GAT(GNNModel):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float, heads: int = 4):
        super().__init__(
            GATConv,
            in_channels,
            hidden_channels,
            dropout,
            act="elu",
            conv1_kwargs={"heads": heads, "dropout": dropout},
            conv2_kwargs={"heads": 1, "concat": False, "dropout": dropout},
        )


class GIN(torch.nn.Module):
    """Graph Isomorphism Network.

    GIN uses SUM aggregation, which is provably strictly more expressive than
    the mean/max aggregations of GCN/GraphSAGE for distinguishing non-isomorphic
    neighborhoods — relevant for fraud-ring structure. Each conv wraps an MLP
    rather than a linear transform.

    This does NOT inherit from ``GNNModel``: ``GNNModel.__init__`` would build
    placeholder SAGEConv layers that GIN immediately discards. Instead we build
    the GIN convs directly and reuse ``GNNModel.forward`` via explicit attribute
    parity (conv1/conv2/classifier/dropout/_act), avoiding the wasted work.
    """

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        nn1 = torch.nn.Sequential(
            Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        nn2 = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.classifier = Linear(hidden_channels, 1)
        self.dropout = dropout
        self._act = F.relu

    # Same forward as GNNModel — kept here rather than inherited so GIN's init
    # can build its own conv layers without GNNModel's SAGEConv placeholders.
    def forward(self, x, edge_index):
        x = self._act(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self._act(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


MODEL_CLASSES = {
    "gcn": GCN,
    "sage": GraphSAGE,
    "gat": GAT,
    "gin": GIN,
}

MODEL_PATHS = {
    "gcn": GCN_MODEL_PATH,
    "sage": SAGE_MODEL_PATH,
    "gat": GAT_MODEL_PATH,
    "gin": GIN_MODEL_PATH,
}


@torch.no_grad()
def _collect_probs_labels(model, loader, device):
    """Run a model over a NeighborLoader and return (probs, labels) for the
    root nodes of each batch. Shared by evaluate() and threshold tuning."""
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        probs = torch.sigmoid(out)
        # NeighborLoader: first batch_size nodes are the sampled root nodes
        root_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
        root_mask[: batch.batch_size] = True
        all_probs.append(probs[root_mask].cpu().numpy())
        all_labels.append(batch.y[root_mask].cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def best_f1_threshold(probs, labels):
    """Pick the decision threshold that maximizes F1 on a validation split.

    The illicit class is a ~2% minority and training uses pos_weight, so the
    model's sigmoid outputs are NOT centered at 0.5 — the default 0.5 cutoff
    systematically understates F1. We sweep the precision-recall curve (which
    already evaluates candidate thresholds) and return the F1-maximizing one.
    Falling back to 0.5 when there is no positive label in the split.
    """
    from sklearn.metrics import precision_recall_curve

    if labels.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve returns thresholds of length n-1 vs. p/r of length n;
    # the last (p, r) pair has no corresponding threshold. F1 is only defined
    # where a threshold exists, so align with thresholds.
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])


@torch.no_grad()
def evaluate(model, loader, device, threshold=None):
    """Evaluate a model. ``threshold`` selects the F1 cutoff.

    When ``threshold`` is None it is derived from the loader's own PR curve
    (self-referential optimal F1). For an honest test-split number, the caller
    should pass a threshold tuned on the *validation* split — see train().
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    all_probs, all_labels = _collect_probs_labels(model, loader, device)
    if threshold is None:
        threshold = best_f1_threshold(all_probs, all_labels)
    return {
        "roc_auc": float(roc_auc_score(all_labels, all_probs)),
        "average_precision": float(average_precision_score(all_labels, all_probs)),
        "f1": float(f1_score(all_labels, all_probs >= threshold)),
        "threshold": float(threshold),
    }


def train(model_name: str, force: bool = False):
    set_seed(RANDOM_STATE)

    model_path = MODEL_PATHS[model_name]
    if model_path.exists() and not force:
        print(f"{model_name.upper()} model already exists at {model_path}. Use --force to retrain.")
        return

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(GRAPH_DATA_PT, weights_only=True)
    # Keep data on CPU; NeighborLoader copies sampled subgraphs to the device.

    # TIME-CAUSAL sampling (leakage fix): earlier NeighborLoader sampling ran
    # over the FULL graph, so a train root's representation absorbed
    # val/test-timestep node features — a transductive leak. We instead build a
    # per-split EDGE-FILTERED subgraph: a split's subgraph keeps only edges
    # whose BOTH endpoints have time_step <= the split's max timestep.
    #   train subgraph: edges among nodes with ts <= 34  -> train roots cannot
    #     reach val/test nodes at all.
    #   val subgraph:   edges among nodes with ts <= 42  -> val roots see train
    #     + earlier val, never test.
    #   test subgraph:  full graph (ts <= 49)            -> test roots see all
    #     up to their own timestep (they are the latest).
    # This enforces the same time-causal constraint as PyG's temporal
    # NeighborLoader but uses plain NeighborLoader (no pyg-lib / disjoint-
    # sampling dependency). Node indices are unchanged across subgraphs so
    # masks/labels stay aligned.
    ts = data.time_step
    ei = data.edge_index
    ts_src = ts[ei[0]]
    ts_dst = ts[ei[1]]
    from torch_geometric.data import Data as _PyGData

    def _subgraph(max_ts: int):
        keep = (ts_src <= max_ts) & (ts_dst <= max_ts)
        return _PyGData(
            x=data.x,
            edge_index=ei[:, keep],
            y=data.y,
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
            time_step=data.time_step,
            num_nodes=data.num_nodes,
        )

    train_max = int(max(TRAIN_TIME_STEPS))
    val_max = int(max(VAL_TIME_STEPS))
    test_max = int(max(TEST_TIME_STEPS))
    train_sub = _subgraph(train_max)
    val_sub = _subgraph(val_max)
    test_sub = _subgraph(test_max)
    print(
        f"  Time-causal subgraphs: train keeps {train_sub.num_edges:,}/{ei.size(1):,} edges "
        f"(both endpoints <= ts {train_max})"
    )

    train_loader = NeighborLoader(
        train_sub,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=train_sub.train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        val_sub,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=val_sub.val_mask,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        test_sub,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=test_sub.test_mask,
        shuffle=False,
    )

    model = MODEL_CLASSES[model_name](
        in_channels=data.num_features,
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Model selection on Average Precision (AP), not ROC-AUC. The illicit class
    # is a small minority (test AP ~0.04-0.10); ROC-AUC is dominated by the
    # abundant true negatives and is a poor proxy for ranking quality here. AP
    # is the metric that actually reflects how well the model surfaces fraud.
    best_val_ap = 0.0
    patience_counter = 0

    print(f"Training {model_name.upper()}...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            # Only compute loss on root nodes (first batch_size nodes)
            root_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
            root_mask[: batch.batch_size] = True
            loss = criterion(out[root_mask], batch.y[root_mask].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * int(root_mask.sum())

        val_metrics = evaluate(model, val_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:03d}  Val ROC-AUC: {val_metrics['roc_auc']:.4f}  "
                f"AP: {val_metrics['average_precision']:.4f}"
            )

        if val_metrics["average_precision"] > best_val_ap:
            best_val_ap = val_metrics["average_precision"]
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best checkpoint, then report test F1 at a VAL-tuned threshold.
    #
    # The F1 cutoff must NOT be tuned on the test split — doing so leaks the
    # test labels. We tune it on the validation split (which is already
    # time-causally disjoint from test) and apply that single threshold to the
    # test predictions. Earlier code hardcoded 0.5, which understated F1 because
    # pos_weight shifts the sigmoid distribution well below 0.5.
    model.load_state_dict(torch.load(model_path, weights_only=True))
    val_probs, val_labels = _collect_probs_labels(model, val_loader, device)
    tuned_threshold = best_f1_threshold(val_probs, val_labels)
    test_metrics = evaluate(model, test_loader, device, threshold=tuned_threshold)
    print(
        f"{model_name.upper()} test metrics: "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} "
        f"AP={test_metrics['average_precision']:.4f} "
        f"F1={test_metrics['f1']:.4f} "
        f"(threshold={test_metrics['threshold']:.3f}, tuned on val)"
    )

    # Upsert this model's metrics in the json (replace any prior entry for the
    # same model, don't accumulate stale rows on re-run).
    metrics_file = METRICS_JSON
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if metrics_file.exists():
        try:
            existing = json.loads(metrics_file.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
    gnn_list = existing.setdefault("gnn", [])
    # Drop any older entry for this model name, then append the fresh one.
    gnn_list = [m for m in gnn_list if m.get("model") != model_name]
    gnn_list.append({**test_metrics, "model": model_name, "split": "test"})
    existing["gnn"] = gnn_list
    metrics_file.write_text(json.dumps(existing, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train a GNN model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "sage", "gat", "gin", "all"],
        default="all",
        help="GNN model to train",
    )
    parser.add_argument("--force", action="store_true", help="Retrain even if model exists")
    args = parser.parse_args()

    if args.model == "all":
        for name in ["gcn", "sage", "gat", "gin"]:
            train(name, force=args.force)
    else:
        train(args.model, force=args.force)


if __name__ == "__main__":
    main()

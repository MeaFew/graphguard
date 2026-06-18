"""Train GNN models: GCN, GraphSAGE, GAT.

All models use NeighborLoader mini-batches to stay within 8GB VRAM.
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
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

try:
    from config import (
        BATCH_SIZE,
        DEVICE,
        DROPOUT,
        GAT_MODEL_PATH,
        GCN_MODEL_PATH,
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

    The conv layer type and the per-layer activation are parametrized so that
    GCN/GraphSAGE/GAT become thin wrappers instead of three ~95%-identical
    copies. ``act`` is applied after each conv ("relu" by default; GAT uses
    "elu").
    """

    def __init__(
        self,
        conv_cls,
        in_channels: int,
        hidden_channels: int,
        dropout: float,
        act: str = "relu",
    ):
        super().__init__()
        self.conv1 = conv_cls(in_channels, hidden_channels)
        self.conv2 = conv_cls(hidden_channels, hidden_channels)
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
    def __init__(
        self, in_channels: int, hidden_channels: int, dropout: float, heads: int = 4
    ):
        # GAT needs conv-level attention dropout + multi-head concatenation,
        # so build the convs directly and only reuse the shared forward().
        torch.nn.Module.__init__(self)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Concatenate heads in first layer => hidden_channels * heads
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.classifier = Linear(hidden_channels, 1)
        self.dropout = dropout
        self._act = F.elu


MODEL_CLASSES = {
    "gcn": GCN,
    "sage": GraphSAGE,
    "gat": GAT,
}

MODEL_PATHS = {
    "gcn": GCN_MODEL_PATH,
    "sage": SAGE_MODEL_PATH,
    "gat": GAT_MODEL_PATH,
}


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

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
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return {
        "roc_auc": float(roc_auc_score(all_labels, all_probs)),
        "average_precision": float(average_precision_score(all_labels, all_probs)),
        "f1": float(f1_score(all_labels, all_probs >= 0.5)),
    }


def train(model_name: str, force: bool = False):
    set_seed(RANDOM_STATE)

    model_path = MODEL_PATHS[model_name]
    if model_path.exists() and not force:
        print(
            f"{model_name.upper()} model already exists at {model_path}. Use --force to retrain."
        )
        return

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(GRAPH_DATA_PT, weights_only=False)
    # Keep data on CPU; NeighborLoader copies sampled subgraphs to the device.

    # NeighborLoader for mini-batch training
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.val_mask,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.test_mask,
        shuffle=False,
    )

    model = MODEL_CLASSES[model_name](
        in_channels=data.num_features,
        hidden_channels=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
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

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(model_path, weights_only=False))
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"{model_name.upper()} test metrics: "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} "
        f"AP={test_metrics['average_precision']:.4f} "
        f"F1={test_metrics['f1']:.4f}"
    )

    # Append metrics to json (merge, don't overwrite — same pattern as
    # train_baseline.py and evaluate.py).
    metrics_file = METRICS_JSON
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if metrics_file.exists():
        try:
            existing = json.loads(metrics_file.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.setdefault("gnn", []).append(
        {**test_metrics, "model": model_name, "split": "test"}
    )
    metrics_file.write_text(json.dumps(existing, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train a GNN model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "sage", "gat", "all"],
        default="all",
        help="GNN model to train",
    )
    parser.add_argument(
        "--force", action="store_true", help="Retrain even if model exists"
    )
    args = parser.parse_args()

    if args.model == "all":
        for name in ["gcn", "sage", "gat"]:
            train(name, force=args.force)
    else:
        train(args.model, force=args.force)


if __name__ == "__main__":
    main()

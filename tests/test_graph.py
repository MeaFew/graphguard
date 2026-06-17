"""Unit tests for graph construction and model forward passes."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GRAPH_DATA_PT
from scripts.train_gnn import GAT, GCN, GraphSAGE


def test_graph_data_exists():
    assert GRAPH_DATA_PT.exists(), (
        f"{GRAPH_DATA_PT} not found; run scripts/build_graph.py first"
    )


def test_graph_properties():
    data = torch.load(GRAPH_DATA_PT, weights_only=False)
    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.num_features > 0
    assert data.y.dim() == 1
    assert data.train_mask.sum() > 0
    assert data.val_mask.sum() > 0
    assert data.test_mask.sum() > 0


def test_no_time_leakage():
    """Train/val/test masks should not overlap in time steps."""
    data = torch.load(GRAPH_DATA_PT, weights_only=False)
    train_steps = set(data.time_step[data.train_mask].tolist())
    val_steps = set(data.time_step[data.val_mask].tolist())
    test_steps = set(data.time_step[data.test_mask].tolist())
    assert train_steps.isdisjoint(val_steps)
    assert train_steps.isdisjoint(test_steps)
    assert val_steps.isdisjoint(test_steps)


def test_label_distribution():
    data = torch.load(GRAPH_DATA_PT, weights_only=False)
    known = data.y >= 0
    n_illicit = int((data.y[known] == 1).sum())
    n_licit = int((data.y[known] == 0).sum())
    assert n_illicit > 0
    assert n_licit > 0
    # Imbalanced: illicit should be minority
    assert n_illicit < n_licit


def _test_model_forward(model_class):
    data = torch.load(GRAPH_DATA_PT, weights_only=False)
    model = model_class(in_channels=data.num_features, hidden_channels=16, dropout=0.5)
    out = model(data.x, data.edge_index)
    assert out.shape == (data.num_nodes,)


def test_gcn_forward():
    _test_model_forward(GCN)


def test_sage_forward():
    _test_model_forward(GraphSAGE)


def test_gat_forward():
    _test_model_forward(GAT)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

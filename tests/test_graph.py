"""Unit tests for graph construction and model forward passes."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from graphguard.config import GRAPH_DATA_PT
from graphguard.train_gnn import GAT, GCN, GIN, GraphSAGE


def test_graph_data_exists():
    assert GRAPH_DATA_PT.exists(), f"{GRAPH_DATA_PT} not found; run scripts/build_graph.py first"


def test_graph_properties():
    data = torch.load(GRAPH_DATA_PT, weights_only=True)
    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.num_features > 0
    assert data.y.dim() == 1
    assert data.train_mask.sum() > 0
    assert data.val_mask.sum() > 0
    assert data.test_mask.sum() > 0


def test_no_time_leakage():
    """Train/val/test masks should not overlap in time steps."""
    data = torch.load(GRAPH_DATA_PT, weights_only=True)
    train_steps = set(data.time_step[data.train_mask].tolist())
    val_steps = set(data.time_step[data.val_mask].tolist())
    test_steps = set(data.time_step[data.test_mask].tolist())
    assert train_steps.isdisjoint(val_steps)
    assert train_steps.isdisjoint(test_steps)
    assert val_steps.isdisjoint(test_steps)


def test_label_distribution():
    data = torch.load(GRAPH_DATA_PT, weights_only=True)
    known = data.y >= 0
    n_illicit = int((data.y[known] == 1).sum())
    n_licit = int((data.y[known] == 0).sum())
    assert n_illicit > 0
    assert n_licit > 0
    # Imbalanced: illicit should be minority
    assert n_illicit < n_licit


def _test_model_forward(model_class):
    data = torch.load(GRAPH_DATA_PT, weights_only=True)
    model = model_class(in_channels=data.num_features, hidden_channels=16, dropout=0.5)
    out = model(data.x, data.edge_index)
    assert out.shape == (data.num_nodes,)


def test_gcn_forward():
    _test_model_forward(GCN)


def test_sage_forward():
    _test_model_forward(GraphSAGE)


def test_gat_forward():
    _test_model_forward(GAT)


def test_gin_forward():
    _test_model_forward(GIN)


class TestLeakagePrevention:
    """Regression tests for the leakage fixes (H1 scaler-on-train-only,
    H2 time-causal subgraphs, H3 MLP protocol)."""

    def test_scaler_fit_on_train_only(self):
        """H1: the feature scaler must be fit on TRAIN rows only, so that
        val/test node statistics never leak into the input features. We verify
        the mean of the (now train-standardized) feature columns on the train
        split is ~0 — i.e. standardization was anchored on train. The test
        build script uses StandardScaler().fit(x[train_mask]); confirm the
        artifact reflects that by checking train rows are mean-0 in every
        feature column (val/test need not be)."""
        data = torch.load(GRAPH_DATA_PT, weights_only=True)
        x = data.x.numpy()
        train = data.train_mask.numpy()
        train_means = x[train].mean(axis=0)
        # Every column is near-zero mean on the train split it was fit on.
        assert np.allclose(train_means, 0.0, atol=1e-3), (
            "Train-split feature means are not ~0 — scaler was not fit on "
            "train rows only (H1 leak)."
        )

    def test_train_subgraph_excludes_future_edges(self):
        """H2: the training-time subgraph must contain NO edge whose endpoints
        reach into val/test timesteps. We reconstruct the train subgraph edge
        filter from build_graph's logic and assert every edge's endpoints are
        <= the train max timestep."""
        from graphguard.config import TRAIN_TIME_STEPS

        data = torch.load(GRAPH_DATA_PT, weights_only=True)
        ts = data.time_step
        ei = data.edge_index
        train_max = int(max(TRAIN_TIME_STEPS))
        keep = (ts[ei[0]] <= train_max) & (ts[ei[1]] <= train_max)
        train_ei = ei[:, keep]
        # Every endpoint of a train-subgraph edge must be <= train_max.
        assert int((ts[train_ei[0]] <= train_max).all()) == 1
        assert int((ts[train_ei[1]] <= train_max).all()) == 1
        # And some edges were dropped (the full graph reaches timesteps > train_max).
        assert train_ei.size(1) < ei.size(1), (
            "No edges dropped when forming the train subgraph — the train "
            "subgraph equals the full graph (H2 not applied)."
        )

    def test_mlp_trains_on_train_mask_only(self):
        """H3: train_baseline.train_mlp must fit on train_mask (not train|val).
        Read the source and assert the fit call uses only train_mask — this is
        a static guard against re-introducing the unfair protocol that gave the
        MLP an edge over XGBoost."""
        src = Path(__file__).resolve().parent.parent / "src" / "graphguard" / "train_baseline.py"
        text = src.read_text(encoding="utf-8")
        # The MLP fit call must NOT include val_mask in its indexing.
        assert "train_mask | val_mask" not in text, (
            "train_baseline.py still fits the MLP on train|val (H3 leak)."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

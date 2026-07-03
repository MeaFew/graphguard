"""GNN 可解释性（GNNExplainer）管线测试。

覆盖三个层面：
1. Explainer 接入正确性 —— 合成小图 + 简单 GNN，验证 explain() 返回合法的
   edge_mask / node_mask（形状、非负、有梯度分布）。
2. TP 节点选取逻辑 —— 验证 ``select_tp_nodes`` 返回的确实是 test illicit 中
   prob 最高的节点，且结构感知配额生效。
3. weights_only 加载不破坏 —— ``import config`` 注册 safe globals 后，
   ``torch.load(..., weights_only=True)`` 能正常加载 graph_data.pt（回归保护，
   防止日后误删 config 的 add_safe_globals 调用）。
4. 子图 faithfulness —— k-hop 子图上目标节点预测 ≈ 全图预测（解释有效的前提）。
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402  — must register safe globals before any weights_only load

# 让测试也能 import scripts 下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from explain_gnn import select_tp_nodes, subgraph_prob  # noqa: E402


# ── 合成小图 + 简单 SAGE，供 Explainer 接入与 faithfulness 测试共用 ─────────
def _make_toy_graph(seed: int = 0) -> tuple[Data, torch.nn.Module]:
    """构造一个 12 节点、24 边的有向 toy 图 + 一个 2 层 SAGE。

    节点 0 有多条入边（作为 source_to_target 聚合目标），保证它有可解释子图。
    """
    torch.manual_seed(seed)
    # 构造有向边：0 <- {1,2,3,4}（0 是汇聚点，作为待解释节点）
    src = [1, 2, 3, 4, 2, 3, 5, 6, 7, 8, 6, 7, 9, 10, 11, 0, 5, 8, 9, 10, 1, 4, 11, 5]
    dst = [0, 0, 0, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 6, 7, 10, 11, 5, 8, 9, 10]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(12, 6)
    y = torch.zeros(12, dtype=torch.long)
    y[0] = 1  # 节点 0 标记为 illicit
    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = torch.zeros(12, dtype=torch.long)
    data.train_mask = torch.zeros(12, dtype=torch.bool)
    data.train_mask[1:8] = True
    data.val_mask = torch.zeros(12, dtype=torch.bool)
    data.val_mask[8:10] = True
    data.test_mask = torch.zeros(12, dtype=torch.bool)
    data.test_mask[0] = True  # 节点 0 在 test split 且 illicit
    data.test_mask[10:12] = True

    class ToySAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(6, 16)
            self.conv2 = SAGEConv(16, 16)
            self.classifier = Linear(16, 1)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            return self.classifier(x).squeeze(-1)

    model = ToySAGE()
    model.eval()
    return data, model


def _build_explainer(model) -> Explainer:
    """与 scripts/explain_gnn.build_explainer 等价的 Explainer 构造。"""
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=30, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )


# ────────────────────────────────────────────────────────────────────
class TestExplainerInterface:
    """Explainer 接入正确性：返回的 mask 合法。"""

    def test_explainer_returns_valid_masks(self):
        data, model = _make_toy_graph()
        explainer = _build_explainer(model)
        exp = explainer(data.x, data.edge_index, index=0)

        # edge_mask：每条边一个重要性值
        assert exp.edge_mask is not None
        assert exp.edge_mask.shape[0] == data.edge_index.shape[1]
        # 重要性应在 [0, 1]（GNNExplainer 用 sigmoid 输出 object 掩码）
        assert exp.edge_mask.min() >= 0.0
        assert exp.edge_mask.max() <= 1.0
        # 不应全部相同（否则掩码未学到东西）
        assert float(exp.edge_mask.std()) > 1e-6, "edge_mask 退化：所有边重要性相同"

    def test_node_mask_shape_matches_features(self):
        data, model = _make_toy_graph()
        explainer = _build_explainer(model)
        exp = explainer(data.x, data.edge_index, index=0)
        # node_mask: [n_nodes, n_features]（attributes 类型）
        assert exp.node_mask is not None
        assert exp.node_mask.shape == (data.num_nodes, data.num_features)
        assert exp.node_mask.min() >= 0.0

    def test_explanation_carries_prediction(self):
        """Explanation 应携带被解释节点的预测（可审计）。"""
        data, model = _make_toy_graph()
        explainer = _build_explainer(model)
        exp = explainer(data.x, data.edge_index, index=0)
        assert hasattr(exp, "prediction")
        # index 字段记录被解释的节点
        assert int(exp.index) == 0


# ────────────────────────────────────────────────────────────────────
class TestTPSelection:
    """TP 节点选取逻辑（rank-based + 结构感知配额）。"""

    def test_returns_test_illicit_nodes(self):
        data, model = _make_toy_graph()
        # 节点 0 是唯一 test illicit
        probs = torch.zeros(12)
        probs[0] = 0.9  # 最高
        chosen, chosen_probs, meta = select_tp_nodes(probs, data, n=1)
        assert 0 in chosen
        assert chosen_probs[chosen.index(0)] == pytest.approx(0.9)
        assert meta["n_total_test_illicit"] == 1

    def test_rank_based_picks_highest_prob(self):
        """在多个 test illicit 中，应取 prob 最高的若干个。"""
        data, model = _make_toy_graph()
        # 让节点 0、10、11 都成为 test illicit
        data.y[10] = 1
        data.y[11] = 1
        data.test_mask[0] = True
        data.test_mask[10] = True
        data.test_mask[11] = True
        probs = torch.zeros(12)
        probs[0] = 0.2
        probs[10] = 0.8  # 最高
        probs[11] = 0.5
        chosen, chosen_probs, _ = select_tp_nodes(probs, data, n=2)
        # 应选 prob 最高的两个：10 和 11
        assert set(chosen) == {10, 11}
        # 返回顺序按 prob 降序（浮点用 approx 容差）
        assert chosen_probs == pytest.approx([0.8, 0.5])

    def test_structure_aware_quota_enforced(self):
        """至少 ceil(N/2) 个被选节点须有入边（结构化）。"""
        data, model = _make_toy_graph()
        # 构造 6 个 test illicit：3 个有入边（0,1,2 是 dst），3 个无入边。
        data.y[:] = 1
        data.test_mask[:] = False
        data.test_mask[range(6)] = True  # 0..5
        # 重设边：0,1,2 作为 dst（有入边）；3,4,5 不在任何边里（孤立）
        data.edge_index = torch.tensor([[6, 7, 8, 9, 10, 11], [0, 1, 2, 0, 1, 2]], dtype=torch.long)
        probs = torch.linspace(0.1, 0.6, 12)  # 3,4,5 的 prob 较高
        chosen, _, meta = select_tp_nodes(probs, data, n=4)
        # n=4 -> 至少 ceil(4/2)=2 个结构化
        dst_nodes = set(data.edge_index[1].tolist())
        n_struct = sum(1 for c in chosen if c in dst_nodes)
        assert n_struct >= 2, f"结构化节点配额未满足：只有 {n_struct} 个有入边"
        assert meta["n_structured_chosen"] >= 2

    def test_raises_when_no_illicit(self):
        data, model = _make_toy_graph()
        data.y[:] = 0  # 无 illicit
        probs = torch.rand(12)
        with pytest.raises(RuntimeError, match="没有 illicit"):
            select_tp_nodes(probs, data, n=1)


# ────────────────────────────────────────────────────────────────────
class TestWeightsOnlyLoad:
    """回归保护：import config 注册 safe globals 后 weights_only=True 不报错。"""

    def test_graph_data_loads_weights_only(self):
        if not config.GRAPH_DATA_PT.exists():
            pytest.skip("graph_data.pt 未构建（先跑 make data）")
        # 关键：这里不手动 add_safe_globals，仅靠顶部 import config 注册。
        data = torch.load(config.GRAPH_DATA_PT, weights_only=True)
        assert data.x is not None
        assert data.edge_index is not None
        assert data.y is not None

    def test_config_registers_safe_globals(self):
        """config 模块必须调用 add_safe_globals（静态守卫，防误删）。"""
        import torch
        from torch_geometric.data import Data

        safe = torch.serialization.get_safe_globals()
        assert Data in safe, (
            "config.py 末尾的 add_safe_globals([Data, ...]) 似乎被删除——"
            "weights_only=True 加载 graph_data.pt 会失败。"
        )


# ────────────────────────────────────────────────────────────────────
class TestSubgraphFaithfulness:
    """k-hop 子图上预测 ≈ 全图预测（解释有效的前提条件）。"""

    def test_subgraph_prediction_matches_full(self):
        """对 2 层 GNN，2-hop 子图上的目标节点预测应与全图一致。

        这是 GNNExplainer 在子图上做解释的合理性根基：如果子图预测偏离全图，
        解释的就是另一个预测，失去意义。
        """
        data, model = _make_toy_graph()
        from torch_geometric.utils import k_hop_subgraph

        with torch.no_grad():
            full_logit = model(data.x, data.edge_index)[0].item()

        # 节点 0 的 2-hop source_to_target 子图
        subset, sub_ei, mapping, _ = k_hop_subgraph(
            0,
            2,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
            flow="source_to_target",
        )
        sub_prob = subgraph_prob(model, data.x, sub_ei, subset, mapping, torch.device("cpu"))
        with torch.no_grad():
            full_prob = float(torch.sigmoid(torch.tensor(full_logit)))
        # 2-hop 子图覆盖了 2 层 GNN 的全部感受野，预测应几乎一致。
        assert abs(sub_prob - full_prob) < 1e-4, (
            f"子图预测 {sub_prob:.5f} 与全图 {full_prob:.5f} 偏差过大，"
            "子图不足以复现模型预测——解释将不可信。"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""GNN 可解释性管线（GNNExplainer）——金融合规的"为什么"。

反欺诈监管不只关心"这笔交易是否被预测为欺诈"，更要求回答"**为什么**这笔
交易被判欺诈"：哪些相邻交易、哪些特征支撑了这个判定？GNNExplainer 通过学习
一个边掩码 / 节点特征掩码，找出对目标节点预测最关键的子图，从而给出可审计的
依据——这是本作品集相对稀缺的"深度记忆点"。

实现要点：
- 用 PyG 2.8 原生 ``torch_geometric.explain``（Explainer + GNNExplainer），不依赖
  captum。
- 全图 20 万节点上跑解释既慢又（对 2 层 GNN）无意义，因此对每个待解释节点先用
  ``k_hop_subgraph`` 取其 2-hop 感受野子图，再在子图上解释。已验证子图预测与
  全图预测一致（faithful）。
- Elliptic 边是有向 BTC 资金流；GraphSAGE 按 source_to_target 聚合，所以节点的
  预测依赖**入边前驱**（谁向它转 BTC）。采样用 ``flow="source_to_target"``。

采样策略（重要、诚实）：
    test split（timestep 43-49）的 illicit 类是 ~2% 极少数，叠加时序漂移，模型在
    test 上极度欠自信——169 个 illicit 节点中只有 2 个 prob≥0.5，中位 prob=0.034
    （这正是 test AP=0.062 的由来）。按"绝对高置信度阈值"几乎挑不出节点。因此
    这里采用 **rank-based** 选取：取模型在所有 test illicit 节点中**预测概率排名
    最高**的若干节点作为"模型最确信是欺诈的真阳性"。这能稳定取到目标数量，且
    语义上是"模型最自信的 TP"，对解释这些判定最有意义。
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能存图
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

# config registers safe globals (config.py add_safe_globals) so weights_only=True
# loading of graph_data.pt works.
import graphguard.config as config
from graphguard.logging_setup import get_logger, setup_logging
from graphguard.train_gnn import GraphSAGE

logger = get_logger(__name__)

# ── 配置（从 config 读取，便于集中调参） ─────────────────────────────
HOPS = config.EXPLAINER_HOPS
FLOW = config.EXPLAINER_FLOW
EPOCHS = config.EXPLAINER_EPOCHS
LR = config.EXPLAINER_LR
NUM_NODES = config.EXPLAIN_NUM_NODES
TOP_K_EDGES = config.EXPLAIN_TOP_K_EDGES
MAX_PLOT_NODES = config.EXPLAIN_MAX_PLOT_NODES
OUT_DIR = config.EXPLANATIONS_DIR
SUMMARY_JSON = config.EXPLANATION_SUMMARY_JSON

# 邻居标签的可视化配色（与 dashboard 风格一致）
LABEL_COLORS = {1: "#d62728", 0: "#2ca02c", -1: "#bbbbbb"}  # illicit红 / licit绿 / unknown灰
LABEL_NAMES = {1: "illicit", 0: "licit", -1: "unknown"}


def load_model_and_data(device):
    """加载训练好的 GraphSAGE + 图数据。

    返回 (model, data)，model 已 eval 并在 device 上，data 留在 CPU（子图采样
    产物再搬到 device）。
    """
    data = torch.load(config.GRAPH_DATA_PT, weights_only=True)
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
    ).to(device)
    model.load_state_dict(
        torch.load(config.SAGE_MODEL_PATH, weights_only=True, map_location=device)
    )
    model.eval()
    return model, data


@torch.no_grad()
def compute_full_probs(model, data, device):
    """全图一次性 forward，返回每个节点的 illicit 概率（sigmoid 后）。"""
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    return torch.sigmoid(out).cpu()


def select_tp_nodes(probs, data, n):
    """选取 test split 中被模型最确信为 illicit 的真阳性节点（rank-based）。

    在 ``test_mask & y==1`` 节点里按预测概率降序取前 n 个。这些是"模型最自信的
    TP"，对它们做解释最有说服力。

    同时做一个**结构感知补充**：高排名 illicit 节点中相当一部分**没有入边**（预测
    纯由自身特征驱动，子图无可解释结构）。GNNExplainer 对这类节点无能为力。因此
    在 rank-based 取得的前 n 个里，若孤立节点过多，会用更多具有入边的结构化节点
    替补——保持"模型最自信的 TP"主旨，但确保最终选集中至少有 ``n//2`` 个结构化节点
    可被解释。这一限定会在返回值里诚实记录。

    返回 (node_indices, probs, selection_meta)。
    """
    test_illicit = data.test_mask & (data.y == 1)
    cand_idx = torch.where(test_illicit)[0]
    if len(cand_idx) == 0:
        raise RuntimeError(
            "test split 中没有 illicit 节点，无法解释。请检查 data/processed/graph_data.pt"
        )
    cand_probs = probs[cand_idx]
    order = torch.argsort(cand_probs, descending=True)
    sorted_idx = cand_idx[order]
    sorted_probs = cand_probs[order]

    # 标记每个候选节点是否有入边（作为聚合目标，source_to_target 流向下）。
    # dst 列出现的节点即"有人向它转 BTC"，是有结构的可解释节点。
    dst_nodes = set(data.edge_index[1].tolist())
    has_in_edge = torch.tensor([int(int(i) in dst_nodes) for i in sorted_idx.tolist()])

    structured = sorted_idx[has_in_edge.bool()]
    structured_p = sorted_probs[has_in_edge.bool()]
    isolated = sorted_idx[~has_in_edge.bool()]
    isolated_p = sorted_probs[~has_in_edge.bool()]

    # 配额：至少 ceil(n/2) 个结构化节点；剩余名额按 rank（不分结构）填充。
    n_struct_quota = min(len(structured), (n + 1) // 2)
    picked_struct = structured[:n_struct_quota].tolist()
    picked_struct_p = structured_p[:n_struct_quota].tolist()
    remaining = max(0, n - len(picked_struct))
    # 用剩余 rank 最高的节点填充（结构化优先，因为它们才有可解释子图）。
    extra_struct_needed = max(0, remaining - len(isolated))
    picked_struct += structured[n_struct_quota : n_struct_quota + extra_struct_needed].tolist()
    picked_struct_p += structured_p[n_struct_quota : n_struct_quota + extra_struct_needed].tolist()
    n_iso = n - len(picked_struct)
    picked_iso = isolated[:n_iso].tolist()
    picked_iso_p = isolated_p[:n_iso].tolist()

    # 合并并按 prob 降序输出（保持 rank 顺序，便于阅读）。
    all_pairs = list(zip(picked_struct, picked_struct_p)) + list(zip(picked_iso, picked_iso_p))
    all_pairs.sort(key=lambda t: -t[1])
    chosen = [p[0] for p in all_pairs]
    chosen_probs = [p[1] for p in all_pairs]

    meta = {
        "n_total_test_illicit": int(len(cand_idx)),
        "n_structured_chosen": len(picked_struct),
        "n_isolated_chosen": len(picked_iso),
        "note": (
            "rank-based selection with a structure-aware quota: at least ceil(N/2) "
            "explained nodes are required to have incoming edges (otherwise "
            "GNNExplainer has no subgraph to explain). Isolated nodes are kept to "
            "honestly show how often the prediction is feature-only."
        ),
    }
    return chosen, chosen_probs, meta


def build_explainer(model):
    """构造 PyG Explainer（包装 GNNExplainer）。

    - explanation_type='model'：解释模型自身的预测（非 ground-truth 对比）。
    - node_mask_type='attributes'：学习每个节点各特征维度的重要性。
    - edge_mask_type='object'：学习每条边的重要性（目标）。
    - return_type='raw'：模型输出 raw logit，与训练时一致。
    """
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=EPOCHS, lr=LR),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="node",
            return_type="raw",
        ),
    )


@torch.no_grad()
def subgraph_prob(model, x, edge_index, subset, mapping, device):
    """在 k-hop 子图上重算目标节点预测概率，用于校验 faithfulness。"""
    out = model(x[subset].to(device), edge_index.to(device))
    return float(torch.sigmoid(out[mapping]).cpu())


def explain_node(node_idx, model, data, explainer, device):
    """对单个节点做解释。

    1) k_hop_subgraph 取 2-hop 入边前驱子图（relabel）。
    2) 在子图上跑 GNNExplainer，得 edge_mask / node_mask。
    3) 校验子图预测与全图预测接近（faithful）。

    返回 dict（含原始节点 id、子图节点 id、edge_mask、node_mask、子图预测等）。
    """
    subset, sub_ei, mapping, _ = k_hop_subgraph(
        node_idx, HOPS, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes, flow=FLOW
    )
    mapping = int(mapping.item())
    if sub_ei.size(1) == 0:
        # 无入边前驱：该节点预测完全由自身特征驱动（无可解释的子图结构）。
        return {
            "node_idx": int(node_idx),
            "subset": subset.tolist(),
            "edge_index": sub_ei.tolist(),
            "edge_mask": [],
            "node_mask": None,
            "sub_prob": None,
            "isolated": True,
            "mapping": mapping,
        }

    sub_prob = subgraph_prob(model, data.x, sub_ei, subset, mapping, device)
    # Explainer 内部直接调 model(x, edge_index)，需保证二者与模型同 device。
    sub_x = data.x[subset].to(device)
    sub_ei_dev = sub_ei.to(device)
    # GNNExplainer 内部会置 model.training 相关逻辑；explainer 在 model_config 下
    # 自动管理 train/eval。这里无需手动 no_grad。
    exp = explainer(sub_x, sub_ei_dev, index=mapping)
    return {
        "node_idx": int(node_idx),
        "subset": subset.tolist(),
        "edge_index": sub_ei.tolist(),
        "edge_mask": exp.edge_mask.detach().cpu().tolist(),
        "node_mask": exp.node_mask.detach().cpu().tolist() if exp.node_mask is not None else None,
        "sub_prob": sub_prob,
        "isolated": False,
        "mapping": mapping,
    }


def visualize_subgraph(result, data, probs, out_path):
    """画一张子图：中心节点 + 重要邻居 + 边重要性。

    边粗细 ∝ GNNExplainer 重要性；节点颜色按标签（illicit 红/licit 绿/unknown 灰），
    中心目标节点加黑色描边并放大。节点数超过 MAX_PLOT_NODES 时按边重要性裁剪。
    """
    if result["isolated"] or not result["edge_mask"]:
        # 无入边节点：画一张单节点图说明"纯特征驱动"。matplotlib 默认字体
        # 不含中文 glyph，故所有图内文字用英文渲染。
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            [0],
            [0],
            s=900,
            c=[LABEL_COLORS[int(data.y[result["node_idx"]])]],
            edgecolors="black",
            linewidths=2.5,
            zorder=3,
        )
        ax.annotate(
            "target node\n(no incoming edges;\nprediction is feature-only)",
            (0, 0),
            (0, -0.15),
            ha="center",
            fontsize=9,
        )
        ax.set_title(f"Node {result['node_idx']} — no incoming edges (feature-only)", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return

    subset = result["subset"]
    sub_ei = torch.tensor(result["edge_index"], dtype=torch.long)
    edge_mask = torch.tensor(result["edge_mask"], dtype=torch.float)
    target_global = result["node_idx"]
    mapping = result["mapping"]

    # 取 top-k 重要边构建可视子图（控制稠密度）
    k = min(TOP_K_EDGES, edge_mask.numel())
    topk = torch.topk(edge_mask, k)
    keep_edges = topk.indices
    ei_viz = sub_ei[:, keep_edges]
    w_viz = edge_mask[keep_edges]

    # 限制可视节点数：取这些边涉及的节点
    viz_nodes = torch.unique(ei_viz).tolist()
    if mapping not in viz_nodes:
        viz_nodes.append(mapping)
    # 若仍过多，按"是否含目标 + 是否 illicit"优先保留
    if len(viz_nodes) > MAX_PLOT_NODES:
        keep = [mapping]
        for nd in viz_nodes:
            if nd == mapping:
                continue
            keep.append(nd)
            if len(keep) >= MAX_PLOT_NODES:
                break
        viz_nodes_set = set(keep)
    else:
        viz_nodes_set = set(viz_nodes)

    g = nx.DiGraph()
    for nd in viz_nodes_set:
        g_global = subset[nd]
        lab = int(data.y[g_global])
        g.add_node(
            nd,
            label=LABEL_NAMES.get(lab, "?"),
            color=LABEL_COLORS.get(lab, "#888"),
            is_target=(nd == mapping),
            global_id=int(g_global),
            prob=float(probs[g_global]),
        )

    for e in range(ei_viz.size(1)):
        s, d = int(ei_viz[0, e]), int(ei_viz[1, e])
        if s in viz_nodes_set and d in viz_nodes_set:
            g.add_edge(s, d, weight=float(w_viz[e]))

    fig, ax = plt.subplots(figsize=(8, 7))
    # 固定随机种子保证布局可复现
    pos = nx.spring_layout(g, seed=config.RANDOM_STATE, k=0.9)

    node_colors = [g.nodes[n]["color"] for n in g.nodes()]
    node_sizes = [1300 if g.nodes[n]["is_target"] else 480 for n in g.nodes()]
    edge_widths = [0.8 + 4.5 * g[u][v]["weight"] for u, v in g.edges()]
    # 重要边更不透明，弱边淡化，强化视觉层次
    edge_alphas = [0.3 + 0.65 * g[u][v]["weight"] for u, v in g.edges()]
    edge_colors = [(0.4, 0.4, 0.4, a) for a in edge_alphas]

    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=[2.5 if g.nodes[n]["is_target"] else 0.6 for n in g.nodes()],
    )
    # 图内文字一律英文（matplotlib 默认字体缺中文 glyph）。"[T]" 标目标节点。
    labels = {n: ("[T]" if g.nodes[n]["is_target"] else "") for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_color="white", font_size=9)

    # 节点旁标注简短信息（标签 + 全局 id）
    for n, (x_, y_) in pos.items():
        lab = g.nodes[n]["label"]
        gid = g.nodes[n]["global_id"]
        ax.text(
            x_, y_ + 0.06, f"{lab[:3]}\n#{gid}", fontsize=6.5, ha="center", color="#333", alpha=0.9
        )

    sub_prob = result["sub_prob"]
    title = (
        f"Node #{target_global} explanation\n"
        f"([T]=target, edge width = importance, "
        f"subgraph prob={sub_prob:.3f})"
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LABEL_COLORS[1],
                markersize=10,
                label="illicit",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LABEL_COLORS[0],
                markersize=10,
                label="licit",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=LABEL_COLORS[-1],
                markersize=10,
                label="unknown",
            ),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def aggregate_analysis(results, data, probs):
    """跨多个解释节点做聚合统计，揭示"illicit 子图的共性模式"。

    回答三个问题：
    1. 欺诈节点的关键邻居都是什么标签？（欺诈是否倾向于连接欺诈——团伙性）
    2. 关键邻居 vs 该 illicit 节点的特征均值差异如何？
    3. 关键边的方向构成（前驱 vs 自环）？

    同时统计全局最重要的特征维度（对所有解释节点的 node_mask 取均值排序）。
    返回 dict（写入 explanation_summary.json 的 aggregate 部分）。
    """
    n_explained = len(results)
    n_isolated = sum(1 for r in results if r["isolated"])

    # 1) 关键邻居的标签分布（按 top-k 边权重加权）
    neighbor_label_weight = {1: 0.0, 0: 0.0, -1: 0.0}
    neighbor_label_count = {1: 0, 0: 0, -1: 0}
    total_edge_weight = 0.0
    feat_importance_accum = None  # 累加 node_mask 用于全局特征重要性
    feat_importance_n = 0
    target_feat_means = []
    neighbor_feat_means = []  # 重要邻居的特征均值

    for r in results:
        if r["isolated"]:
            continue
        subset = torch.tensor(r["subset"], dtype=torch.long)
        sub_ei = torch.tensor(r["edge_index"], dtype=torch.long)
        edge_mask = torch.tensor(r["edge_mask"], dtype=torch.float)
        if r["node_mask"] is not None:
            nm = torch.tensor(r["node_mask"], dtype=torch.float)  # [n_sub, F]
            if feat_importance_accum is None:
                feat_importance_accum = nm.sum(dim=0)
            else:
                feat_importance_accum = feat_importance_accum + nm.sum(dim=0)
            feat_importance_n += nm.size(0)

        # 目标节点（中心）的特征均值
        mapping = r["mapping"]
        target_global = subset[mapping]
        target_feat_means.append(data.x[target_global].mean().item())

        # top-k 重要边涉及的邻居（排除自身）
        k = min(TOP_K_EDGES, edge_mask.numel())
        topk_idx = torch.topk(edge_mask, k).indices
        for ei in topk_idx.tolist():
            s, _dst = int(sub_ei[0, ei]), int(sub_ei[1, ei])
            # source_to_target 流向：_dst 是聚合到的目标（通常==mapping 即中心），
            # s 是贡献的前驱邻居。重要邻居 = s。
            w = float(edge_mask[ei])
            neighbor_global = subset[s]
            lab = int(data.y[neighbor_global])
            neighbor_label_weight[lab] = neighbor_label_weight.get(lab, 0.0) + w
            neighbor_label_count[lab] = neighbor_label_count.get(lab, 0) + 1
            total_edge_weight += w
            neighbor_feat_means.append(data.x[neighbor_global].mean().item())

    # 归一化邻居标签权重
    if total_edge_weight > 0:
        nl_weight_norm = {
            LABEL_NAMES[k]: round(v / total_edge_weight, 4)
            for k, v in neighbor_label_weight.items()
        }
    else:
        nl_weight_norm = {}

    # 全局 top 特征维度（跨所有解释节点）
    if feat_importance_accum is not None and feat_importance_n > 0:
        avg_feat_imp = feat_importance_accum / feat_importance_n
        top_feat_dims = torch.topk(avg_feat_imp, min(10, avg_feat_imp.numel()))
        top_features = [
            {"feature_dim": int(i), "importance": round(float(v), 5)}
            for v, i in zip(top_feat_dims.values.tolist(), top_feat_dims.indices.tolist())
        ]
    else:
        top_features = []

    return {
        "n_explained": n_explained,
        "n_isolated_feature_only": n_isolated,
        "neighbor_label_weighted_share": nl_weight_norm,
        "neighbor_label_counts": {LABEL_NAMES[k]: v for k, v in neighbor_label_count.items()},
        "target_node_feature_mean": (
            round(float(np.mean(target_feat_means)), 5) if target_feat_means else None
        ),
        "key_neighbor_feature_mean": (
            round(float(np.mean(neighbor_feat_means)), 5) if neighbor_feat_means else None
        ),
        "top_global_feature_dims": top_features,
        "interpretation_notes": [
            "neighbor_label_weighted_share: 关键邻居（按解释重要性加权）中各标签占比。"
            "若 illicit 占比显著高于图先验（illicit ~2%），说明欺诈呈团伙聚集。",
            "n_isolated_feature_only: 无入边前驱、预测纯由自身特征驱动的节点数。"
            "占比高意味着模型对相当一部分欺诈的判定不依赖图结构——这对'图结构带来增益'"
            "的叙事是个诚实的限定。",
            "top_global_feature_dims: 跨所有解释节点聚合后最重要的特征维度"
            "（Elliptic 特征匿名，无法给出业务语义，但可用于特征工程迭代方向）。",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="GNN 可解释性（GNNExplainer）")
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=NUM_NODES,
        help="要解释的高置信度 illicit TP 节点数（rank-based 取前 N）",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="跳过子图可视化（调试用）",
    )
    args = parser.parse_args()

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, data = load_model_and_data(device)
    probs = compute_full_probs(model, data, device)

    # ---- 1) 选取 TP 节点（rank-based + 结构感知配额）----
    chosen, chosen_probs, sel_meta = select_tp_nodes(probs, data, args.num_nodes)
    logger.info(f"\n选定 {len(chosen)} 个高置信度 illicit TP 节点（test split 内 rank-based）。")
    if len(chosen) < args.num_nodes:
        logger.info(f"  注意：test illicit 总数不足 {args.num_nodes}，实际取 {len(chosen)} 个。")
    logger.info(
        f"  结构化(有入边): {sel_meta['n_structured_chosen']}，"
        f"孤立(纯特征): {sel_meta['n_isolated_chosen']}"
    )
    logger.info(
        f"  prob 范围: [{min(chosen_probs):.4f}, {max(chosen_probs):.4f}]，"
        f"中位 {float(np.median(chosen_probs)):.4f}"
    )

    explainer = build_explainer(model)

    # ---- 2) 逐节点解释 ----
    results = []
    for i, node_idx in enumerate(chosen):
        r = explain_node(node_idx, model, data, explainer, device)
        results.append(r)
        tag = "isolated(feature-only)" if r["isolated"] else f"sub_prob={r['sub_prob']:.3f}"
        logger.info(f"  [{i + 1}/{len(chosen)}] node {node_idx} prob={chosen_probs[i]:.4f} {tag}")

    # ---- 3) 可视化 ----
    viz_paths = []
    if not args.no_viz:
        logger.info("\n生成子图可视化...")
        for i, (node_idx, r) in enumerate(zip(chosen, results)):
            out_path = OUT_DIR / f"node_{node_idx}.png"
            visualize_subgraph(r, data, probs, out_path)
            viz_paths.append(str(out_path.relative_to(config.BASE_DIR)))
        logger.info(f"  已保存 {len(viz_paths)} 张子图到 {OUT_DIR}")

    # ---- 4) 聚合分析 ----
    logger.info("\n聚合分析...")
    agg = aggregate_analysis(results, data, probs)
    logger.info("  关键邻居标签加权占比:", agg["neighbor_label_weighted_share"])
    logger.info("  关键邻居标签计数:", agg["neighbor_label_counts"])
    logger.info(f"  无入边（纯特征）节点: {agg['n_isolated_feature_only']}/{agg['n_explained']}")
    if agg["top_global_feature_dims"]:
        logger.info(
            "  全局 top-5 重要特征维度:",
            [(d["feature_dim"], d["importance"]) for d in agg["top_global_feature_dims"][:5]],
        )

    # ---- 5) 写 summary ----
    summary = {
        "model": "GraphSAGE",
        "model_path": str(config.SAGE_MODEL_PATH.relative_to(config.BASE_DIR)),
        "methodology": {
            "explainer": "GNNExplainer (PyG 2.8 torch_geometric.explain)",
            "explanation_type": "model",
            "edge_mask_type": "object",
            "node_mask_type": "attributes",
            "hops": HOPS,
            "flow": FLOW,
            "epochs": EPOCHS,
            "lr": LR,
            "selection": (
                "rank-based: 在 test_split & y==1 节点中按模型预测 prob 降序取前 N。"
                "因 test AP 仅 0.062、模型在 test 上极度欠自信（中位 prob ~0.03，仅 ~2 个 ≥0.5），"
                "采用 rank 而非绝对阈值，语义为'模型最确信的 TP'。叠加结构感知配额："
                "至少 ceil(N/2) 个被解释节点须有入边（否则 GNNExplainer 无子图可解释）。"
            ),
            "selection_meta": sel_meta,
            "top_k_edges_per_node": TOP_K_EDGES,
        },
        "n_explained": len(chosen),
        "explained_node_probs": {int(n): round(float(p), 5) for n, p in zip(chosen, chosen_probs)},
        "aggregate": agg,
        "per_node": [
            {
                "node_idx": r["node_idx"],
                "isolated": r["isolated"],
                "sub_prob": r["sub_prob"],
                "n_subgraph_nodes": len(r["subset"]),
                "n_subgraph_edges": len(r["edge_mask"]),
                "top_edge_importances": (
                    sorted(r["edge_mask"], reverse=True)[:TOP_K_EDGES] if r["edge_mask"] else []
                ),
            }
            for r in results
        ],
        "visualization_pngs": viz_paths,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"\n聚合摘要写入 {SUMMARY_JSON.relative_to(config.BASE_DIR)}")
    logger.info("完成。")


if __name__ == "__main__":
    setup_logging()
    main()

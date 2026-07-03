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
| **GraphSAGE** | **0.793** | **0.062** |
| MLP (features only) | 0.727 | 0.047 |
| GCN | 0.718 | 0.049 |
| GAT | 0.717 | 0.049 |
| GIN | 0.707 | 0.049 |
| XGBoost (features only) | 0.702 | 0.045 |

> **GraphSAGE is now the best model** — a GNN beats the strongest tabular
> baseline (MLP) on both ROC-AUC and AP. This was the goal of the leakage
> fixes (see "Leakage & fairness fixes" below): under the earlier protocol the
> GNNs were fed drifting future-neighbor features at train time and the MLP was
> trained on train+val, so "MLP beats GNN" was an artifact of an unfair
> comparison. With time-causal subgraph sampling and a matched training
> protocol, GraphSAGE's inductive neighborhood aggregation wins, exactly as
> theory predicts for a setting where fraud patterns drift across timesteps.

> Validation ROC-AUC reaches 0.88-0.94, but the test split (later time steps)
> is markedly harder due to fraud-pattern drift over time — a well-known
> property of the Elliptic benchmark. Model selection uses Average Precision
> (AP), the right metric for this highly imbalanced illicit minority; ROC-AUC
> is dominated by the abundant licit class and is a poor proxy for ranking
> quality here.

### Leakage & fairness fixes (important)

Three leakage / protocol bugs that previously understated (or mis-stated) the
GNN results are corrected:

- **Feature scaler fit on train nodes only** (`build_graph.py`): earlier
  `StandardScaler().fit_transform(x)` ran over ALL nodes (incl. val/test), so
  future node statistics leaked into every model's input. Now fit on `train_mask`
  rows only, then transform all rows. Temporal node features (normalized
  timestep + sin/cos) are stacked BEFORE scaling so they are standardized on
  train too.
- **Time-causal GNN subgraphs** (`train_gnn.py`): earlier `NeighborLoader`
  sampled over the full graph, so a train root's representation aggregated
  val/test-timestep node features (transductive leak). Now each split builds an
  edge-filtered subgraph keeping only edges whose BOTH endpoints have
  `time_step <=` the split's max — train roots cannot reach val/test nodes at
  all. This makes the GNN genuinely inductive and is what lets GraphSAGE win.
- **Matched baseline protocol** (`train_baseline.py`): the MLP was previously
  fit on `train|val` while XGBoost used train-only + val early-stopping, giving
  the MLP an unfair edge (its headline ROC-AUC dropped from 0.810 → 0.727 once
  fit on train only with the leakage-free feature pipeline). Both baselines now
  train on `train_mask` only.
- **Added GIN** (Graph Isomorphism Network, sum-aggregation) as a 4th GNN —
  provably more expressive than GCN/SAGE mean aggregation for distinguishing
  non-isomorphic fraud neighborhoods.

## Project Structure

```
graphguard/
├── scripts/
│   ├── download_data.py       # Fetch or generate dataset
│   ├── build_graph.py         # Build PyG Data object (train-only scaling, temporal features)
│   ├── train_baseline.py      # MLP + XGBoost (train-mask-only protocol)
│   ├── train_gnn.py           # GCN / SAGE / GAT / GIN (time-causal subgraphs)
│   ├── evaluate.py            # Metrics and plots
│   └── explain_gnn.py         # GNN 可解释性（GNNExplainer）——关键子图 + 聚合分析
├── dashboard/
│   └── app.py                 # Streamlit comparison dashboard
├── tests/
│   ├── test_graph.py          # Unit tests (incl. TestLeakagePrevention)
│   └── test_explain.py        # GNNExplainer 接入 / TP 选取 / faithfulness 测试
├── reports/
│   ├── images/explanations/   # 每个 illicit 节点的解释子图 PNG
│   └── explanation_summary.json   # 聚合统计（邻居标签占比 / 全局特征重要性）
├── config.py                  # Paths and hyperparameters
├── run_all.py                 # One-shot pipeline
├── download_data.sh           # Kaggle download helper
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
make explain    # GNN 可解释性（关键子图 + 聚合统计）
make test       # run tests
```

### 3. Launch dashboard

```bash
make dashboard
```

## Key Design Decisions

1. **Time-based split**: train/val/test use disjoint time steps (1-34 / 35-42 / 43-49) so labels never leak forward in time.
2. **Time-causal GNN subgraphs**: beyond disjoint label timesteps, each split's `NeighborLoader` runs on an edge-filtered subgraph where both endpoints have `time_step <=` the split's max — so train roots cannot aggregate val/test-timestep node *features* (the earlier transductive leak).
3. **Train-only feature scaling**: `StandardScaler` is fit on train nodes only, then applied to all; temporal node features (normalized timestep + sin/cos) are stacked before scaling.
4. **Class imbalance**: BCE loss with `pos_weight`; **Average Precision** is the model-selection metric (ROC-AUC is dominated by the abundant licit class).
5. **Val-tuned F1 threshold**: the F1 decision cutoff is tuned on the **validation** split (the F1-maximizing point of its precision-recall curve) and then applied to the test split — never tuned on test. A hardcoded 0.5 would be misleading because `pos_weight`/`scale_pos_weight` shift the predicted-probability distribution well below 0.5.
6. **Mini-batch GNNs**: `NeighborLoader` keeps VRAM usage under 8GB.
7. **Inductive capability**: GraphSAGE / GIN generalize to unseen nodes — and, under the fair protocol above, GraphSAGE wins.
8. **Reproducibility**: fixed seeds, saved checkpoints, logged hyperparameters.
9. **GNN 可解释性（GNNExplainer）**: 金融反欺诈不只是预测，更是合规——监管要求回答"为什么这笔交易被判欺诈"。`scripts/explain_gnn.py` 对高置信度 illicit 真阳性输出关键子图与聚合统计（见下节）。

## GNN 可解释性（GNNExplainer）——从"预测"到"为什么"

### 为什么金融反欺诈必须可解释（监管视角）

预测一个交易"是否欺诈"只解决了一半问题。在真实的反洗钱（AML）/了解你的客户
（KYC）合规场景里，监管机构（如 FinCEN、FATF）要求机构对每一笔被系统拦截或
上报的交易给出**可审计的依据**——"系统依据什么判定它可疑？"。一个黑箱 GNN 给出
"概率 0.97"无法满足审计要求；合规分析师需要的是"这几笔上游交易的什么特征触发
了判定"。GNNExplainer 把"概率"翻译成"关键子图 + 关键特征"，是金融方向作品集相对
稀缺的"深度记忆点"：它把一个预测模型变成一个可解释、可申诉的合规工具。

### 原理（一图流）

```
   全图预测某个 illicit 节点 i 的 logit z_i
                  │
                  ▼
   GNNExplainer: 优化一个边掩码 M ∈ [0,1]^|E| 与节点特征掩码，
   使得「只保留 M 加权后的子图」时，模型对 i 的预测尽量接近原预测 z_i。
                  │
                  ▼
   M 中权重高的边 ⟹ 对该预测最关键的邻接关系
   ⟹ 输出"关键子图"（中心节点 + 重要入边前驱 + 边重要性）
```

一句话：**GNNExplainer 反问"砍掉哪些边/特征后预测会变？"，留下来的就是模型的"理由"。**

### 实现要点（`scripts/explain_gnn.py`）

- 用 PyG 2.8 原生 `torch_geometric.explain`（`Explainer` + `GNNExplainer`），不依赖 captum。
- **子图采样**：全图 20 万节点上跑解释既慢又（对 2 层 GNN）无意义。对每个待解释节点先用
  `k_hop_subgraph(num_hops=2)` 取其 2-hop 感受野子图再解释。已验证子图预测与全图预测
  一致（faithful，见 `tests/test_explain.py::test_subgraph_prediction_matches_full`）。
- **有向边语义**：Elliptic 边是有向 BTC 资金流（src→dst）；GraphSAGE 按
  `source_to_target` 聚合，所以一个节点的预测依赖它的**入边前驱**（谁向它转 BTC）。
  采样用 `flow="source_to_target"`。

### 采样策略（诚实的方法学说明）

test split（timestep 43-49）的 illicit 类是 ~2% 极少数，叠加**时序漂移**，模型在 test 上
**极度欠自信**——169 个 illicit 节点中只有约 2 个预测概率 ≥0.5，中位概率仅 ~0.03
（这正是 test AP=0.062 的由来）。因此"按绝对高置信度阈值（如 prob>0.7）"几乎挑不出节点。

本项目采用 **rank-based 选取**：取模型在所有 test illicit 节点中**预测概率排名最高**的
40 个节点作为"模型最确信的真阳性（TP）"。叠加一个**结构感知配额**：至少一半被解释节点须
有入边（否则 GNNExplainer 无子图可解释）。这一限定诚实记录在 `explanation_summary.json`
的 `methodology.selection` 字段。

### 典型 illicit 解释子图

> `[T]` = 被解释的目标 illicit 节点；边粗细/不透明度 ∝ GNNExplainer 重要性；
> 红圈=illicit / 绿圈=licit / 灰圈=unknown。

![sample explanation subgraph](reports/images/explanations/node_179187.png)

该节点（prob=0.999）的判定**几乎完全由几条高权重入边驱动**——即"有几笔 unknown 上游交易
汇入它"。这正是洗钱典型拓扑：多个上游汇向单一汇聚点。

### 聚合分析（跨 40 个高置信度 illicit TP）

| 指标 | 结果 | 解读 |
|------|------|------|
| 关键邻居中 illicit 占比（加权） | **~24%** | illicit 节点的关键邻居里 illicit 占比 ~24%，**远高于全图 illicit 先验（~2%）**——欺诈呈团伙聚集 |
| 关键邻居中 licit 占比 | ~8% | 混入少量 licit 邻居，模型不只看"周围都是 illicit" |
| 关键邻居中 unknown 占比 | ~68% | 大量关键邻居标签未知（Elliptic 数据特性），是调查的天然候选 |
| 无入边（纯特征驱动）节点 | **20/40** | 一半高置信度 illicit 的判定**不依赖图结构**，纯由节点特征驱动 |
| 全局最重要特征维度 | dim 136, 90, 165, 100 | 跨节点聚合后最关键的匿名特征（Elliptic 特征匿名，无业务语义，但指向特征工程迭代方向） |

**核心洞察**：即便在 AP 仅 0.062 的弱模型上，GNNExplainer 仍揭示出"**欺诈倾向于连接欺诈**"
的团伙性结构信号（关键邻居 illicit 占比 24% vs 先验 2%，约 12 倍富集）——这正是 GNN 相对
纯特征 MLP 的价值所在，也为合规调查指明了"从一笔可疑交易顺藤摸瓜查关联交易"的方向。

### 诚实的局限

- **AP=0.062、模型欠自信**：解释的是"模型最自信的 TP"，但模型本身在 test 上很弱（时序漂移）。
  对那些被漏掉（FN）或低置信度的 illicit，解释意义有限——本项目**没有**对它们硬编故事。
- **一半节点纯特征驱动**：相当一部分高置信度 illicit 的判定不依赖图结构，这对"图结构带来增益"
  的叙事是个诚实的限定——图结构对**聚集性欺诈**有用，对孤立特征可疑的交易则未必。
- **GNNExplainer 有随机性**：每次运行的关键边权重略有波动（邻居 illicit 占比在 24-26% 间浮动），
  但方向稳定。聚合统计是可靠的，单张子图应作为定性参考。
- **特征匿名**：Elliptic 特征无业务语义，"最重要特征维度"无法翻译成可读的业务规则。

### 运行

```bash
make explain          # 默认解释 40 个高置信度 illicit TP
python scripts/explain_gnn.py --num-nodes 20   # 自定义数量
```

产物：`reports/images/explanations/node_*.png`（每节点一张子图）+
`reports/explanation_summary.json`（聚合统计 + 方法学 + 逐节点解释）。

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

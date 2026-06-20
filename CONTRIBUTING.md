# Contributing Guide

感谢你对本项目的兴趣。本指南面向希望本地运行、调试或扩展该图神经网络反欺诈项目的开发者。

## 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/MeaFew/graphguard.git
cd graphguard

# 2. 创建虚拟环境 (Python 3.11+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装 PyTorch (按你的 CUDA 版本选择，下例为 CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装其余依赖
pip install -r requirements.txt
```

> 本项目在 RTX 4060 Laptop (8GB VRAM) 上验证。CPU 亦可运行，但 GNN 训练会显著变慢。

## 数据准备

数据通过 `scripts/download_data.py` 获取 [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)（Kaggle）。

```bash
# 方式 A：配置 Kaggle 凭证后自动下载
#   设置 ~/.kaggle/kaggle.json（或环境变量 KAGGLE_USERNAME / KAGGLE_KEY）
python scripts/download_data.py

# 方式 B：手动下载三个 CSV 放到 data/raw/
#   elliptic_txs_features.csv / elliptic_txs_edgelist.csv / elliptic_txs_classes.csv
```

若未配置 Kaggle 凭证，`build_graph.py` 会自动回退到 `scripts/generate_synthetic_graph.py`
生成的统计性质相似的合成交易图，便于本地测试与 CI。

## 开发工作流

```bash
make verify   # lint (ruff) + test (pytest)  — 提交前确保全绿
```

- **代码风格**：ruff（lint + format）。pre-commit hook 已配置（`.pre-commit-config.yaml`），`pre-commit install` 后每次 commit 自动 lint。
- **测试**：`pytest tests/`。新增算法/修复需配套回归测试（见 `tests/test_graph.py::TestLeakagePrevention`）。
- **Commit 规范**：建议 Conventional Commits（`feat:` / `fix:` / `docs:` / `test:`）。

## 防泄漏约定（重要）

图反欺诈极易引入传导式（transductive）泄漏。本项目的约定：

1. **特征标准化只在训练节点上 fit**（`build_graph.py`：`StandardScaler().fit(x[train_mask])`），不得用全图统计。
2. **GNN 采样使用时间因果子图**（`train_gnn.py`：每个 split 的子图只保留两端 timestep ≤ 该 split 上限的边），训练根节点不得聚合 val/test timestep 的节点特征。
3. **baseline 与 GNN 训练协议一致**（MLP/XGBoost 都只在 `train_mask` 上训练）。

任何改动若涉及上述任一点，必须在 `tests/test_graph.py::TestLeakagePrevention` 中保留或增强对应的回归测试。

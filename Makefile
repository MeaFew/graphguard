.PHONY: all setup data baselines gnn evaluate explain test lint format format-check verify dashboard clean

PYTHON := python

# ── One-shot pipeline ─────────────────────────────────────────────
all: data baselines gnn evaluate explain test

# ── Environment ───────────────────────────────────────────────────
setup:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip install -r requirements.txt

# ── Data pipeline ─────────────────────────────────────────────────
data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/build_graph.py

# ── Modeling ──────────────────────────────────────────────────────
baselines:
	$(PYTHON) scripts/train_baseline.py

gnn:
	$(PYTHON) scripts/train_gnn.py --model all

# ── Evaluation ────────────────────────────────────────────────────
evaluate:
	$(PYTHON) scripts/evaluate.py

# ── GNN 可解释性（GNNExplainer） ───────────────────────────────────
# 对高置信度 illicit 真阳性做解释，输出关键子图 PNG + 聚合统计 JSON。
# 这是金融合规视角的"深度记忆点"：回答"为什么这笔交易被判欺诈"。
explain:
	$(PYTHON) scripts/explain_gnn.py

# ── Quality gates ─────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -q

lint:
	ruff check scripts/ tests/ dashboard/

format:
	ruff format scripts/ tests/ dashboard/

format-check:
	ruff format --check scripts/ tests/ dashboard/

# Full local quality gate (lint + format-check + test). Mirrors the
# CONTRIBUTING.md instructions; test requires a built graph_data.pt
# (run `make data` first) and the torch/torch-geometric stack.
verify: lint format-check test
	@echo "All quality gates passed"

# ── Dashboard ─────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

# ── Utilities ─────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

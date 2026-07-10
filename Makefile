.PHONY: all setup data baselines gnn evaluate explain test typecheck lint format format-check verify dashboard clean

PYTHON := python

# ── One-shot pipeline ─────────────────────────────────────────────
all: data baselines gnn evaluate explain test

# ── Environment ───────────────────────────────────────────────────
setup:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# ── Data pipeline ─────────────────────────────────────────────────
data:
	$(PYTHON) -m graphguard.download_data
	$(PYTHON) -m graphguard.build_graph

# ── Modeling ──────────────────────────────────────────────────────
baselines:
	$(PYTHON) -m graphguard.train_baseline

gnn:
	$(PYTHON) -m graphguard.train_gnn --model all

# ── Evaluation ────────────────────────────────────────────────────
evaluate:
	$(PYTHON) -m graphguard.evaluate

# ── GNN explainability (GNNExplainer) ─────────────────────────────
explain:
	$(PYTHON) -m graphguard.explain_gnn

# ── Quality gates ─────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -q --cov=graphguard --cov-report=term-missing --cov-fail-under=15

typecheck:
	mypy src/graphguard

lint:
	ruff check src/ tests/ dashboard/

format:
	ruff format src/ tests/ dashboard/

format-check:
	ruff format --check src/ tests/ dashboard/

verify: lint format-check typecheck test
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

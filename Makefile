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
	$(PYTHON) -m pytest tests/ -q --cov=graphguard --cov-report=term-missing --cov-fail-under=30

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
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
	$(PYTHON) -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache') if p.is_dir()]"
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ruff_cache') if p.is_dir()]"

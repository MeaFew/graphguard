.PHONY: all setup data baselines gnn evaluate test dashboard clean

PYTHON := python

# ── One-shot pipeline ─────────────────────────────────────────────
all: data baselines gnn evaluate test

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

# ── Quality gates ─────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -q

lint:
	ruff check scripts/ tests/ dashboard/

format:
	ruff format scripts/ tests/ dashboard/

format-check:
	ruff format --check scripts/ tests/ dashboard/

# ── Dashboard ─────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

# ── Utilities ─────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

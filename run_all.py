"""End-to-end pipeline runner for graphguard."""

import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 mode on Windows for child processes
os.environ.setdefault("PYTHONUTF8", "1")


PYTHON = sys.executable


def run(cmd: str, cwd: Path | None = None):
    print(f"\n{'=' * 60}")
    print(f">>> {cmd}")
    print("=" * 60)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
        return False
    return True


def main():
    here = Path(__file__).resolve().parent

    steps = [
        ("Download / generate data", f"{PYTHON} scripts/download_data.py"),
        ("Build graph", f"{PYTHON} scripts/build_graph.py"),
        ("Train baselines", f"{PYTHON} scripts/train_baseline.py"),
        ("Train GNNs", f"{PYTHON} scripts/train_gnn.py --model all"),
        ("Evaluate", f"{PYTHON} scripts/evaluate.py"),
        ("Test", f"{PYTHON} -m pytest tests/ -q"),
    ]

    print("GraphGuard — Full Pipeline")
    print("=" * 60)

    for name, cmd in steps:
        if not run(cmd, cwd=here):
            print(f"\nPipeline stopped at step: {name}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Run the dashboard with: make dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()

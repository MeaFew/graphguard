"""Download the Elliptic Data Set.

Tries Kaggle Hub first, then falls back to a synthetic graph so the project
can run end-to-end without API credentials.
"""

import sys
from pathlib import Path

try:
    from config import RAW_DATA_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import RAW_DATA_DIR


def download_from_kaggle(output_dir: Path) -> bool:
    """Attempt to download the Elliptic Data Set from Kaggle."""
    try:
        import kagglehub

        print("Attempting to download Elliptic Data Set from Kaggle...")
        path = kagglehub.dataset_download(
            "ellipticco/elliptic-data-set",
            path=str(output_dir),
        )
        print(f"Downloaded to: {path}")
        return True
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False


def main():
    output_dir = RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    required = [
        output_dir / "elliptic_txs_features.csv",
        output_dir / "elliptic_txs_edgelist.csv",
        output_dir / "elliptic_txs_classes.csv",
    ]
    if all(f.exists() for f in required):
        print("Elliptic data files already exist. Skipping download.")
        return

    if not download_from_kaggle(output_dir):
        print("\nFalling back to synthetic graph generation...")
        from scripts.generate_synthetic_graph import generate_synthetic_data

        generate_synthetic_data(output_dir=output_dir)


if __name__ == "__main__":
    main()

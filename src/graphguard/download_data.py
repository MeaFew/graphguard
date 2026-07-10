"""Download the Elliptic Data Set.

Tries Kaggle Hub first, then falls back to a synthetic graph so the project
can run end-to-end without API credentials.
"""

import shutil
from pathlib import Path

from graphguard.config import RAW_DATA_DIR
from graphguard.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def download_from_kaggle(output_dir: Path) -> bool:
    """Attempt to download the Elliptic Data Set from Kaggle.

    Returns True on success, False on any expected failure (missing
    ``kagglehub`` dependency, network error, or missing Kaggle credentials).
    Catches specific exception types so a genuine programming error is not
    silently swallowed and reported as "download failed".

    The ``path`` parameter of ``kagglehub.dataset_download`` refers to a path
    *inside* the dataset, not the local output directory. We therefore download
    to kagglehub's cache directory and copy the three required CSVs into
    ``output_dir``.
    """
    try:
        import kagglehub

        logger.info("Attempting to download Elliptic Data Set from Kaggle...")
        cache_dir = kagglehub.dataset_download("ellipticco/elliptic-data-set")
        logger.info(f"Downloaded to cache: {cache_dir}")

        required_files = {
            "elliptic_txs_features.csv": output_dir / "elliptic_txs_features.csv",
            "elliptic_txs_edgelist.csv": output_dir / "elliptic_txs_edgelist.csv",
            "elliptic_txs_classes.csv": output_dir / "elliptic_txs_classes.csv",
        }

        cache_path = Path(cache_dir)
        missing = []
        for fname, dest in required_files.items():
            src = cache_path / fname
            if not src.exists():
                # kagglehub may return a parent directory or a zip-extracted
                # subdirectory; search one level deep for the file.
                candidates = list(cache_path.rglob(fname))
                if not candidates:
                    missing.append(fname)
                    continue
                src = candidates[0]
            shutil.copy2(src, dest)
            logger.info(f"  Copied {fname} -> {dest}")

        if missing:
            logger.info(f"Kaggle download failed: missing required files {missing}")
            return False
        return True
    except ImportError:
        logger.info("Kaggle download failed: 'kagglehub' is not installed.")
        logger.info("  Fix: pip install kagglehub  (and configure Kaggle credentials).")
        return False
    except (OSError, ConnectionError) as e:
        # OSError covers network/socket errors raised by the underlying HTTP
        # client on connection failures.
        logger.info(f"Kaggle download failed (network): {type(e).__name__}: {e}")
        return False
    except RuntimeError as e:
        # kagglehub raises RuntimeError for credential / HTTP-status problems.
        logger.info(f"Kaggle download failed (credentials/server): {type(e).__name__}: {e}")
        logger.info("  Set KAGGLE_USERNAME/KAGGLE_KEY or place ~/.kaggle/kaggle.json.")
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
        logger.info("Elliptic data files already exist. Skipping download.")
        return

    if not download_from_kaggle(output_dir):
        # Make the synthetic fallback LOUD and unmissable: any results obtained on
        # the synthetic graph are not representative of the real Elliptic task,
        # so a user must not unknowingly train on it.
        logger.info("\n" + "=" * 72)
        logger.info("WARNING: REAL DATA NOT AVAILABLE — generating a SYNTHETIC graph.")
        logger.info("Synthetic results are NOT comparable to the real Elliptic benchmark.")
        logger.info("Configure Kaggle credentials and re-run to use the real dataset.")
        logger.info("=" * 72 + "\n")
        from graphguard.generate_synthetic_graph import generate_synthetic_data

        generate_synthetic_data(output_dir=output_dir)


if __name__ == "__main__":
    setup_logging()
    main()

#!/usr/bin/env bash
# Download the Elliptic Data Set (Kaggle) into data/raw/.
#
# Requires Kaggle credentials: either ~/.kaggle/kaggle.json or the
# KAGGLE_USERNAME / KAGGLE_KEY environment variables. If credentials are not
# configured, scripts/build_graph.py will fall back to a synthetic transaction
# graph (scripts/generate_synthetic_graph.py) so the pipeline still runs.
set -euo pipefail
DEST_DIR="$(dirname "$0")"
cd "${DEST_DIR}"

if [ -f data/raw/elliptic_txs_features.csv ]; then
  echo "Elliptic data already present in data/raw/. Skipping download."
  echo "Remove it and re-run to force re-download."
  exit 0
fi

echo "Downloading Elliptic Data Set from Kaggle ..."
python scripts/download_data.py

echo "Done. Run 'python scripts/build_graph.py --force' to build the graph."

#!/usr/bin/env bash
# Download the Elliptic Data Set (Kaggle) into data/raw/.
#
# Requires Kaggle credentials: either ~/.kaggle/kaggle.json or the
# KAGGLE_USERNAME / KAGGLE_KEY environment variables. If credentials are not
# configured, graphguard.build_graph will fall back to a synthetic transaction
# graph (graphguard.generate_synthetic_graph) so the pipeline still runs.
set -euo pipefail
DEST_DIR="$(dirname "$0")"
cd "${DEST_DIR}"

if [ -f data/raw/elliptic_txs_features.csv ]; then
  echo "Elliptic data already present in data/raw/. Skipping download."
  echo "Remove it and re-run to force re-download."
  exit 0
fi

echo "Downloading Elliptic Data Set from Kaggle ..."
python -m graphguard.download_data

echo "Done. Run 'python -m graphguard.build_graph --force' to build the graph."

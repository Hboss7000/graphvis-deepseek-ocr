#!/usr/bin/env bash
# setup_data.sh
#
# Fetches the QA-GNN preprocessed ConceptNet + OpenBookQA release and places
# it at data_preprocessed_release/{cpnet,obqa}, matching the paths expected by
# scripts/generate_graphvis_datasets.py:
#   data_preprocessed_release/cpnet/concept.txt
#   data_preprocessed_release/obqa/graph/{split}.graph.adj.pk
#   data_preprocessed_release/obqa/statement/{split}.statement.jsonl
#
# This does NOT re-host or redistribute the data itself. It clones the
# official QA-GNN repo (Yasunaga et al., NAACL 2021) and runs their own
# download_preprocessed_data.sh, then copies just the cpnet/ and obqa/
# folders we need into this project.
#
# NOTE: this has not been fully verified end-to-end by the author of this
# script. QA-GNN's data is hosted externally (Stanford NLP group servers);
# if the download step fails, check https://github.com/michiyasunaga/qagnn
# for the current download instructions/links.
#
# Usage:
#   ./setup_data.sh [target_dir]
#
# target_dir defaults to ./data_preprocessed_release

set -euo pipefail

TARGET_DIR="${1:-data_preprocessed_release}"
QAGNN_REPO="https://github.com/michiyasunaga/qagnn.git"

command -v git >/dev/null 2>&1 || { echo "git is required but not found." >&2; exit 1; }

if [ -d "$TARGET_DIR/cpnet" ] && [ -d "$TARGET_DIR/obqa" ]; then
    echo "Found existing $TARGET_DIR/cpnet and $TARGET_DIR/obqa — nothing to do."
    echo "Delete or rename $TARGET_DIR if you want to re-fetch."
    exit 0
fi

WORKDIR="$(mktemp -d)"
echo "Working in temporary directory: $WORKDIR"
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "Cloning QA-GNN repo (shallow) for its official download script..."
git clone --depth 1 "$QAGNN_REPO" "$WORKDIR/qagnn"

cd "$WORKDIR/qagnn"

if [ ! -f "./download_preprocessed_data.sh" ]; then
    echo "ERROR: download_preprocessed_data.sh not found in the cloned repo." >&2
    echo "The QA-GNN repo may have changed. Check https://github.com/michiyasunaga/qagnn manually." >&2
    exit 1
fi

echo "Running QA-GNN's official download_preprocessed_data.sh..."
echo "(This downloads ConceptNet + CSQA + OBQA preprocessed data; it may take a while and use several GB.)"
chmod +x ./download_preprocessed_data.sh
./download_preprocessed_data.sh

if [ ! -d "./data/cpnet" ] || [ ! -d "./data/obqa" ]; then
    echo "ERROR: expected ./data/cpnet and ./data/obqa after download, but didn't find them." >&2
    echo "Check the QA-GNN repo's current instructions -- the download layout may have changed." >&2
    exit 1
fi

echo "Copying cpnet/ and obqa/ into $TARGET_DIR ..."
mkdir -p "$OLDPWD/$TARGET_DIR"
cp -r ./data/cpnet "$OLDPWD/$TARGET_DIR/"
cp -r ./data/obqa "$OLDPWD/$TARGET_DIR/"

cd "$OLDPWD"

echo ""
echo "Done. Verify with:"
echo "  ls $TARGET_DIR/cpnet/concept.txt"
echo "  ls $TARGET_DIR/obqa/graph/train.graph.adj.pk"
echo "  ls $TARGET_DIR/obqa/statement/train.statement.jsonl"
echo ""
echo "Note: the QA-GNN download also includes data/csqa/ (CommonsenseQA)."
echo "This project only needs cpnet/ and obqa/, so csqa/ was left out of the copy."
echo "The temporary clone (including csqa/) has been deleted."

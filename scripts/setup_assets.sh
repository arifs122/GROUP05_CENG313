#!/usr/bin/env bash
#
# Copy the competition assets that we are NOT allowed to redistribute into
# git-ignored folders.
#
# Usage:
#   bash scripts/setup_assets.sh [CG_DIR] [DATA_DIR]
#
# Defaults assume the Kaggle sample submission and card CSVs are in ./pokemon/
# (the folder used during the initial exploration).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CG_SRC="${1:-$ROOT/pokemon/sample_submission/sample_submission/sample_submission/cg}"
DATA_SRC="${2:-$ROOT/pokemon}"

if [[ ! -d "$CG_SRC" ]]; then
  cat >&2 <<EOF
Engine folder not found: $CG_SRC

Download the competition sample submission from Kaggle and pass its cg/
folder explicitly:

  bash scripts/setup_assets.sh /path/to/sample_submission/cg

The engine is competition-use-only and must not be committed to git.
EOF
  exit 1
fi

mkdir -p "$ROOT/vendor"
rm -rf "$ROOT/vendor/cg"
cp -R "$CG_SRC" "$ROOT/vendor/cg"
echo "engine copied to vendor/cg"

mkdir -p "$ROOT/data"
copied=0
for name in EN_Card_Data_R2_full.csv JP_Card_Data_R2_full.csv; do
  if [[ -f "$DATA_SRC/$name" ]]; then
    cp "$DATA_SRC/$name" "$ROOT/data/$name"
    copied=$((copied + 1))
  fi
done
if [[ "$copied" -gt 0 ]]; then
  echo "card data copied to data/"
else
  echo "warning: card CSVs not found in $DATA_SRC (data analysis will be limited)"
fi

echo
echo "Next steps:"
echo "  python3 scripts/check_env.py"
echo "  python3 tools/arena.py --games 4"

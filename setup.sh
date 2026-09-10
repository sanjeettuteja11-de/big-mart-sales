#!/usr/bin/env bash
# One-time environment setup.
#
# The virtualenv lives OUTSIDE the project: this folder sits in iCloud Drive,
# and a .venv here would sync thousands of package files and make imports crawl.
#
# On macOS, the LightGBM and XGBoost wheels look for libomp (OpenMP) only in
# Homebrew's folders. Without Homebrew they fail to import, so this script
# points them at the copy of libomp that scikit-learn already ships.
#
# Usage:  ./setup.sh

set -euo pipefail

VENV="${VENV_PATH:-$HOME/.venvs/big-mart-sales}"
PYTHON="${PYTHON:-python3}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating virtualenv at $VENV"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip --quiet
"$VENV/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt" --quiet

if [[ "$(uname)" == "Darwin" ]]; then
  SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  for entry in "lightgbm:lightgbm/lib/lib_lightgbm.dylib" "xgboost:xgboost/lib/libxgboost.dylib"; do
    module="${entry%%:*}"
    lib="$SITE/${entry#*:}"
    if ! "$VENV/bin/python" -c "import $module" 2>/dev/null && [[ -f "$lib" ]]; then
      echo "Pointing $module at scikit-learn's bundled libomp"
      install_name_tool -add_rpath @loader_path/../../sklearn/.dylibs "$lib" 2>/dev/null || true
      codesign --force --sign - "$lib"
    fi
  done
fi

"$VENV/bin/python" -c "import lightgbm, xgboost, catboost; print('LightGBM, XGBoost and CatBoost import OK')"

cat <<EOF

Environment ready. Next:
  1. Put the Analytics Vidhya train and test CSVs in $PROJECT_DIR/data/raw/
  2. $VENV/bin/python -m src.train
EOF

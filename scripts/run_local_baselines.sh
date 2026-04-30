#!/bin/bash
# ============================================================
# Run the BS baseline (A1) locally — no GPU needed, ~10 sec.
# Use this for the BS column of the master benchmark table.
#
# Usage:
#   bash scripts/run_local_baselines.sh
# ============================================================

set -e

# Find Python — prefer the conda env if available
if [ -x "$HOME/miniconda3/envs/pinn_env/bin/python" ]; then
    PYTHON="$HOME/miniconda3/envs/pinn_env/bin/python"
elif [ -x "/opt/homebrew/Caskroom/miniforge/base/envs/pinn_env/bin/python" ]; then
    PYTHON="/opt/homebrew/Caskroom/miniforge/base/envs/pinn_env/bin/python"
else
    PYTHON="python"
fi

cd "$(dirname "$0")/.."

echo "=== BS baseline (A1) ==="
$PYTHON run_bs_baseline.py --output_dir results/bs_baseline
echo ""
echo "Saved to results/bs_baseline/results.json"

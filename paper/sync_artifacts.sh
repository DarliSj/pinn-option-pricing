#!/bin/bash
# Refresh every figure the paper references.
# Called from the project root:  bash paper/sync_artifacts.sh
# Idempotent — safe to re-run any time results refresh.
set -e

cd "$(dirname "$0")/.."        # project root
mkdir -p paper/figures

# 1. Rebuild the master table (drives the table chunks AND the
#    Pareto / stratified figures generated below)
echo "→ rebuilding master table…"
python scripts/build_master_table.py >/dev/null

# 2. Generate matplotlib figures (architecture + Pareto + stratified +
#    walk-forward timeline + stability curves + val curves)
echo "→ regenerating matplotlib figures…"
python paper/scripts/generate_figures.py

# 3. (Optional) compile the TikZ vector version of the architecture
#    diagram if pdflatex is on PATH. The matplotlib version (PNG) is
#    canonical; the TikZ PDF is a higher-fidelity swap-in.
if command -v pdflatex >/dev/null 2>&1 && [ -f paper/figures/architecture.tex ]; then
    echo "→ compiling TikZ architecture diagram…"
    (cd paper/figures && pdflatex -interaction=nonstopmode architecture.tex >/dev/null \
        && rm -f architecture.aux architecture.log)
fi

# Stage 1 representative diagnostics (B10 = best-RMSE warm-start config)
B10=runs/walk_forward/modified_hybrid_mu0.75_warmup5000
B12=runs/walk_forward/modified_hybrid_mu0.25_warmup5000
for fold in Apr2020 Aug2020 Nov2020 Dec2020; do
  for config_dir in "$B10" "$B12"; do
    name=$(basename "$config_dir")
    src="$config_dir/fold_${fold}.png"
    [ -f "$src" ] && cp -f "$src" "paper/figures/stage1_${name}_${fold}.png" || true
  done
done

# Stage 2 vol surfaces (the headline plots for §5.2)
for warm in B10 B12; do
  for vol in cvol avol; do
    for fold in Nov2020 Dec2020; do
      src="runs/stage2_${warm}/${vol}/fold_${fold}_vol_surface.png"
      [ -f "$src" ] && cp -f "$src" "paper/figures/stage2_vol_surface_${warm}_${vol}_${fold}.png" || true
    done
  done
done

# Test-set scatters
for warm in B10 B12; do
  for vol in cvol avol; do
    src="runs/stage2_${warm}/${vol}/fold_Nov2020_test_scatter.png"
    [ -f "$src" ] && cp -f "$src" "paper/figures/stage2_scatter_${warm}_${vol}_Nov2020.png" || true
  done
done

echo "Synced $(ls paper/figures/*.png 2>/dev/null | wc -l | xargs) figures into paper/figures/"

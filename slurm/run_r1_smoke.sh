#!/bin/bash
# ============================================================
# R1 smoke: validates the regime-conditioning path ON GPU before the
# 6-task array commits ~30 GPU-hours.
#
# Checks (both already pass on CPU locally):
#   - 3-input Fourier embedding builds and trains on CUDA
#   - nu tensors land on the right device (colloc / TC / BC / data)
#   - the no-nu path is unchanged alongside it
#   - results.json + checkpoint record regime_input
# Output -> runs/v2/smoke/ (gitignored transient).
#
# Submit:  sbatch slurm/run_r1_smoke.sh
# (Usually launched indirectly by slurm/submit_r1.sh as the gate.)
# ============================================================

#SBATCH --job-name=pinn_r1_smoke
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/r1_smoke_%j.out
#SBATCH --error=slurm/logs/r1_smoke_%j.err

set -e

cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

echo "Node: $SLURMD_NODENAME  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

for REG in none atm_iv_lag; do
    EXTRA=""; [ "$REG" != "none" ] && EXTRA="--regime_input $REG"
    echo ""
    echo "=== R1 smoke: regime_input=$REG ==="
    python run_walk_forward.py \
        --arch modified --mode hybrid --rwf_mu 0.75 \
        --epochs 1500 --data_loss_warmup 200 \
        --val_months 1 --folds Nov2020 \
        --balancer relobralo --track_test_curve \
        --output_dir runs/v2/smoke/r1 $EXTRA
done

echo ""
echo "=== verifying the 3-input net was actually built ==="
python - <<'PY'
import json, torch, glob
for d in sorted(glob.glob('runs/v2/smoke/r1/*')):
    ck = torch.load(f'{d}/fold_Nov2020.pt', map_location='cpu', weights_only=False)
    B = ck['model_state_dict']['embedding.B']
    print(f"  {d.split('/')[-1]:<55} B={tuple(B.shape)}  regime={ck['run_info'].get('regime_input')}")
PY

echo ""
echo "R1 smoke done: $(date)"

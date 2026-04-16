#!/bin/bash
# ============================================================
# Quick smoke test: one fold, 500 epochs, one config.
# Run this first to verify GPU + env are working before
# submitting the full benchmark array.
#
# Submit with:
#   sbatch slurm/run_smoke_test.sh
# ============================================================

#SBATCH --job-name=pinn_smoke
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/smoke_%j.out
#SBATCH --error=slurm/logs/smoke_%j.err

cd /hpc/group/fisherlab/ds555/pinn_code
source /hpc/group/fisherlab/ds555/miniconda3/etc/profile.d/conda.sh
conda activate pinn_env

echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date:  $(date)"

# Verify torch sees the GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"

# One fold, 500 epochs — should finish in ~2 min
python run_walk_forward.py \
    --arch modified \
    --mode hybrid \
    --rwf_mu 0.75 \
    --epochs 500 \
    --folds Nov2020 \
    --output_dir runs/smoke

echo "Smoke test done: $(date)"

#!/bin/bash
#SBATCH --job-name=cdp_fixed
#SBATCH --output=logs/cdp_fixed_%j.out
#SBATCH --error=logs/cdp_fixed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

module load cuda/11.7.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdp-tabdiff

nvidia-smi || echo "nvidia-smi not available"

# Retrain only the (fixed) CDP-TabDiff model, then re-run the unified
# evaluation against the already-generated baseline CSVs in results/.
python src/benchmarking/benchmark_cdp_tabdiff.py "$@"

echo "=== Running unified evaluation ==="
python src/benchmarking/evaluate_all.py

#!/bin/bash
# =============================================================================
# Step 5 — evaluate a fine-tuned adapter (or the pretrained baseline) on the
# HDF5 test set and report WER. Needs one GPU. Set model_ckpt_path + path_to_data
# in the chosen eval config first. Generic SLURM template — fill in every <...>:
#     sbatch slurm/evaluate.sh
# =============================================================================
#SBATCH --job-name=asr_eval
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load <CUDA_MODULE>
source "<CONDA_BASE>/etc/profile.d/conda.sh"
conda activate <CONDA_ENV>

export MODEL_PATH=<MODEL_PATH>
export DATA_PATH=<DATA_PATH>

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Smoke test first (5 batches):
# python -m evaluation.evaluate -c configs/eval/small_hailmary.config --max_eval_batches 5

python -u -m evaluation.evaluate -c configs/eval/small_hailmary.config

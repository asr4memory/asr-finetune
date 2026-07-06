#!/bin/bash
# =============================================================================
# Step 3 — compute the pretrained model's per-shard baseline WER.
# Writes src/trainers/data/validation_summary_<tag>.csv, which the eval_wer_diff
# training objective is measured against. Needs one GPU.
# Generic SLURM template — fill in every <...>, then: sbatch slurm/compute_baseline.sh
# =============================================================================
#SBATCH --job-name=asr_baseline
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

python -u -m evaluation.compute_baseline_wer --model_type whisper-small

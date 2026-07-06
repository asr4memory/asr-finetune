#!/bin/bash
# =============================================================================
# EXAMPLE ONLY — a concrete, working configuration of slurm/train_hpo.sh for the
# FU Berlin "curta" cluster (H100 / A100 scavenger partition). Copy the generic
# template and adapt; usernames are read from $USER so nothing is hard-coded.
# =============================================================================
#SBATCH --job-name=asr_hpo_small
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --partition=scavenger
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load CUDA/12.6.0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate asr-finetune

export MODEL_PATH="/scratch/$USER"
export DATA_PATH="/scratch/$USER/datasets"
SCRATCH="/scratch/$USER"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export VALIDATION_SUMMARY_CSV="$PWD/src/trainers/data/validation_summary_ws_frac0.05.csv"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR" "$SCRATCH/ray_results" "$SCRATCH/optuna"

python -u -m train_hyper \
    -c configs/train/small_hailmary_phase1.config \
    --storage_path   "$SCRATCH/ray_results" \
    --optuna_db_path "$SCRATCH/optuna/small_wer_diff.db"

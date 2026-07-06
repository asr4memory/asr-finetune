#!/bin/bash
# =============================================================================
# EXAMPLE ONLY — a concrete configuration of slurm/train_hpo.sh for an
# NHR@ZIB A100 cluster (module stack differs from curta). Copy the generic
# template and adapt; usernames come from $USER so nothing is hard-coded.
# =============================================================================
#SBATCH --job-name=asr_hpo_large
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpu-a100:shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:4
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load NHRZIBenv
module load sw.a100.el9
module load cuda/12.9
module load anaconda3/2023.09
conda activate asr-finetune

export MODEL_PATH="/scratch/usr/$USER"
export DATA_PATH="/scratch/usr/$USER/data"
SCRATCH="/scratch/usr/$USER"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export VALIDATION_SUMMARY_CSV="$PWD/src/trainers/data/validation_summary.csv"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="/scratch/usr/$USER/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR" "$SCRATCH/ray_results" "$SCRATCH/optuna"

python -u -m train_hyper \
    -c configs/train/medium_hailmary_phase1.config \
    --storage_path   "$SCRATCH/ray_results" \
    --optuna_db_path "$SCRATCH/optuna/medium_wer_diff.db"

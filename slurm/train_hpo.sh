#!/bin/bash
# =============================================================================
# Step 4 — Whisper LoRA/DoRA hyper-parameter optimisation (Ray Tune + Optuna).
# Generic SLURM template — fill in every <...>, then: sbatch slurm/train_hpo.sh
# Re-submitting the same script resumes (resume_training=True in the config).
# For a concrete filled-in version see slurm/examples/{curta,nhr}_train_hpo.sh.
# =============================================================================
#SBATCH --job-name=asr_hpo
#SBATCH --time=72:00:00
#SBATCH --mem=80G
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# --- Environment (edit for your cluster) -------------------------------------
module load <CUDA_MODULE>                       # e.g. CUDA/12.6.0
source "<CONDA_BASE>/etc/profile.d/conda.sh"    # e.g. $HOME/miniconda3
conda activate <CONDA_ENV>                       # e.g. asr-finetune

# --- Paths (point at fast scratch on a cluster) ------------------------------
export MODEL_PATH=<MODEL_PATH>   # holds <model_type>/{model,processor,tokenizer,feature_extractor}
export DATA_PATH=<DATA_PATH>     # holds the Parquet train/val shards
SCRATCH=<SCRATCH>                # writable scratch for Ray results + the Optuna DB

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Per-shard pretrained-WER baseline the eval_wer_diff objective is measured against.
# Must match the model + eval_sample_fraction of the chosen config (see README step 3).
export VALIDATION_SUMMARY_CSV="$PWD/src/finetuning/trainers/data/validation_summary_ws_frac0.05.csv"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$SCRATCH/ray_results" "$SCRATCH/optuna"

python -u -m finetuning.train_hyper \
    -c configs/train/small_hailmary_phase1.config \
    --storage_path   "$SCRATCH/ray_results" \
    --optuna_db_path "$SCRATCH/optuna/small_wer_diff.db"

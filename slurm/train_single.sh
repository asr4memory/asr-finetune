#!/bin/bash
# =============================================================================
# Single fixed-config PEFT run — reproduce the best HPO trial deterministically
# (no Ray Tune search). Generic SLURM template — fill in every <...>, then:
#     sbatch slurm/train_single.sh
# =============================================================================
#SBATCH --job-name=asr_single
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

module load <CUDA_MODULE>
source "<CONDA_BASE>/etc/profile.d/conda.sh"
conda activate <CONDA_ENV>

export MODEL_PATH=<MODEL_PATH>
export DATA_PATH=<DATA_PATH>
SCRATCH=<SCRATCH>

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export VALIDATION_SUMMARY_CSV="$PWD/src/finetuning/trainers/data/validation_summary_ws_frac0.05.csv"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$SCRATCH/ray_results"

python -u -m finetuning.train_single_peft \
    -c configs/train/small_hailmary_phase1.config \
    --storage_path "$SCRATCH/ray_results" \
    --auto_resume_from_output_dir

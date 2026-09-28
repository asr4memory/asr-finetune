#!/bin/bash
# =============================================================================
# Step 2 — materialize an HDF5 corpus into sharded Parquet features.
# CPU/Ray-heavy; a GPU is not required. Run one split at a time (train/val/test).
# Generic SLURM template — fill in every <...>, then: sbatch slurm/prepare_data.sh
# =============================================================================
#SBATCH --job-name=asr_prepare
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# --- Environment (edit for your cluster) -------------------------------------
module load <CUDA_MODULE>                       # e.g. CUDA/12.6.0
source "<CONDA_BASE>/etc/profile.d/conda.sh"    # e.g. $HOME/miniconda3
conda activate <CONDA_ENV>                       # e.g. asr-finetune

export MODEL_PATH=<MODEL_PATH>   # holds <model_type>/{model,processor,tokenizer,feature_extractor}
export DATA_PATH=<DATA_PATH>     # input *.h5 corpora + output Parquet shards

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# Settings live in configs/prepare/materialize.config; override per split on the CLI.
# Repeat per split: train_parquet / val_parquet / test_parquet
python -u -m prepare_data.materialize_dataset -c configs/prepare/materialize.config \
    --split train_parquet

#!/bin/bash
# =============================================================================
# Step 1 — download a base Whisper model into $MODEL_PATH/<model_type>/.
# Run on a node with internet access (usually a login node); no GPU or SLURM
# allocation required. Edit the <...> values, then: bash slurm/download_model.sh
# =============================================================================
set -euo pipefail

export MODEL_PATH=<MODEL_PATH>          # parent dir for the model sub-directory

cd "$(dirname "$0")/.."                  # repo root
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python src/scripts/download_hf_model.py \
    --model_id openai/whisper-large-v3 \
    --output_dir "$MODEL_PATH/whisper-large-v3"

# Also download the WER metric once so training/eval can run offline later:
python -c "import evaluate; evaluate.load('wer')"

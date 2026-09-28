#!/usr/bin/env python3
"""
Compute pretrained-model baseline WER on the fine-tuning val shards.

eval_wer_diff = fine_tuned_WER - baseline_WER (per shard)

baseline_WER must be the SAME model that will be fine-tuned, in pretrained state.
This script computes that baseline and saves a validation_summary_<tag>.csv that
projects_paths.py picks up via the VALIDATION_SUMMARY_CSV env var.

Usage:
    python compute_baseline_wer.py --model_type whisper-small
    python compute_baseline_wer.py --model_type whisper-medium
    python compute_baseline_wer.py --model_type whisper-tiny

Model resolution:
    MODEL_PATH/<model_type>/  with sub-dirs: model/, processor/, feature_extractor/, tokenizer/
    (matches models/whisper_models.py::get_model_and_processor)

Dataset:
    For tiny/small/medium (80-dim): eq_complete_openai_whisper-small/val_parquet/
    For large-v2/large-v3 (128-dim): eg_dataset_openai_whisper_largev3/val_parquet/

Output:
    trainers/data/validation_summary_<tag>.csv
    columns: shard, eval_loss, eval_wer, n_samples
    (same schema as validation_summary.csv; eval_loss left blank)
"""

import argparse
import os
import glob
import csv
import re
import numpy as np
import torch
import pandas as pd
from jiwer import wer as jiwer_wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="whisper-small",
                    help="subdirectory under MODEL_PATH containing the model")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--language", type=str, default="german")
parser.add_argument("--task", type=str, default="transcribe")
parser.add_argument("--num_beams", type=int, default=1,
                    help="greedy (1) is fast; beam=5 adds ~5x latency for ~0.5% WER gain")
args_cli = parser.parse_args()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_REPO_ROOT, "models_local"))
DATA_PATH  = os.environ.get("DATA_PATH",  os.path.join(_REPO_ROOT, "data"))
MODEL_TYPE = args_cli.model_type
BATCH_SIZE = args_cli.batch_size
LANGUAGE   = args_cli.language
TASK       = args_cli.task
NUM_BEAMS  = args_cli.num_beams

# Choose val parquet dir and n_mels based on model family.
# tiny/base/small/medium → 80-dim; large/large-v*/CrisperWhisper → 128-dim
IS_LARGE = any(x in MODEL_TYPE for x in ("large", "Large"))
N_MELS   = 128 if IS_LARGE else 80

if IS_LARGE:
    VAL_DIR = f"{DATA_PATH}/eg_dataset_openai_whisper_largev3/val_parquet"
else:
    VAL_DIR = f"{DATA_PATH}/eq_complete_openai_whisper-small/val_parquet"

# Short tag for the output CSV (e.g. "whisper-small" → "ws", "whisper-medium" → "wm")
_TAG_MAP = {
    "whisper-tiny":   "tiny",
    "whisper-base":   "base",
    "whisper-small":  "ws",
    "whisper-medium": "wm",
    "whisper-large":  "wlv2",
    "whisper-large-v2": "wlv2",
    "whisper-large-v3": "wlv3",
}
TAG = _TAG_MAP.get(MODEL_TYPE, MODEL_TYPE.replace("/", "_").replace("-", "_"))

# This file lives at src/evaluation/; the baseline CSVs live at
# src/finetuning/trainers/data/, which is what projects_paths.VALIDATION_SUMMARY_CSV
# points at.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_CSV = os.path.join(SRC_DIR, "finetuning", "trainers", "data", f"validation_summary_{TAG}.csv")

def normalize(text):
    def _one(t):
        return re.sub(r"[!\?\.,;]", "", t.strip().lower())
    if isinstance(text, list):
        return [_one(t) for t in text]
    return _one(text)

# ── load model ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = os.path.join(MODEL_PATH, MODEL_TYPE)
print(f"model_type : {MODEL_TYPE}")
print(f"model_dir  : {model_dir}")
print(f"val_dir    : {VAL_DIR}")
print(f"n_mels     : {N_MELS}")
print(f"output_csv : {OUTPUT_CSV}")
print(f"device     : {device}\n")

processor = WhisperProcessor.from_pretrained(
    f"{model_dir}/processor",
    local_files_only=True,
    language=LANGUAGE,
    task=TASK,
)
model = WhisperForConditionalGeneration.from_pretrained(
    f"{model_dir}/model",
    dtype=torch.float16,
    local_files_only=True,
)
model = model.to(device).eval()
print("model ready.\n")

# ── find val parquets ─────────────────────────────────────────────────────────
parquet_files = sorted(glob.glob(f"{VAL_DIR}/*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"no parquet files in {VAL_DIR}")
print(f"found {len(parquet_files)} parquet files:")
for pf in parquet_files:
    print(f"  {os.path.basename(pf)}")
print()

# ── per-shard inference ───────────────────────────────────────────────────────
def to_feat_tensor(f):
    """Normalise a single feature array to (n_mels, 3000)."""
    a = np.array(f, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(N_MELS, -1)        # flat storage: (N_MELS*3000,) → (N_MELS, 3000)
    elif a.ndim == 2 and a.shape[0] != N_MELS:
        a = a.T                           # transposed: (3000, N_MELS) → (N_MELS, 3000)
    return a

results = []

for shard_idx, pfile in enumerate(parquet_files, 1):
    shard_name = f"val_{shard_idx}"
    fname = os.path.basename(pfile)
    print(f"=== {shard_name} ({fname}) ===")

    df = pd.read_parquet(pfile)
    n  = len(df)
    f0 = np.array(df["input_features"].iloc[0], dtype=np.float32)
    print(f"  {n} samples  raw_feat_shape={f0.shape}")

    all_refs = []
    all_hyps = []
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for bi in range(n_batches):
        rows = df.iloc[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]

        feats_np = np.stack([to_feat_tensor(f) for f in rows["input_features"].tolist()])
        input_features = torch.from_numpy(feats_np).to(device, dtype=torch.float16)

        pad_id = processor.tokenizer.pad_token_id
        decoded_refs = []
        for lab in rows["labels"].tolist():
            lab_clean = [t if t != -100 else pad_id for t in lab]
            decoded_refs.append(processor.tokenizer.decode(lab_clean, skip_special_tokens=True))
        all_refs.extend(normalize(decoded_refs))

        with torch.no_grad():
            pred_ids = model.generate(
                input_features,
                language=LANGUAGE,
                task=TASK,
                num_beams=NUM_BEAMS,
                return_timestamps=False,
            )
        all_hyps.extend(normalize(
            processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ))

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            print(f"  batch {bi+1}/{n_batches}")

    wer_val = 100.0 * jiwer_wer(all_refs, all_hyps)
    print(f"  WER = {wer_val:.4f}%\n")
    results.append({"shard": shard_name, "eval_loss": "",
                    "eval_wer": round(wer_val, 6), "n_samples": n})

# ── write CSV ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["shard", "eval_loss", "eval_wer", "n_samples"])
    writer.writeheader()
    writer.writerows(results)

print(f"saved {len(results)} rows → {OUTPUT_CSV}")
for r in results:
    print(f"  {r['shard']}: WER={r['eval_wer']:.4f}%  n={r['n_samples']}")

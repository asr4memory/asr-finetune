#!/usr/bin/env python3
"""
Project Paths Configuration

Defines and centralizes key directory paths used throughout the Whisper ASR project.
Paths are built dynamically based on the current user's environment variables.

This script is typically imported wherever consistent access to model, data,
results, and training directories is needed.

Paths defined:
- PROJECT_ROOT: Root folder for the current user’s local project
- MODEL_PATH: Path to the Whisper model directory (on shared scratch space)
- TRAINERS_PATH: Subdirectory containing trainer logic/scripts
- DATA_PATH: Base directory for datasets (HDF5, Parquet, etc.)
- RESULTS_PATH: Output directory for metrics, logs, and checkpoint summaries
"""
from pathlib import Path
import os

# Derive PROJECT_ROOT from this file's location so the repo can live anywhere.
# PROJECT_ROOT is the finetuning package directory (this file sits at
# src/finetuning/projects_paths.py). Override via env var ASR_FINETUNE_ROOT.
PROJECT_ROOT = os.environ.get(
    "ASR_FINETUNE_ROOT",
    str(Path(__file__).resolve().parent),
)

# Repository root (two levels above the finetuning package: src/finetuning -> src
# -> repo). Used only for the portable default fallbacks below.
REPO_ROOT = str(Path(__file__).resolve().parents[2])

# Base directory holding the pre-downloaded Whisper models, one sub-directory per
# model type (e.g. MODEL_PATH/whisper-large-v3/{model,processor,tokenizer,
# feature_extractor}). On an HPC cluster point MODEL_PATH at fast scratch
# storage; the default keeps everything inside the checkout for local runs.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(REPO_ROOT, "models_local"),
)

# Directory containing training modules or script entry points
TRAINERS_PATH = os.path.join(PROJECT_ROOT, "trainers")

# Centralized path to input datasets (HDF5, Parquet, etc.). Override with
# DATA_PATH on a cluster (fast scratch); defaults to <repo>/data locally.
DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(REPO_ROOT, "data"),
)

# Output directory for results like checkpoints, logs, and evaluations
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

# Canonical location of the per-shard whisper-large-v3 baseline CSV. Used as the
# default for validation_summary_csv across all evaluators so the path is never
# hardcoded to a specific user's $HOME.
# Override via env var when running a different model's HPO (e.g. whisper-small
# needs validation_summary_ws.csv with val_1..val_10 instead of val_1..val_20).
VALIDATION_SUMMARY_CSV = os.environ.get(
    "VALIDATION_SUMMARY_CSV",
    os.path.join(TRAINERS_PATH, "data", "validation_summary.csv"),
)

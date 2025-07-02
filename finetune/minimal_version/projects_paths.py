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

# Define base path for the current user's project (e.g. /home/username/minimal_version/)
PROJECT_ROOT = "/home/" + os.getenv('USER') + "/minimal_version/"

# Define path to model directory on shared scratch storage
MODEL_PATH = os.path.join("/scratch/usr/", os.getenv('USER') + "/whisper-large-v3")

# Directory containing training modules or script entry points
TRAINERS_PATH = os.path.join(PROJECT_ROOT, "trainers")

# Centralized path to input datasets (HDF5, Parquet, etc.)
DATA_PATH = "/scratch/usr/bemchrvt/data/"

# Output directory for results like checkpoints, logs, and evaluations
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

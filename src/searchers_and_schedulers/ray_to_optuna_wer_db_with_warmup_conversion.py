#!/usr/bin/env python3
"""
Import Ray Tune trials into an Optuna study (SQLite),
filtering to linear scheduler + batch_size=8,
and converting warmup_steps -> warmup_ratio with rounding diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import optuna
from optuna.trial import TrialState
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
)

# ---------------------------
# Constants / policy
# ---------------------------

TOTAL_STEPS = 2 * 20131
ALLOWED_WARMUP_RATIOS = [0.01, 0.03, 0.05, 0.1]
FIXED_BATCH_SIZE = 8
REQUIRED_SCHEDULER = "linear"

# ---------------------------
# Utilities
# ---------------------------

def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out


def iter_result_json_paths(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        if "result.json" in filenames:
            yield os.path.join(dirpath, "result.json")


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def nearest_bucket(value: float, buckets: List[float]) -> float:
    return min(buckets, key=lambda b: abs(b - value))

# ---------------------------
# Trial record
# ---------------------------

@dataclass
class TrialRecord:
    trial_id: str
    params: Dict[str, Any]
    eval_wer: float
    warmup_steps: int
    warmup_ratio_raw: float
    warmup_ratio_bucketed: float
    source_path: str

# ---------------------------
# Parsing logic
# ---------------------------

def parse_trial(result_path: str) -> Optional[TrialRecord]:
    best_wer = None
    trial_id = None
    params = None

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = safe_json_loads(line)
            if not obj:
                continue

            if trial_id is None and "trial_id" in obj:
                trial_id = obj["trial_id"]

            if params is None and "config" in obj:
                flat = flatten_dict(obj["config"])
                params = {f"config.{k}": v for k, v in flat.items()}

            if "eval_wer" in obj:
                val = float(obj["eval_wer"])
                best_wer = val if best_wer is None else min(best_wer, val)

    if trial_id is None or params is None or best_wer is None:
        return None

    # ---- filters ----
    scheduler = params.get("config.train_loop_config.lr_scheduler_type")
    batch_size = params.get("config.train_loop_config.per_device_train_batch_size")
    warmup_steps = params.get("config.train_loop_config.warmup_steps")

    if scheduler != REQUIRED_SCHEDULER:
        return None
    if batch_size != FIXED_BATCH_SIZE:
        return None
    if warmup_steps is None:
        return None

    warmup_steps = int(warmup_steps)
    warmup_ratio_raw = warmup_steps / TOTAL_STEPS
    warmup_ratio_bucketed = nearest_bucket(warmup_ratio_raw, ALLOWED_WARMUP_RATIOS)

    # ---- rewrite params ----
    params = dict(params)  # copy
    params.pop("config.train_loop_config.warmup_steps")
    params["config.train_loop_config.warmup_ratio"] = warmup_ratio_bucketed

    return TrialRecord(
        trial_id=trial_id,
        params=params,
        eval_wer=best_wer,
        warmup_steps=warmup_steps,
        warmup_ratio_raw=warmup_ratio_raw,
        warmup_ratio_bucketed=warmup_ratio_bucketed,
        source_path=result_path,
    )

# ---------------------------
# Optuna insertion
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ray_root", required=True)
    ap.add_argument("--db_path", required=True)
    ap.add_argument("--study_name", required=True)
    args = ap.parse_args()

    storage = f"sqlite:///{os.path.abspath(args.db_path)}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    records: List[TrialRecord] = []

    for path in iter_result_json_paths(args.ray_root):
        rec = parse_trial(path)
        if rec:
            records.append(rec)

    print(f"\nFound {len(records)} eligible trials\n")

    for r in records:
        print(
            f"{r.trial_id} | "
            f"warmup_steps={r.warmup_steps:5d} | "
            f"ratio_raw={r.warmup_ratio_raw:.5f} -> "
            f"bucketed={r.warmup_ratio_bucketed}"
        )

    print("\nAdding trials to Optuna study...\n")

    for r in records:
        dists = {
            "config.train_loop_config.learning_rate": FloatDistribution(5e-6, 1e-4, log=True),
            "config.train_loop_config.weight_decay": FloatDistribution(1e-6, 1e-2, log=True),
            "config.train_loop_config.warmup_ratio": CategoricalDistribution(ALLOWED_WARMUP_RATIOS),
            "config.train_loop_config.per_device_train_batch_size": CategoricalDistribution([8]),
            "config.train_loop_config.lr_scheduler_type": CategoricalDistribution(["linear"]),
        }

        trial = optuna.trial.create_trial(
            params=r.params,
            distributions=dists,
            value=r.eval_wer,
            state=TrialState.COMPLETE,
        )

        try:
            study.add_trial(trial)
        except Exception as e:
            print(f"[SKIP] {r.trial_id}: {e}")

    print(
        f"\nDone. Study now has {len(study.trials)} trials. "
        f"Best eval_wer = {study.best_value:.4f}"
    )


if __name__ == "__main__":
    main()


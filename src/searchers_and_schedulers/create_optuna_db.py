#!/usr/bin/env python3
"""
Convert Ray Tune trial result.json logs into an Optuna study (SQLite DB),
using a chosen evaluation metric as the objective and configs as parameters.

Folder pattern expected (example):
  .../ray_results/v3_large_jun/<trial_dir>/result.json

Each result.json is JSONL (one JSON object per line).
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
    BaseDistribution,
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
)

# ---------------------------
# Utilities
# ---------------------------

def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dicts into a single-level dict with dot-separated keys."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


def iter_result_json_paths(root: str) -> Iterable[str]:
    """Yield absolute paths to all result.json files under root."""
    for dirpath, _, filenames in os.walk(root):
        if "result.json" in filenames:
            yield os.path.join(dirpath, "result.json")


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
        return None
    except json.JSONDecodeError:
        return None


# ---------------------------
# Space handling
# ---------------------------

def load_space_json(path: str) -> Dict[str, BaseDistribution]:
    """
    Load a search-space description from JSON.

    Expected format (example):
    {
      "config.train_loop_config.per_device_train_batch_size": {"type":"int","low":2,"high":16,"step":2},
      "config.train_loop_config.learning_rate": {"type":"float","low":1e-6,"high":1e-4,"log":true},
      "config.train_loop_config.lr_scheduler_type": {"type":"categorical","choices":["linear","cosine"]},
      "config.train_loop_config.warmup_steps": {"type":"int","low":0,"high":500}
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("space.json must be a JSON object mapping param_name -> spec")

    dists: Dict[str, BaseDistribution] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"Invalid spec for {name}: {spec}")

        t = spec["type"].lower()
        if t == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))
            step = spec.get("step", None)
            step_f = None if step is None else float(step)
            dists[name] = FloatDistribution(low=low, high=high, log=log, step=step_f)
        elif t == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            log = bool(spec.get("log", False))
            step = int(spec.get("step", 1))
            dists[name] = IntDistribution(low=low, high=high, log=log, step=step)
        elif t == "categorical":
            choices = spec["choices"]
            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError(f"categorical choices must be non-empty list for {name}")
            dists[name] = CategoricalDistribution(choices=choices)
        else:
            raise ValueError(f"Unknown distribution type '{t}' for {name}")

    return dists


def infer_distribution_from_value(v: Any) -> BaseDistribution:
    """
    Fallback: create a degenerate distribution that only allows exactly v.

    WARNING: This can cause "inconsistent parameter distribution" later if you
    continue the study with broader ranges. Prefer providing --space.
    """
    if isinstance(v, bool):
        return CategoricalDistribution([v])
    if isinstance(v, int) and not isinstance(v, bool):
        return IntDistribution(low=v, high=v, step=1, log=False)
    if isinstance(v, float):
        return FloatDistribution(low=v, high=v, step=None, log=False)
    # strings / None / others:
    return CategoricalDistribution([v])


# ---------------------------
# Parsing Ray trial logs
# ---------------------------

@dataclass
class TrialRecord:
    trial_id: str
    params: Dict[str, Any]
    objective_value: float
    source_path: str
    source_status: str
    training_iteration: int


def infer_ray_trial_status(result_path: str, saw_done: bool, saw_metric: bool) -> str:
    trial_dir = os.path.dirname(result_path)
    error_path = os.path.join(trial_dir, "error.txt")

    if os.path.exists(error_path):
        return "ERROR"
    if saw_done:
        return "TERMINATED"
    if saw_metric:
        return "INCOMPLETE"
    return "UNKNOWN"


def parse_trial_from_result_json(
    result_path: str,
    metric_key: str = "eval_wer",
    mode: str = "min",  # "min" or "max"
    pick: str = "best", # "best" or "last"
) -> Optional[TrialRecord]:
    """
    Parse a single result.json and return a TrialRecord for that file.
    Uses trial_id and config from the JSONL records.

    pick="best": take min/max over all eval_wer entries
    pick="last": take the last line that contains eval_wer
    """
    best_val: Optional[float] = None
    last_val: Optional[float] = None
    trial_id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    max_training_iteration = 0
    saw_done = False
    saw_metric = False

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = safe_json_loads(line)
            if not obj:
                continue

            if trial_id is None and "trial_id" in obj:
                trial_id = str(obj["trial_id"])

            # Config is typically stable; keep the first seen
            if params is None and "config" in obj and isinstance(obj["config"], dict):
                # Flatten and prefix with "config." so it’s unambiguous
                flat = flatten_dict(obj["config"])
                params = {f"config.{k}": v for k, v in flat.items()}

            if obj.get("done") is True:
                saw_done = True

            if "training_iteration" in obj:
                try:
                    max_training_iteration = max(max_training_iteration, int(obj["training_iteration"]))
                except (TypeError, ValueError):
                    pass

            if metric_key in obj and obj[metric_key] is not None:
                try:
                    val = float(obj[metric_key])
                except (TypeError, ValueError):
                    continue
                saw_metric = True
                last_val = val
                if best_val is None:
                    best_val = val
                else:
                    if mode == "min":
                        best_val = min(best_val, val)
                    else:
                        best_val = max(best_val, val)

    if trial_id is None or params is None:
        return None

    chosen = best_val if pick == "best" else last_val
    if chosen is None:
        return None

    source_status = infer_ray_trial_status(
        result_path=result_path,
        saw_done=saw_done,
        saw_metric=saw_metric,
    )

    return TrialRecord(
        trial_id=trial_id,
        params=params,
        objective_value=float(chosen),
        source_path=result_path,
        source_status=source_status,
        training_iteration=max_training_iteration,
    )


def collect_trials(
    root: str,
    metric_key: str = "eval_wer",
    mode: str = "min",
    pick: str = "best",
    include_errored_with_metric: bool = False,
    min_training_iteration: int = 0,
) -> List[TrialRecord]:
    """
    Collect trials across all result.json files.
    Deduplicate by trial_id (keep the best among duplicates).
    """
    by_id: Dict[str, TrialRecord] = {}

    for path in iter_result_json_paths(root):
        rec = parse_trial_from_result_json(path, metric_key=metric_key, mode=mode, pick=pick)
        if rec is None:
            continue
        if rec.training_iteration < min_training_iteration:
            continue
        if rec.source_status == "ERROR" and not include_errored_with_metric:
            continue

        existing = by_id.get(rec.trial_id)
        if existing is None:
            by_id[rec.trial_id] = rec
        else:
            # Keep the better record for the objective
            if mode == "min":
                if rec.objective_value < existing.objective_value:
                    by_id[rec.trial_id] = rec
            else:
                if rec.objective_value > existing.objective_value:
                    by_id[rec.trial_id] = rec

    return list(by_id.values())


# ---------------------------
# Writing Optuna DB
# ---------------------------

def create_or_load_study(storage_url: str, study_name: str, direction: str) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        load_if_exists=True,
    )


def add_trials_to_study(
    study: optuna.Study,
    trials: List[TrialRecord],
    space: Optional[Dict[str, BaseDistribution]] = None,
) -> Tuple[int, int]:
    """
    Add trials to the Optuna study.
    Returns (added, skipped).
    """
    added = 0
    skipped = 0

    for rec in trials:
        # Build distributions: prefer provided "space", otherwise infer degenerate ones
        if space is not None:
            dists: Dict[str, BaseDistribution] = {}
            ok = True
            for k, v in rec.params.items():
                if k not in space:
                    # If your space.json doesn't cover all params, we skip this trial
                    ok = False
                    break
                dists[k] = space[k]
            if not ok:
                skipped += 1
                continue
        else:
            dists = {k: infer_distribution_from_value(v) for k, v in rec.params.items()}

        trial = optuna.trial.create_trial(
            params=rec.params,
            distributions=dists,
            value=rec.objective_value,
            state=TrialState.COMPLETE,
            user_attrs={
                "ray_trial_id": rec.trial_id,
                "source_path": rec.source_path,
                "source_status": rec.source_status,
                "source_training_iteration": rec.training_iteration,
            },
        )

        try:
            study.add_trial(trial)
            added += 1
        except Exception as e:
            # Common failure: distribution mismatch when study already has trials
            # We skip and continue.
            skipped += 1
            # Optional: print detail
            print(f"[SKIP] trial_id={rec.trial_id} from {rec.source_path}: {type(e).__name__}: {e}")

    return added, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ray_root", required=True, help="Path to Ray experiment root (contains TorchTrainer_* trial dirs).")
    ap.add_argument("--db_path", required=True, help="Output SQLite DB file path (e.g., /path/optuna_study.db).")
    ap.add_argument("--study_name", required=True, help="Optuna study name to create/load.")
    ap.add_argument("--mode", default="min", choices=["min", "max"], help="Optimize direction for eval_wer.")
    ap.add_argument("--metric_key", default="eval_wer_diff", help="Metric key in result.json (default: eval_wer_diff).")
    ap.add_argument("--pick", default="best", choices=["best", "last"], help="Pick best or last metric from each result.json.")
    ap.add_argument("--space", default=None, help="Optional JSON file defining distributions to avoid mismatch issues.")
    ap.add_argument("--include_errored_with_metric", action="store_true", help="Import Ray trials with error.txt as COMPLETE if they logged the requested metric.")
    ap.add_argument("--min_training_iteration", type=int, default=0, help="Only import trials that reached at least this training_iteration.")
    args = ap.parse_args()

    direction = "minimize" if args.mode == "min" else "maximize"
    storage_url = f"sqlite:///{os.path.abspath(args.db_path)}"

    print(f"[1/4] Scanning Ray results under: {args.ray_root}")
    trials = collect_trials(
        args.ray_root,
        metric_key=args.metric_key,
        mode=args.mode,
        pick=args.pick,
        include_errored_with_metric=args.include_errored_with_metric,
        min_training_iteration=args.min_training_iteration,
    )
    print(f"      Found {len(trials)} unique trial_id(s) with '{args.metric_key}' and config.")

    space = None
    if args.space:
        print(f"[2/4] Loading search space from: {args.space}")
        space = load_space_json(args.space)
        print(f"      Loaded {len(space)} param distributions.")
    else:
        print("[2/4] No --space provided. Will infer degenerate distributions (may not be extendable safely).")

    print(f"[3/4] Creating/loading Optuna study '{args.study_name}' in {args.db_path}")
    study = create_or_load_study(storage_url=storage_url, study_name=args.study_name, direction=direction)

    print("[4/4] Adding trials...")
    added, skipped = add_trials_to_study(study, trials, space=space)
    print(f"Done. Added={added}, Skipped={skipped}. Total trials in study now: {len(study.trials)}")
    print(f"Best value ({args.metric_key}) in study: {study.best_value}")


if __name__ == "__main__":
    main()

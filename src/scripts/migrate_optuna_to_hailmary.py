"""Re-file the old Optuna study under the new (hail-mary) search space.

Why: Phase 1 uses a narrower LR range and adds ``lora_dropout`` as a new
hyperparameter. The existing study (``cw_wer_diff_study``) was built under the
old space, so pointing Phase 1 at it directly would crash with
``distribution_mismatch`` the first time Optuna's TPE tries to sample a new
trial. This script reads the old trials, drops the ones whose params fall
outside the new ranges, and writes the survivors into a new study using the
new distributions.

Run on the login node (cheap, ~5 s):

    PYTHONPATH=src python -m scripts.migrate_optuna_to_hailmary \
        --old_db   "$OPTUNA_DIR/cw_wer_diff.db" \
        --old_name cw_wer_diff_study \
        --new_db   "$OPTUNA_DIR/cw_wer_diff.db" \
        --new_name cw_hailmary_phase1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is importable so we can read the canonical JSON space.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState


# ---------- the new (hail-mary Phase 1) search space ----------
# Keep in sync with searchers_and_schedulers/spaces_peft_hailmary.json and the
# hyperparameter wiring in ray_searchers_and_schedulers.py.
NEW_SPACE: dict[str, BaseDistribution] = {
    "config.train_loop_config.per_device_train_batch_size": CategoricalDistribution([8]),
    "config.train_loop_config.learning_rate":               FloatDistribution(low=1e-6, high=3e-5, log=True),
    "config.train_loop_config.warmup_ratio":                CategoricalDistribution([0.03, 0.05, 0.1, 0.15]),
    "config.train_loop_config.alpha":                       CategoricalDistribution([8, 16, 32]),
    "config.train_loop_config.target_r":                    CategoricalDistribution([4, 8, 12]),
    "config.train_loop_config.lora_dropout":                CategoricalDistribution([0.0, 0.05, 0.1]),
}

# Old trials predate the addition of lora_dropout; the trainer code used
# lora_dropout=0.05 hardcoded, so we treat that as the implicit value.
IMPLICIT_DEFAULTS: dict[str, object] = {
    "config.train_loop_config.lora_dropout": 0.05,
}


def in_distribution(value, dist: BaseDistribution) -> bool:
    """Return True iff ``value`` is sampleable under ``dist`` (narrowed space)."""
    if isinstance(dist, CategoricalDistribution):
        return value in dist.choices
    if isinstance(dist, FloatDistribution):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return dist.low <= v <= dist.high
    if isinstance(dist, IntDistribution):
        try:
            v = int(value)
        except (TypeError, ValueError):
            return False
        return dist.low <= v <= dist.high
    return False


def migrate(
    old_db: str,
    old_name: str,
    new_db: str,
    new_name: str,
    direction: str = "minimize",
    drop_existing_new: bool = False,
    dry_run: bool = False,
    new_space: dict[str, BaseDistribution] | None = None,
    implicit_defaults: dict[str, object] | None = None,
) -> dict:
    """Re-file ``old_name`` trials into ``new_name`` under the hail-mary space.

    Returns a stats dict: ``{seen, added, dropped_state, dropped_out_of_range,
    dropped_missing, drop_reasons}``.

    Safe to call repeatedly: ``optuna.create_study(load_if_exists=True)`` so
    re-running just no-ops if the destination already has the migrated trials
    (Optuna deduplicates by trial identity? — no, it doesn't; call only when
    the destination is empty or use ``drop_existing_new=True``).
    """
    space = new_space if new_space is not None else NEW_SPACE
    defaults = implicit_defaults if implicit_defaults is not None else IMPLICIT_DEFAULTS

    src_url = f"sqlite:///{Path(old_db).resolve()}"
    dst_url = f"sqlite:///{Path(new_db).resolve()}"

    print(f"[migrate] source: study='{old_name}' db='{src_url}'")
    print(f"[migrate] dest:   study='{new_name}' db='{dst_url}'")

    old_study = optuna.load_study(study_name=old_name, storage=src_url)
    print(f"[migrate] loaded source study with {len(old_study.trials)} trial(s)")

    if drop_existing_new and not dry_run:
        try:
            optuna.delete_study(study_name=new_name, storage=dst_url)
            print(f"[migrate] deleted existing destination study '{new_name}'")
        except Exception:
            pass

    new_study = None
    if not dry_run:
        new_study = optuna.create_study(
            study_name=new_name,
            storage=dst_url,
            direction=direction,
            load_if_exists=True,
        )

    n_seen = n_added = n_dropped_state = n_dropped_oor = n_dropped_missing = 0
    drop_reasons: dict[str, int] = {}

    for t in old_study.trials:
        n_seen += 1

        if t.state != TrialState.COMPLETE or t.value is None:
            n_dropped_state += 1
            continue

        new_params: dict[str, object] = {}
        new_dists: dict[str, BaseDistribution] = {}
        bad_reason = None
        for key, dist in space.items():
            if key in t.params:
                val = t.params[key]
            elif key in defaults:
                val = defaults[key]
            else:
                bad_reason = f"missing param: {key}"
                break

            if not in_distribution(val, dist):
                bad_reason = f"out-of-range {key}={val}"
                break

            new_params[key] = val
            new_dists[key] = dist

        if bad_reason is not None:
            if "missing" in bad_reason:
                n_dropped_missing += 1
            else:
                n_dropped_oor += 1
            drop_reasons[bad_reason] = drop_reasons.get(bad_reason, 0) + 1
            continue

        if dry_run:
            n_added += 1
            continue

        ft = optuna.trial.create_trial(
            params=new_params,
            distributions=new_dists,
            value=float(t.value),
            state=TrialState.COMPLETE,
            user_attrs={
                "migrated_from_study": old_name,
                "migrated_from_trial": int(t.number),
                **(t.user_attrs or {}),
            },
        )
        try:
            new_study.add_trial(ft)
            n_added += 1
        except Exception as e:
            n_dropped_state += 1
            drop_reasons[f"add_trial error: {type(e).__name__}"] = (
                drop_reasons.get(f"add_trial error: {type(e).__name__}", 0) + 1
            )

    print(f"[migrate] seen={n_seen}  added={n_added}  "
          f"dropped(state)={n_dropped_state}  "
          f"dropped(out_of_range)={n_dropped_oor}  "
          f"dropped(missing)={n_dropped_missing}")
    if drop_reasons:
        print("[migrate] drop reasons:")
        for reason, count in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:3d}  {reason}")

    if not dry_run and new_study is not None and n_added > 0:
        try:
            print(f"[migrate] new study best value: {new_study.best_value:.4f}")
            print(f"[migrate] new study best params: {new_study.best_params}")
        except ValueError:
            pass  # study has no completed trials with finite values

    return {
        "seen": n_seen,
        "added": n_added,
        "dropped_state": n_dropped_state,
        "dropped_out_of_range": n_dropped_oor,
        "dropped_missing": n_dropped_missing,
        "drop_reasons": drop_reasons,
    }


def maybe_auto_migrate(
    db_path: str,
    new_study_name: str,
    migrate_from_study: str | None,
    direction: str = "minimize",
) -> bool:
    """Auto-migrate iff the destination study is missing or empty AND a
    ``migrate_from_study`` is configured. Idempotent: re-running once the
    destination has trials is a fast no-op.

    Returns True if migration ran, False otherwise.
    """
    if not migrate_from_study:
        return False

    db_url = f"sqlite:///{Path(db_path).resolve()}"

    # destination already populated?
    try:
        dst = optuna.load_study(study_name=new_study_name, storage=db_url)
        n_complete = sum(1 for t in dst.trials if t.state == TrialState.COMPLETE)
        if n_complete > 0:
            print(f"[migrate] destination study '{new_study_name}' already has "
                  f"{n_complete} completed trial(s); skipping migration.")
            return False
    except (KeyError, ValueError):
        # study doesn't exist yet - that's fine, migrate will create it
        pass

    # source exists?
    try:
        optuna.load_study(study_name=migrate_from_study, storage=db_url)
    except (KeyError, ValueError) as e:
        print(f"[migrate] WARN: source study '{migrate_from_study}' not found in "
              f"{db_url} ({e!r}); skipping migration.")
        return False

    print(f"[migrate] auto-migrating '{migrate_from_study}' -> '{new_study_name}' in {db_url}")
    migrate(
        old_db=db_path, old_name=migrate_from_study,
        new_db=db_path, new_name=new_study_name,
        direction=direction,
    )
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--old_db", required=True)
    p.add_argument("--old_name", required=True)
    p.add_argument("--new_db", required=True)
    p.add_argument("--new_name", required=True)
    p.add_argument("--direction", default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--drop_existing_new", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    migrate(
        old_db=a.old_db, old_name=a.old_name,
        new_db=a.new_db, new_name=a.new_name,
        direction=a.direction,
        drop_existing_new=a.drop_existing_new,
        dry_run=a.dry_run,
    )


if __name__ == "__main__":
    main()

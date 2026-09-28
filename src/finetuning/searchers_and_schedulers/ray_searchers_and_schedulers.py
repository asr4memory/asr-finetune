from ray.tune.schedulers import ASHAScheduler
from finetuning.utils import steps_per_epoch, calculate_grace_period
import optuna
import os
import sys
from pathlib import Path

# Ray stuff
from ray import tune

# ``scripts/`` is importable because entry points put src/ on sys.path; this
# enables on-demand auto-migration of Optuna studies to a new search space.
from finetuning.searchers_and_schedulers.migrate_optuna_to_hailmary import maybe_auto_migrate as _maybe_auto_migrate

import logging

logger = logging.getLogger(__name__)


def _enable_sqlite_wal(db_path: str) -> None:
    """Switch a SQLite Optuna DB to WAL journaling with a generous busy timeout.

    WAL lets readers proceed without blocking the writer and (most relevant to
    parallel HPO) drastically reduces the ``database is locked`` errors that
    appear when multiple Ray Tune trials in one job report results at the same
    time. The PRAGMA is persistent — once set on a file it survives across
    connections — so this is a one-shot setup per DB file. A separate file per
    SLURM job is still the right way to handle cross-job concurrency.
    """
    import sqlite3
    try:
        con = sqlite3.connect(db_path, timeout=30.0)
        try:
            cur = con.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            mode = cur.fetchone()
            cur.execute("PRAGMA busy_timeout = 30000;")
            cur.close()
            con.commit()
            logger.info("Optuna SQLite at %s: journal_mode=%s, busy_timeout=30000ms",
                        db_path, mode[0] if mode else "?")
        finally:
            con.close()
    except Exception as e:
        logger.warning("Could not enable WAL on %s: %s. Continuing with default journal.",
                       db_path, e)


def get_searcher_and_scheduler(args):
    """
    Return the appropriate Ray Tune Search Algorithm and Scheduler for hyperparameter optimization.

    The function selects a search-scheduler pair depending on the problem size and tuning space:

    - 'small_small'            → Grid/random search with ASHAScheduler (for small-scale tasks with few HPs)
    - 'large_small_OPTUNA'     → OptunaSearch + ASHAScheduler (for large-scale tasks with small HP space)
    - 'large_large'            → Population Based Training (for large-scale tasks with large HP space)

    Notes:
        - `max_t_` defines the maximum number of training steps. This is used by all schedulers.
        - We calculate a grace period for early stopping using heuristics that take into account warmup duration.

    Returns:
        Tuple of (searcher, scheduler): Ray Tune searcher and scheduler objects

    Reference:
      https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-scheduler-pbt
      https://docs.ray.io/en/latest/tune/faq.html#how-does-early-termination-e-g-hyperband-asha-work
      https://docs.ray.io/en/latest/tune/faq.html#which-search-algorithm-scheduler-should-i-choose

    """
    # Compute total training steps based on dataset size and batch size
    max_t = steps_per_epoch(args.len_train_set, args.per_device_train_batch_size, gradient_accumulation_steps = args.gradient_accumulation_steps) * args.num_train_epochs
    max_t_ = max_t
    
    if logger.isEnabledFor(logging.DEBUG):
        grace_period = 1
    else:
        if args.num_samples > 1:
            grace_period = args.grace_period
            max_t_ = args.max_steps
        else:
            grace_period = max_t_
            args.max_steps = max_t
            
            #int(round(max_t * 0.1)) + 100  # start kick out trials after LR warmup finished
#    calculate_grace_period(max_t, warmup_steps = args.warmup_steps,
#                                          warmup_ratio = args.warmup_ratio,
#                                          max_warmup_steps = args.max_warmup_steps)
    
    # Log the configuration for early stopping
    # TODO: fix this max step max_step_ confusion
    logger.info(f"Early stopping after {max_t_} steps for scheduler.\n"
                f"Actualy number of steps: {max_t} \n"
                f"Fraction we train: {round(100 * max_t_ / max_t, 2)} \n \n"
                f"Grace Period before scheduler kicks in: {grace_period}")
    
    # --- Option 1: Basic Variant Generator + ASHAScheduler (Simple brute-force or grid search) ---
    if args.search_schedule_mode == 'small_small':
        from ray.tune.search.basic_variant import BasicVariantGenerator
        scheduler = ASHAScheduler(
            max_t=max_t_,
            reduction_factor=args.reduction_factor,
            grace_period=args.grace_period,
        )
        searcher = BasicVariantGenerator()
    # --- Option 2: OptunaSearch + ASHAScheduler (Bayesian Optimization with pruning) ---
    elif args.search_schedule_mode == 'large_small_OPTUNA':
        from ray.tune.search.optuna import OptunaSearch
        # https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-optuna
        scheduler = ASHAScheduler(
            time_attr="step",
            max_t=max_t_,
            reduction_factor=args.reduction_factor,
            grace_period=grace_period,
        )
        if args.optuna_db_path:
            db_path = args.optuna_db_path
            if not db_path.endswith(".db"):
                db_path = f"{db_path}.db"

            # Ensure the parent directory exists so the SQLite file can be created.
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Persistently switch this SQLite DB to WAL + long busy_timeout so
            # concurrent trial reports inside one job don't trip 'database is
            # locked'. Cross-job concurrency is handled by giving each model its
            # own DB file in the configs.
            _enable_sqlite_wal(db_path)

            study_name = args.optuna_study_name
            mode = args.modes[0][0]
            direction = "minimize" if mode == "min" else "maximize"

            # Auto-migrate the trials from a prior study (old search space) into
            # ``study_name`` under the current (hail-mary) distributions. No-op if
            # the destination study already has completed trials, or if
            # ``optuna_migrate_from`` was not provided.
            migrate_from = getattr(args, "optuna_migrate_from", None) or None
            if migrate_from:
                try:
                    _maybe_auto_migrate(
                        db_path=db_path,
                        new_study_name=study_name,
                        migrate_from_study=migrate_from,
                        direction=direction,
                    )
                except Exception as e:
                    logger.warning(
                        "Optuna auto-migration from '%s' to '%s' in %s failed: %s. "
                        "Continuing with whatever the destination study currently holds.",
                        migrate_from, study_name, db_path, e,
                    )

            storage_url = f"sqlite:///{db_path}"

            fresh_study = getattr(args, "fresh_study", False)
            if fresh_study:
                try:
                    optuna.delete_study(study_name=study_name, storage=storage_url)
                    logger.info("--fresh_study: deleted existing Optuna study '%s' from %s",
                                study_name, db_path)
                except KeyError:
                    logger.info("--fresh_study: study '%s' did not exist yet, nothing to delete.",
                                study_name)

            try:
                # Legacy Ray Tune API (~ <= 2.6): study_name + storage in __init__.
                storage = optuna.storages.RDBStorage(url=storage_url)
                optuna_searcher = OptunaSearch(
                    metric=args.metric_to_optimize[0][0],
                    mode=mode,
                    study_name=study_name,
                    storage=storage,
                )
                action = "Created fresh" if fresh_study else "Restoring"
                logger.info(f"{action} Optuna study '{study_name}' in {db_path} "
                            f"via legacy OptunaSearch(study_name=, storage=)")
            except TypeError:
                # Newer Ray Tune (~ 2.7+): OptunaSearch no longer accepts
                # study_name/storage. Pre-create the persistent study and late-bind
                # it onto the searcher's internal study slot so trials are written
                # back to the SQLite file across phases.
                direction = "minimize" if mode == "min" else "maximize"
                optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    direction=direction,
                    load_if_exists=True,
                )
                optuna_searcher = OptunaSearch(
                    metric=args.metric_to_optimize[0][0],
                    mode=mode,
                )
                persistent_study = optuna.load_study(
                    study_name=study_name, storage=storage_url
                )
                # `_ot_study` is the internal attr OptunaSearch reads/writes for
                # ask/tell. Setting it before the first ask() bypasses its default
                # in-memory study and gives us full SQLite persistence.
                optuna_searcher._ot_study = persistent_study
                action = "Created fresh" if fresh_study else "Restoring"
                logger.info(f"{action} Optuna study '{study_name}' in {db_path} "
                            f"via _ot_study override (new Ray Tune API)")
            if migrate_from:
                logger.info(f"  (auto-migrated from '{migrate_from}')")

            # IMPORTANT: do NOT wrap OptunaSearch in a ConcurrencyLimiter here.
            # OptunaSearch + ConcurrencyLimiter trickles trials in one per
            # previous-trial-setup-completion (~30s gap), severely limiting
            # parallelism. Let TuneConfig.max_concurrent_trials be the gate,
            # and let Optuna's ask() run as fast as it can.
            searcher = optuna_searcher
            return searcher, scheduler

        else:
            # fallback: fresh optuna (no warm-start)
            optuna_searcher = OptunaSearch(
                metric=args.metric_to_optimize[0][0],
                mode=args.modes[0][0],
            )
        # No ConcurrencyLimiter wrap - see comment in the if-branch above.
        # TuneConfig.max_concurrent_trials provides the parallelism cap.
        searcher = optuna_searcher
    # --- Option 3: Population Based Training (PBT) for wide search spaces ---
    elif args.search_schedule_mode == 'large_large':
        from ray.tune.schedulers import PopulationBasedTraining
        from ray.tune.search.basic_variant import BasicVariantGenerator
        # https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-scheduler-pbt
        # https://docs.ray.io/en/latest/tune/examples/pbt_guide.html
        scheduler = PopulationBasedTraining(
            time_attr='step',
            # defines the "time" as training iterations (steps in our case)...training_iteration is steps / tune.report() calls where tune.report()=args.save_steps in our case
            perturbation_interval=args.perturbation_interval,
            hyperparam_mutations={
                "train_loop_config": {
                    "learning_rate": tune.loguniform(1e-5, 1e-1),
                    "weight_decay": tune.uniform(0.0, 0.2),
                }
            }
        )
        searcher = BasicVariantGenerator()  # default searcher

    return searcher, scheduler


def get_whisper_hyperparameters(args):
    """
    Build the Ray Tune parameter search space for Whisper training.

    This function dynamically defines which hyperparameters should be tuned and their search distributions
    based on the command-line arguments passed via `args.hyperparameters`.

    Supported hyperparameters (defined in HYPERPARAMETERS list):
        - learning_rate: log-uniform in [1e-5, 1e-1]
        - warmup_steps: integer in [0, max_warmup_steps]
        - weight_decay: uniform in [0.0, 0.2]
        - batch_size: one of [1, 2, 4, 8] (affects `per_device_train_batch_size`)
        - scheduler: choice of ["linear", "cosine"]
        - alpha, rank: integers for PEFT-specific settings (e.g., LoRA)

    Returns:
        dict: Nested config dictionary for Ray Tune with parameter sampling strategies.
    """
    HYPERPARAMETERS = ['learning_rate', 'warmup_steps', 'warmup_ratio', 'weight_decay', 'batch_size', 'scheduler', 'alpha', 'alpha_coupled', 'rank',
    'warmup_steps_feb', 'learning_rate_feb', 'weight_decay_feb',
    'target_r', 'target_r_wide', 'lora_dropout', 'learning_rate_narrow', 'learning_rate_lora', 'learning_rate_lora_cw']
    train_loop_config_ = {}
    # Add default static batch size, unless overridden by tuning
    train_loop_config_["per_device_train_batch_size"] = args.per_device_train_batch_size

    print("Hyperparameters", args.hyperparameters)
    # Choose between fixed warmup steps or dynamic warmup ratio
    if args.warmup_steps == 0:
        logger.info(f"Will do LR warmup of {args.warmup_ratio}%")
        train_loop_config_["warmup_ratio"] = args.warmup_ratio
    else:
        logger.info(f"Will do LR warmup using {args.warmup_steps} steps")
        train_loop_config_["warmup_steps"] = args.warmup_steps
        
    # Dynamically build hyperparameter search space
    for hyper_param in args.hyperparameters[0]:
        
        if args.num_samples == 1:
            logger.debug("Skipping hyperparameter %s (num_samples=1)", hyper_param)
            continue
            
        logger.debug("Adding hyperparameter %s to the search space", hyper_param)
            
        assert hyper_param in HYPERPARAMETERS, (
            f"Hyperparameter search for {hyper_param} not implemented"
        )

        if hyper_param == 'learning_rate':
            train_loop_config_[hyper_param] = tune.loguniform(1e-6, 1e-4)

        elif hyper_param == 'learning_rate_feb':
            train_loop_config_['learning_rate'] = tune.loguniform(1e-6, 5e-4)

        elif hyper_param == 'learning_rate_narrow':
            # Phase-1 of the hail-mary HPO: cap LR to kill the loss-down / WER-up zone.
            train_loop_config_['learning_rate'] = tune.loguniform(1e-6, 3e-5)

        elif hyper_param == 'learning_rate_lora':
            # LoRA-appropriate LR band per community recipes (1e-4..1e-3 typical),
            # widened on the low end to keep cautious trials in play.
            train_loop_config_['learning_rate'] = tune.loguniform(1e-5, 1e-3)

        elif hyper_param == 'learning_rate_lora_cw':
            # CrisperWhisper-specific narrow band. CW's base weights are already
            # heavily fine-tuned (German + word-level timestamps); the community
            # LoRA 1e-4..1e-3 band that works for vanilla Whisper destabilises
            # CW within ~100 gradient steps and produces near-gibberish at
            # inference (train loss stays low but eval_wer explodes — observed
            # in v4 with eval_wer_diff up to +200). Prior successful CW runs
            # all sampled in 1e-6..1e-5; this keyword restores that band while
            # WLV3 stays on `learning_rate_lora`. See
            # notes/lora_dora_pissa_session_decisions.md §9.
            train_loop_config_['learning_rate'] = tune.loguniform(1e-6, 1e-5)

        elif hyper_param == 'lora_dropout':
            train_loop_config_[hyper_param] = tune.choice([0.0, 0.05, 0.1])
            
        elif hyper_param == 'warmup_ratio':
            # Dropped 0.1: with AdaLoRA tinit at 5% of max_steps, warmup_ratio=0.1
            # has warmup ending after pruning starts. RankAllocator then computes
            # importance scores during the overlap on warmup-modulated gradients
            # — exactly the v1-era pathology the v2 schedule (5/60) was meant to
            # avoid. 0.05 is the safe upper bound (warmup ends as pruning starts).
            train_loop_config_[hyper_param] = tune.choice([0.0, 0.01, 0.05])
        
        elif hyper_param == 'warmup_steps':
            train_loop_config_[hyper_param] = tune.choice([100, 500, 1000, 2000])
   
        elif hyper_param == 'warmup_steps_feb':
            train_loop_config_['warmup_steps'] = tune.randint(0, 29 + 1)
                                              
        elif hyper_param == 'batch_size':
            train_loop_config_["per_device_train_batch_size"] = tune.choice([1, 2, 4, 8, 16, 32, 64, 128, 256])
            
        elif hyper_param == 'alpha':
            train_loop_config_[hyper_param] = tune.choice([8, 16, 32, 64])

        elif hyper_param == 'alpha_coupled':
            # Sample an alpha-to-rank multiplier instead of alpha directly.
            # The trainer translates this into lora_alpha = multiplier * target_r,
            # which keeps the LoRA scaling (lora_alpha / r) in {1, 2} — the band
            # used by DoRA / HF PEFT Whisper / Swiss German + Yoruba ASR recipes.
            # Removes pathological pairings like alpha=8, target_r=32 (scale=0.25)
            # that were reachable under the independent alpha-rank sampling.
            train_loop_config_[hyper_param] = tune.choice([1, 2])
            
        elif hyper_param == 'rank':
            train_loop_config_[hyper_param] = tune.randint(1, 17)
        
        elif hyper_param == 'target_r':
            train_loop_config_[hyper_param] = tune.choice([4, 8, 12, 16])

        elif hyper_param == 'target_r_wide':
            # Wider LoRA rank choices for the WLV3 hail-mary (community recipes
            # use rank 8-32). Kept separate from `target_r` so the CrisperWhisper
            # Optuna study, whose stored distribution is [4, 8, 12, 16], is not
            # invalidated by a distribution-mismatch on resume.
            train_loop_config_['target_r'] = tune.choice([4, 8, 12, 16, 24, 32])
        
        elif hyper_param == 'weight_decay':
            train_loop_config_[hyper_param] = tune.loguniform(1e-6, 1e-2)
            
        elif hyper_param == 'weight_decay_feb':
            train_loop_config_['weight_decay'] = tune.uniform(0.0, 0.2)

        elif hyper_param == 'scheduler':
            # Options: add more if you want!
            # LINEAR = "linear"
            # COSINE = "cosine"
            # COSINE_WITH_RESTARTS = "cosine_with_restarts"
            # POLYNOMIAL = "polynomial"
            # CONSTANT = "constant"
            # CONSTANT_WITH_WARMUP = "constant_with_warmup"
            train_loop_config_["lr_scheduler_type"] = tune.choice(["linear", "cosine"])
    
    if args.num_samples == 1:
        logger.info(f"Single Model Training!")
        train_loop_config_["learning_rate"] = tune.choice([args.learning_rate])
        
        train_loop_config_["weight_decay"] = tune.choice([args.weight_decay])
        
        train_loop_config_["lr_scheduler_type"] = tune.choice([args.lr_scheduler_type])
        
    
    return train_loop_config_

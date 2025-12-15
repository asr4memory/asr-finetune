from ray.tune.schedulers import ASHAScheduler
from utils import steps_per_epoch, calculate_grace_period

# Ray stuff
from ray import tune

import logging

logger = logging.getLogger(__name__)


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
    max_t = steps_per_epoch(args.len_train_set, args.per_device_train_batch_size) * args.num_train_epochs
    max_t_ = args.max_steps
    
    if logger.isEnabledFor(logging.DEBUG):
        grace_period = 1
    else:
        grace_period = 5000#int(round(max_t * 0.1)) + 100  # start kick out trials after LR warmup finished
#    calculate_grace_period(max_t, warmup_steps = args.warmup_steps,
#                                          warmup_ratio = args.warmup_ratio,
#                                          max_warmup_steps = args.max_warmup_steps)
    
    # Log the configuration for early stopping
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
        # metric = args.metric_to_optimize, mode = args.modes
        searcher = tune.search.ConcurrencyLimiter(
            OptunaSearch(metric=args.metric_to_optimize[0], mode=args.modes[0]),
            max_concurrent=args.max_concurrent_trials
        )
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
    HYPERPARAMETERS = ['learning_rate', 'warmup_steps', 'warmup_ratio', 'weight_decay', 'batch_size', 'scheduler', 'alpha', 'rank']
    train_loop_config_ = {}
    # Add default static batch size, unless overridden by tuning
    train_loop_config_["per_device_train_batch_size"] = args.per_device_train_batch_size
    
    # Choose between fixed warmup steps or dynamic warmup ratio
    if args.warmup_steps == 0:
        logger.info(f"Will do LR warmup of {args.warmup_ratio}%")
        train_loop_config_["warmup_ratio"] = args.warmup_ratio
    else:
        logger.info(f"Will do LR warmup using {args.warmup_steps} steps")
        train_loop_config_["warmup_steps"] = args.warmup_steps
        
    # Dynamically build hyperparameter search space
    for hyper_param in args.hyperparameters[0]:
        logger.debug("Adding hyperparameter %s to the search space", hyper_param)
        assert hyper_param in HYPERPARAMETERS, logger.info("Hyperparameter search for %s not implemented", hyper_param)

        if hyper_param == 'learning_rate':
            train_loop_config_[hyper_param] = tune.loguniform(5e-6, 1e-4)
            
        elif hyper_param == 'warmup_ratio':
            train_loop_config_[hyper_param] = tune.choice([0.01, 0.03, 0.05, 0.1])
        
        elif hyper_param == 'warmup_steps':
            train_loop_config_[hyper_param] = tune.choice([100, 500, 1000, 2000])
                                              
        elif hyper_param == 'batch_size':
            train_loop_config_["per_device_train_batch_size"] = tune.choice([1, 2, 4, 8])
            
        elif hyper_param == 'alpha':
            train_loop_config_[hyper_param] = tune.randint(2, 6)
            
        elif hyper_param == 'rank':
            train_loop_config_[hyper_param] = tune.randint(1, 17)
            
        elif hyper_param == 'weight_decay':
            train_loop_config_[hyper_param] = tune.loguniform(1e-6, 1e-2)

        elif hyper_param == 'scheduler':
            # Options: add more if you want!
            # LINEAR = "linear"
            # COSINE = "cosine"
            # COSINE_WITH_RESTARTS = "cosine_with_restarts"
            # POLYNOMIAL = "polynomial"
            # CONSTANT = "constant"
            # CONSTANT_WITH_WARMUP = "constant_with_warmup"
            train_loop_config_["lr_scheduler_type"] = tune.choice(["linear", "cosine"])

    return train_loop_config_

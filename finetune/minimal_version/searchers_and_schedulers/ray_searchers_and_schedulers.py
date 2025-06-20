from ray.tune.schedulers import ASHAScheduler
from utils import steps_per_epoch

# Ray stuff
from ray import tune

import logging

logger = logging.getLogger(__name__)


def get_searcher_and_scheduler(args):
    """STEP 3: Returns Searcher and Scheduler Object

   A search algorithm searches the Hyperparameter space for good combinations.
   A scheduler can early terminate bad trials, pause trials, clone trials, and alter hyperparameters of a running
   trial.

   Different Algorithms for different problem settings are recommended based on whether you problem is small or
   large, and whether you have many hyperparameter to fine-tune or only a small. Based on that, we implement 4
   different scenarios:
      small_small: Small problem for small hyperparameters -> does a brute-force grid-search
      large_small_BOHB: Large problems (like for whisper) but small Hyperparameters (just a hand full) -> bayesian
                         optimization which is mathematically optimal
      large_small_OPTUNA: A variant of the above with a different searcher (but also bayes on Bayesian Optimization)
      large_large: Population based training for large problems with a  large Hyperparameter space

   Note:
       - The max_t_ paramater, which are the training steps, needs to be specified. However, this number depends on
       the per_device_train_batch_size which is why we can't search for the optimal batch size at the moment.
       - We got an error using the default bohb_search function. We fixed this error by defining our own
         TuneBOHB_fix.

   TODO:
    * Allow for batch size searching. Requires to restructure this function
    * Test and debug the different Schedulers. So far, I only really used large_small_BOHB

   Reference:
      https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-scheduler-pbt
      https://docs.ray.io/en/latest/tune/faq.html#how-does-early-termination-e-g-hyperband-asha-work
      https://docs.ray.io/en/latest/tune/faq.html#which-search-algorithm-scheduler-should-i-choose

   """
    max_t = steps_per_epoch(args.len_train_set, args.per_device_train_batch_size) * args.num_train_epochs
    max_t_ = args.max_steps

    logger.info(f"Early stopping after {max_t_} steps for scheduler.\n"
                f"Actualy number of steps: {max_t} \n"
                f"Fraction we train: {round(100 * max_t / max_t_, 2)}")

    if args.search_schedule_mode == 'small_small':
        from ray.tune.search.basic_variant import BasicVariantGenerator
        scheduler = ASHAScheduler(
            max_t=max_t_,
            reduction_factor=args.reduction_factor,
            grace_period=args.grace_period,
        )
        searcher = BasicVariantGenerator()

    elif args.search_schedule_mode == 'large_small_OPTUNA':
        from ray.tune.search.optuna import OptunaSearch
        # https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-optuna
        scheduler = ASHAScheduler(
            time_attr="step",
            max_t=max_t_,
            reduction_factor=args.reduction_factor,
            grace_period=args.grace_period,
        )
        # metric = args.metric_to_optimize, mode = args.modes
        searcher = tune.search.ConcurrencyLimiter(
            OptunaSearch(metric=args.metric_to_optimize[0], mode=args.modes[0]),
            max_concurrent=args.max_concurrent_trials
        )

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
    HYPERPARAMETERS = ['learning_rate', 'warmup_steps', 'weight_decay', 'batch_size', 'scheduler', 'alpha', 'rank']
    train_loop_config_ = {}
    train_loop_config_["per_device_train_batch_size"] = args.per_device_train_batch_size
    train_loop_config_["warmup_steps"] = 0

    for hyper_param in args.hyperparameters[0]:
        logger.debug("Adding hyperparameter %s to the search space", hyper_param)
        assert hyper_param in HYPERPARAMETERS, logger.info("Hyperparameter search for %s not implemented", hyper_param)

        if hyper_param == 'learning_rate':
            train_loop_config_[hyper_param] = tune.loguniform(1e-5, 1e-1)
        elif hyper_param == 'warmup_steps':
            train_loop_config_[hyper_param] = tune.randint(0, args.max_warmup_steps + 1)
        elif hyper_param == 'batch_size':
            train_loop_config_["per_device_train_batch_size"] = tune.choice([1, 2, 4, 8])
        elif hyper_param == 'alpha':
            train_loop_config_[hyper_param] = tune.randint(2, 6)
        elif hyper_param == 'rank':
            train_loop_config_[hyper_param] = tune.randint(1, 17)
        elif hyper_param == "weight_decay":
            train_loop_config_[hyper_param] = tune.uniform(0.0, 0.2)
        elif hyper_param == "scheduler":
            # Options: add more if you want!
            # LINEAR = "linear"
            # COSINE = "cosine"
            # COSINE_WITH_RESTARTS = "cosine_with_restarts"
            # POLYNOMIAL = "polynomial"
            # CONSTANT = "constant"
            # CONSTANT_WITH_WARMUP = "constant_with_warmup"
            train_loop_config_["lr_scheduler_type"] = tune.choice(["linear", "cosine"])

    return train_loop_config_

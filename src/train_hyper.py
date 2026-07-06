"""Main Hyper-Parameter Finetuning script for Whisper ASR using Ray Tune.

This script uses Hugging Face Transformers and Ray Tune to optimize training
of Whisper models for automatic speech recognition (ASR).

Training logic is based on the HF Whisper finetuning tutorial:
https://huggingface.co/blog/fine-tune-whisper

Ray Tune integration follows:
https://docs.ray.io/en/latest/train/getting-started-transformers.html

Overview of steps:
1. Define Ray Tune Tuner object with hyperparameters, scheduler, resources, etc.
2. Each trial runs train_model(), a wrapper around HF Seq2SeqTrainer.

Note:
To adapt to another task or model, modify get_models(), train_model(), and get_hyperparameters().
"""
# General
import os
import pprint
import numpy as np
from utils import list_of_strings, save_file
from transformers import set_seed
from functools import partial

# For code organization and reporting
import configargparse
import logging

# For hyperparameter optimization and ray stuff
import ray.train
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import  ScalingConfig, CheckpointConfig
from ray.tune import Tuner
# RunConfig moved between ray.tune / ray.train / ray.air across Ray releases.
try:
    from ray.tune import RunConfig            # Ray <= ~2.6
except ImportError:
    try:
        from ray.train import RunConfig        # Ray ~2.7+
    except ImportError:
        from ray.air import RunConfig          # Ray ~2.0-2.4

from trainers.trainers import make_seq2seq_training_kwargs as make_training_kwargs
from trainers.trainers import train_whisper_model, train_whisper_peft_model
from searchers_and_schedulers.ray_searchers_and_schedulers import get_searcher_and_scheduler
from searchers_and_schedulers.ray_searchers_and_schedulers import get_whisper_hyperparameters as get_hyperparameters
from searchers_and_schedulers.stoppers import DecorrelationStopper

# Datasets
from data_and_collator.datasets_and_collators import get_datasets_and_collators, make_dataset_kwargs
from projects_paths import DATA_PATH

# Logging control
os.environ["RAY_AIR_NEW_OUTPUT"] = "1"
os.environ["RAY_VERBOSITY"] = "1"
#os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
ray.data.context.DataContext.get_current().enable_operator_progress_bars = True
ray.data.context.DataContext.get_current().enable_progress_bars = True

logger = logging.getLogger(__name__)

# options for different ray searchers and scheduler (see the get_searcher_and_scheduler function for details)
TUNE_CHOICES = ['small_small', 'large_small_OPTUNA', 'large_large']
DATA_MODES = ['h5', 'parquet', 'parquet_h5']
def parse_args():
    """ Parses command line arguments for the training.

    In particular:
            model_type...Whisper model type choices: https://huggingface.co/models?search=openai/whisper
            Ray tune general options: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
            Ray Searcher options: https://docs.ray.io/en/latest/tune/api/schedulers.html#resourcechangingscheduler
    """
    parser = configargparse.ArgumentParser()

    # Training settings for Seq2SeqTrainingArguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="increase by 2x for every 2x decrease in batch size")
    # The only reason to use gradient accumulation steps is when your whole batch size does not fit on one GPU, so you pay a price in terms of speed to overcome a memory issue.
    parser.add_argument("--output_tag", type=str,
                        default="whisper-tiny-de",
                        help="Base directory where model is save.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max Number of gradient steps")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Max Number of gradient steps")
    # adalora_deltaT was used by the prior AdaLoRA path. Kept for backward
    # compat with any config still passing it (silently ignored).
    parser.add_argument("--adalora_deltaT", type=int, default=50,
                        help="Deprecated. Was AdaLoRA's rank-update interval; "
                             "ignored since the switch to plain LoRA + DoRA + PiSSA.")
    parser.add_argument("--generation_max_length", type=int, default=225, help="Max length of token output")
    parser.add_argument("--save_steps", type=int, default=1000, help="After how many steps to save the model?")
    # requires the saving steps to be a multiple of the evaluation steps
    parser.add_argument("--eval_steps", type=int, default=1000, help="After how many steps to evaluate model")
    parser.add_argument("--eval_delay", type=int, default=0, help="Wait eval_delay steps before evaluating.")
    # dataloader
    parser.add_argument("--dataloader_num_workers", type=int, default=1, help="How many CPUs to allocate for dataloaders")
    parser.add_argument("--logging_steps", type=int, default=25, help="After how many steps to do some logging")

    # model settings
    parser.add_argument("--model_type", type=str, default="openai/whisper-tiny", help="Model to optimize")
    parser.add_argument("--target_language", type=str, default="german", help="Target Language")
    parser.add_argument("--return_timestamps", action="store_true", help="Return Timestemps mode for model")
    parser.add_argument("--peft", action="store_true", help="Whether or not to do Parameter Efficient Training")
    parser.add_argument("--simple", action="store_true", help="Which Collator to use")
    #https://github.com/Vaibhavs10/fast-whisper-finetuning?tab=readme-ov-file#evaluation-and-inference

    # Dataset settings
    parser.add_argument("--test_split", type=float, default=0.2, help="Percentage of test data.")
    parser.add_argument("--h5", action="store_true", help="If data is in .h5 format")
    parser.add_argument("--data_mode", type=str, default="h5", choices=DATA_MODES, help="Target Language")


    ## Hyperparameter Optimization settings for Ray Tune
    # Training configs
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear warmup of LR.")
    parser.add_argument("--warmup_ratio", type=float, default=0.0,
                        help="Ratio of total training steps used for a linear warmup. Will only work when warmup_steps = 0 otherwise warmup_steps overides this.")
    parser.add_argument("--max_warmup_steps", type=int, default=10,
                        help="For Hyperparam search. What is the max value for warmup steps?")
    parser.add_argument("--len_train_set", type=int, default=10,
                        help="Helper variable to define max_t which is required for schedulers.")
    parser.add_argument("--max_concurrent_trials", type=int, default=1,
                        help="Maximum number of trials to run concurrently.")
    parser.add_argument("--prefetch_batches", type=int, default=1,
                        help="How many batches to prefetch data? Keep in mind: is using VRAM.")
    parser.add_argument("--optuna_db_path", type=str, default=None, help="Directory from which to restore Optuna Search (for better priors).")
    parser.add_argument("--optuna_study_name", type=str, default=None, help="Study name to load when reusing an Optuna SQLite DB.")
    parser.add_argument("--fresh_study", action="store_true", default=False,
                        help="Drop and recreate the Optuna study before starting. "
                             "Use when you want a truly fresh HPO run without inheriting "
                             "prior trial history from the DB.")
    parser.add_argument("--optuna_migrate_from", type=str, default=None,
                        help="If set, auto-migrate trials from this old study (built under the old search space) into "
                             "optuna_study_name (the hail-mary space) on first run. Idempotent: skipped once the "
                             "destination has trials.")
    parser.add_argument("--load_ds_in_trainer", action="store_true", default=False, help="Wheter to load the ds within the "
                                                                                         "trainer or outside. ")

    # tune options: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
    parser.add_argument("--num_samples", type=int, default=5, help="Number of times to sample from the hyperparameter space.")
    parser.add_argument("--num_to_keep", type=int, default=1, help="number of checkpoints to keep on disk")
    parser.add_argument("--max_t", type=int, default=10, help="Max number of steps for finding the best hyperparameters")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of trials that can run at the same time")
    parser.add_argument("--cpus_per_trial", type=int, default=1, help="Number of CPUs per Ray actor")
    parser.add_argument("--gpus_per_trial", type=float, default=0, help="Number of GPUs per Ray actor")
    parser.add_argument("--use_gpu", action="store_true", help="If using GPU for the finetuning")
    parser.add_argument("--fp16", action="store_true", default=False, help="Training with floating point 16 ")
    parser.add_argument("--reuse_actors", action="store_true", help="Reusing Ray Actors should accelarate training")
    parser.add_argument("--metric_to_optimize", type=list_of_strings, action ="append", help="Metric which is used for deciding what a good trial is.")
    parser.add_argument("--wer_weight", type=float, default=1.0, help="Weight of WER in eval_loss_wer metric to optimize.")
    parser.add_argument("--modes", type=list_of_strings, action ="append", help="Modes of metrics (mix or max). Position 1 refers to metric 1 etc.")
    parser.add_argument("--eval_sample_fraction", type=float, default=1.0, help="Fraction of Eval set per shard (e.g. 0.05 = ~5%).")
    parser.add_argument("--bf16", action="store_true", default=False, help="Train and evaluate in bf16 (preferred on A100).")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="Decay for adapter-EMA shadow used at evaluation.")
    parser.add_argument("--ema_start_step", type=int, default=0, help="Step at which EMA updates start (default: from step 0).")
    parser.add_argument("--shard_schedule", type=str, default="deterministic", choices=["deterministic", "random"],
                        help="How to pick the eval shard each call. deterministic = step-keyed round-robin (cleaner ASHA comparisons); random = legacy.")
    parser.add_argument("--decorr_stopper", action="store_true", default=False, help="Enable DecorrelationStopper that kills trials whose loss decreases while WER increases for 2 consecutive evals after grace_period.")
    parser.add_argument("--decorr_loss_key", type=str, default="eval_loss", help="Loss key the DecorrelationStopper watches.")
    parser.add_argument("--decorr_wer_key", type=str, default="eval_wer", help="WER key the DecorrelationStopper watches.")
    # For Scheduler and Search algorithm specifically: https://docs.ray.io/en/latest/tune/api/schedulers.html#resourcechangingscheduler
    parser.add_argument("--search_schedule_mode", type=str, default="large_small_BOHB", choices=TUNE_CHOICES, help="Which Searcher Algorithm and Scheduler combination. See 'get_searcher_and_scheduler' function for details.")
    parser.add_argument("--reduction_factor", type=int, default=2, help="Factor by which trials are reduced after grace_period of time intervals")
    parser.add_argument("--grace_period", type=int, default=5000, help="With grace_period=n you can force ASHA to train each trial at least for n time interval")
    # For Population Based Training (PBT): https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    parser.add_argument("--perturbation_interval", type=int, default=10, help="Models will be considered for perturbation at this interval of time_attr.")
    parser.add_argument("--burn_in_period", type=int, default=1, help="Grace Period for PBT")
    # which hyperparameter to search for for
    parser.add_argument('--hyperparameters', type=list_of_strings, action ="append", help="List of Hyperparameter to tune")
    # hyperparameters for single model training
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight Decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning Rate Scheduler Type")
    
    # Other settings
    parser.add_argument("--run_on_local_machine", action="store_true", help="Store true if training is on local machine.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base directory where outputs are saved.")
    parser.add_argument("--storage_path", type=str, default="./output/scratch", help="Where to store ray tune results. ")
    parser.add_argument("--resume_training", action="store_true", help="Whether or not to resume training.")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--path_to_data", type=str, default="",
                        help="Path to audio batch-prepared audio files if in debug mode. Otherwise: all data in datasets are loaded")
    parser.add_argument("--dataset_name", type=str, default="eg_dataset_subset_1000.h5",
                        help="Name of dataset")
    parser.add_argument("--random_seed", type=int, default=1337, help="Random Seed for reproducibility")
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")

    args = parser.parse_args()
    return args


import math
import ray

def partition_dataset(ds, fraction: float):
    """
    Split a Ray Dataset into ``ceil(1/fraction)`` equal-sized ``val_i`` shards.

    The HPO objective is a *running mean* of ``eval_wer_diff`` over the trial's
    evals (see Seq2SeqTrainerEvalSamplingPeft). Each eval call uses one shard
    chosen by step-keyed schedule, so over the trial all shards are visited and
    no validation data is wasted.

    Args:
        ds (ray.data.Dataset): The dataset to split.
        fraction (float): Desired fraction per subset (e.g., 0.05 for ~5%).

    Returns:
        dict[str, ray.data.Dataset]: ``{"val_1": ..., "val_2": ..., ...}``.
    """
    total_len = ds.count()
    n_splits = math.ceil(1 / fraction)
    target_size = math.ceil(total_len * fraction)
    split_indices = [min((i + 1) * target_size, total_len) for i in range(n_splits - 1)]
    splits = ds.split_at_indices(split_indices)
    return {f"val_{i+1}": subset.materialize() for i, subset in enumerate(splits)}


if __name__ == "__main__":
    """Ray Tune Workers Configuration

    We organize the workflow in several main steps:
    1. Load and preprocess the dataset (Ray Dataset API).
    2. Define the TorchTrainer per trial, including resource allocation.
    3. Set up hyperparameter search strategy and early stopping schedulers.
    4. Create and run a Ray Tune Tuner for automated hyperparameter optimization.
    """

    # Arguments and Logger
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", \
                        level=logging.DEBUG if args.debug else logging.INFO,
                        )
                        # filename=os.path.join(args.output_dir,'memory_usage.log'))

    logger.info("Hi!")
    set_seed(args.random_seed)   # set random seed for reproducibility

    # Save Arguments for reproducibility
    config_ = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__))
    args.output_dir = os.path.join(args.output_dir, args.search_schedule_mode, args.output_tag )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_file(config_,args.output_dir)

#    if args.run_on_local_machine:
#        args.storage_path = os.path.join(os.getcwd(),"output")
#        ray.init()
#    else:
    print("Already initialized?", ray.is_initialized())
    ip_head = os.getenv("ip_head")
    print("ip_head:", repr(ip_head))
    if ip_head:
        # Multi-node mode: connect to a head started by the SBATCH wrapper.
        ray.init(address=ip_head)
    else:
        # Single-node mode: start a NEW local Ray on the SLURM node. Force
        # address="local" so Ray does not auto-pick a stale RAY_ADDRESS env var
        # or a session_latest symlink left in $TMPDIR by a previous failed job.
        # Constrain CPU/GPU counts to what SLURM actually gave us; Ray's auto-
        # detect reads /proc/cpuinfo and would otherwise try to spawn workers
        # for *all* cores on the physical node, exceeding the cgroup and
        # hanging on worker startup.
        os.environ.pop("RAY_ADDRESS", None)
        ray_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or None
        ray_gpus_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        ray_gpus = (len([x for x in ray_gpus_env.split(",") if x.strip()])
                    if ray_gpus_env else None)
        # Cap Ray's object store so it doesn't claim ~25% of node RAM
        # unconditionally (200GB on a 1TB node), which pushed our cgroup over
        # its SLURM RAM allocation and triggered OOM-kills of trial workers.
        # 32GB is more than enough for the parquet shards + intermediate batches.
        ray_object_store_bytes = int(os.environ.get(
            "RAY_OBJECT_STORE_BYTES", str(32 * 1024 * 1024 * 1024)
        ))
        print(f"No ip_head env var; starting a fresh local Ray on this SLURM node "
              f"(num_cpus={ray_cpus}, num_gpus={ray_gpus}, "
              f"object_store={ray_object_store_bytes / 1e9:.0f} GB).")
        ray.init(
            address="local",
            num_cpus=ray_cpus,
            num_gpus=ray_gpus,
            object_store_memory=ray_object_store_bytes,
            include_dashboard=False,
            ignore_reinit_error=True,
            _temp_dir=os.environ.get("TMPDIR", "/tmp"),
        )

    logger.info("Ray Nodes info: %s", ray.nodes())
    logger.info("Ray Cluster Resources: %s", ray.cluster_resources())


    """STEP 1: Data Loading and Preprocessing

    Depending on config:
    - Dataset and collator can be loaded externally or inside the trainer.
    - Dataset format (.h5 or .parquet) is determined by args.data_mode.
    """

    args.path_to_data = DATA_PATH #os.path.join(DATA_PATH, args.dataset_name)

#    args.data_mode = "train_parquet"
    args.data_mode = "parquet"
    dataset_kwargs = make_dataset_kwargs(args)
    if not args.load_ds_in_trainer:
        try:
            ray_datasets, data_collators = get_datasets_and_collators(dataset_kwargs)
            
            dataset_kwargs["data_mode"] = "val_parquet"
            data_collators["val"] = dataset_kwargs
            logger.info("Successfully loaded Dataset.")
        except Exception as e:
            ray_datasets, data_collators = {}, None
            logger.info(f"Could not load the ray_datasets: {e}")
    else:
        # this is not the right way to do it here TODO: update this to allow mixture dat aloading (within and outside)
        ray_datasets = {}
        ray_datasets_, data_collators = get_datasets_and_collators(dataset_kwargs)
        train_ds = ray_datasets_["train"]
        val_ds = ray_datasets_["val"]

        # Split into equal-sized random shards. The trainer runs one shard per eval
        # (step-keyed) and reports a running mean of eval_wer_diff across the trial's
        # evals, so all shards are visited over training.
        val_subsets = partition_dataset(val_ds, fraction=args.eval_sample_fraction)

        # Merge into main dict
        ray_datasets = {"train": train_ds, **val_subsets}

#        ray_datasets["val_1"] = ray_datasets_["val"].random_sample(0.05).materialize()
#        ray_datasets["val_2"] = ray_datasets_["val"].random_sample(0.05).materialize()
#        ray_datasets["val_3"] = ray_datasets_["val"].random_sample(0.05).materialize()
        logger.info(f"Created {len(val_subsets)} validation subsets, "
            f"each about {args.eval_sample_fraction * 100:.1f}% of total.")
#        ray_datasets["val"] = None
#        data_collators["val"] = None
#        ray_datasets, data_collators = None, dataset_kwargs


    try:
        args.len_train_set = ray_datasets["train"].count()
        logger.info('len_train_set: %s', args.len_train_set)
        for name in list(val_subsets.keys())[:4]:
            logger.info('len_test_set[%s]: %s', name, ray_datasets[name].count())
    except Exception as e:
        logger.info(f"ray_datasets does not have key train or \n "
              f"was not able to .count() length: {e}")

    logger.info("Starting Finetuning for model %s", args.model_type)

    # get the relevant hyperparameter config and training kwargs (necessary for finetuning with ray)
    training_kwargs = make_training_kwargs(args)

    """STEP 2: Define Ray TorchTrainer

    Trainer wraps the training loop per Ray trial.
    - Uses HF Seq2SeqTrainer via train_whisper_model or train_whisper_peft_model.
    - ScalingConfig controls resource usage (CPU/GPU) and worker distribution.
    - Dataset either passed here or loaded inside training function (optional).
    """


    resources_per_trial={"CPU": args.cpus_per_trial, "GPU": args.gpus_per_trial}

    train_model = train_whisper_peft_model if args.peft else train_whisper_model

    trainer = TorchTrainer(
        partial(train_model,
                training_kwargs=training_kwargs,
                data_collators=data_collators,
                eval_sample_fraction=args.eval_sample_fraction,
                eval_names = list(val_subsets.keys())
                ),  # the training function to execute on each worker.
        scaling_config=ScalingConfig(num_workers=args.num_workers,
                                     use_gpu=args.use_gpu,
                                     resources_per_worker = resources_per_trial,
                                     # The trainer-driver actor is a thin
                                     # coordinator (it doesn't run the model);
                                     # don't let it reserve a CPU that the
                                     # worker bundles could use instead.
                                     trainer_resources={"CPU": 0},
                                     # PACK for single-node: SPREAD prefers cross-node
                                     # placement, which on a 1-node SLURM allocation
                                     # makes Ray's scheduler under-pack and only run
                                     # 2 trials at a time instead of max_concurrent_trials.
                                     placement_strategy="PACK"),
        # NCCL should be default, not need to set it
        # The configuration for setting up the PyTorch Distributed backend.
        #             If set to None, a default configuration will be used in which
        #             GPU training uses NCCL and CPU training uses Gloo.
        # torch_config=ray.train.torch.TorchConfig(
        # backend="NCCL"  # Use DDP for initialization
        # ),
        datasets = ray_datasets
    #None if args.load_ds_in_trainer else { "train": ray_datasets["train"],"eval": ray_datasets["validation"]}
    )

    tune_searcher, tune_scheduler = get_searcher_and_scheduler(args)
    logger.info(
        "Concurrency knobs:  args.max_concurrent_trials=%s  args.num_samples=%s  "
        "args.cpus_per_trial=%s  args.gpus_per_trial=%s",
        args.max_concurrent_trials, args.num_samples,
        args.cpus_per_trial, args.gpus_per_trial,
    )

    """STEP 3: Configure Ray Tune Tuner

    - Tuner handles full HPO (Hyperparameter Optimization) loop:
        * Samples from parameter space defined in get_hyperparameters().
        * Uses Ray Tune's searchers and schedulers to explore space efficiently.
        * Stores checkpoints and logs results to output directory.
    - Can optionally resume from a previous run if interrupted.

    References:
        [1] https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html
        [2] https://docs.ray.io/en/latest/tune/tutorials/tune-fault-tolerance.html#tune-fault-tolerance-ref
    """

    def _build_fresh_tuner():
        return Tuner(
            trainer,
            param_space={
                "train_loop_config": get_hyperparameters(args)
            },
            tune_config=tune.TuneConfig(
                metric=args.metric_to_optimize[0][0],
                mode=args.modes[0][0],
                num_samples=args.num_samples,
                search_alg=tune_searcher,
                scheduler= tune_scheduler,
                reuse_actors=args.reuse_actors,
                # Belt-and-braces: in some Ray versions TuneConfig has its own
                # implicit cap; setting it here in addition to the
                # ConcurrencyLimiter on the searcher makes the limit explicit.
                max_concurrent_trials=int(args.max_concurrent_trials),
            ),
            # run_config
            run_config=RunConfig(
                name=args.output_tag,     # folder name where ray workers results are saved and basis for tensorboard viz.
                 storage_path=args.storage_path,
                checkpoint_config=CheckpointConfig(
                    num_to_keep= args.num_to_keep,
                    checkpoint_score_attribute=args.metric_to_optimize[0][0],
                    checkpoint_score_order=args.modes[0][0],
                ),
                stop=(
                    DecorrelationStopper(
                        grace_period=args.grace_period,
                        loss_key=args.decorr_loss_key,
                        wer_key=args.decorr_wer_key,
                    ) if args.decorr_stopper else None
                ),
            ),
        )

    # Multi-slot resume: if resume_training=True but no Ray Tune experiment dir
    # exists yet (first launch of a run that will later resume across SLURM
    # slots), fall through to a fresh Tuner instead of crashing inside
    # Tuner.restore. Same config now works for slot 1 and slot N.
    experiment_dir = os.path.join(args.storage_path, args.output_tag)
    can_restore = False
    if args.resume_training and not getattr(args, "fresh_study", False):
        try:
            can_restore = Tuner.can_restore(experiment_dir)
        except Exception as e:
            logger.warning("Tuner.can_restore(%s) failed: %s. Treating as not restorable.",
                           experiment_dir, e)
            can_restore = False
    elif getattr(args, "fresh_study", False) and args.resume_training:
        logger.info("--fresh_study: skipping Ray Tune experiment restore from %s.", experiment_dir)

    if args.resume_training and can_restore:
        logger.info("Resuming Ray Tune experiment from %s", experiment_dir)
        tuner = Tuner.restore(experiment_dir,
                              trainable=trainer,
                              resume_unfinished=True,
                              resume_errored=True)
    else:
        if args.resume_training:
            logger.warning(
                "resume_training=True but no restorable experiment found at %s. "
                "Starting a fresh Tuner; subsequent re-launches will resume it.",
                experiment_dir,
            )
        tuner = _build_fresh_tuner()
        
    # Sort and retrieve best result based on configured metric (e.g. eval_wer).
    tune_results = tuner.fit()
    # args.metric_to_optimize is a list-of-lists from list_of_strings + append;
    # other call sites flatten it with [0][0], do the same here.
    tune_results.get_dataframe().sort_values(args.metric_to_optimize[0][0])
    best_result = tune_results.get_best_result()

    logger.info('Best Result %s', best_result)

    # save best result into folder
    best_result_list = {}
    best_result_list["metrics"] = best_result.metrics
    best_result_list["path"] = best_result.path
    best_result_list["checkpoint"] = best_result.checkpoint
    np.save(os.path.join(args.output_dir,'best_result.npy'), best_result_list)

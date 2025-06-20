"""Main Hyper-Parameter Finetuning script for Whisper ASR using Ray Tune.

For training whisper, we use the Huggingface (HF) transformers package. We base the training on the following tutorial for
finetuning using HF-Whisper: https://huggingface.co/blog/fine-tune-whisper

For allowing Ray Tune Hyperparmeter finetunging with HF-Whisper, we follow the following tutorial:
https://docs.ray.io/en/latest/train/getting-started-transformers.html

A high-level overview of the training procsess:
    1.  We prepare the ray tune finetuning (Tuner) object. Defining the hyperparameters to tune, which scheduler to use
        how many resources to reserve for each training, how many trials etc.
    2.  train_model will be called by each trial. This is the main training function based on the HF Seq2SeqTrainer,
        see trainers.py

For a description of what each function is doing, we refer to the docstrings of the very function.

*Note*: For using this finetuning script for a different set-up (different model, different task the ASR), the
get_models, train_model, and get_hyperparameters needs to be adjusted.
"""
# General
import os
import pdb
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
from ray.tune import Tuner, RunConfig

from trainers.trainers import make_seq2seq_training_kwargs as make_training_kwargs
from trainers.trainers import train_whisper_model, train_whisper_peft_model
from searchers_and_schedulers.ray_searchers_and_schedulers import get_searcher_and_scheduler
from searchers_and_schedulers.ray_searchers_and_schedulers import get_whisper_hyperparameters as get_hyperparameters

# Datasets
from data_and_collator.datasets_and_collators import get_datasets_and_collators, make_dataset_kwargs

# Logging control
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
os.environ["RAY_VERBOSITY"] = "0"
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
ray.data.context.DataContext.get_current().enable_operator_progress_bars = False
ray.data.context.DataContext.get_current().enable_progress_bars = False

logger = logging.getLogger(__name__)

# options for different ray searchers and scheduler (see the get_searcher_and_scheduler function for details)
TUNE_CHOICES = ['small_small', 'large_small_OPTUNA', 'large_large']
DATA_MODES = ['h5', 'parquet']
def parse_args():
    """ Parses command line arguments for the training.

    In particular:
            model_type...Whisper model type choices: https://huggingface.co/models?search=openai/whisper
            Ray tune general options: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
            Ray Searcher options: https://docs.ray.io/en/latest/tune/api/schedulers.html#resourcechangingscheduler
    """
    parser = configargparse.ArgumentParser()

    # Training settings for Seq2SeqTrainingArguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="increase by 2x for every 2x decrease in batch size")
    # The only reason to use gradient accumulation steps is when your whole batch size does not fit on one GPU, so you pay a price in terms of speed to overcome a memory issue.
    parser.add_argument("--output_tag", type=str,
                        default="whisper-tiny-de",
                        help="Base directory where model is save.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max Number of gradient steps")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Max Number of gradient steps")
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
    parser.add_argument("--max_warmup_steps", type=int, default=10,
                        help="For Hyperparam search. What is the max value for warmup?")
    parser.add_argument("--len_train_set", type=int, default=10,
                        help="Helper variable to define max_t which is required for schedulers.")
    parser.add_argument("--max_concurrent_trials", type=int, default=1,
                        help="Maximum number of trials to run concurrently.")
    parser.add_argument("--prefetch_batches", type=int, default=1,
                        help="How many batches to prefetch data? Keep in mind: is using VRAM.")

    parser.add_argument("--load_ds_in_trainer", action="store_true", default=False, help="Wheter to load the ds within the "
                                                                                         "trainer or outside. ")

    # tune options: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
    parser.add_argument("--num_samples", type=int, default=5, help="Number of times to sample from the hyperparameter space.")
    parser.add_argument("--num_to_keep", type=int, default=1, help="number of checkpoints to keep on disk")
    # parser.add_argument("--max_t", type=int, default=10, help="Max number of steps for finding the best hyperparameters")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of trials that can run at the same time")
    parser.add_argument("--cpus_per_trial", type=int, default=1, help="Number of CPUs per Ray actor")
    parser.add_argument("--gpus_per_trial", type=float, default=0, help="Number of GPUs per Ray actor")
    parser.add_argument("--use_gpu", action="store_true", help="If using GPU for the finetuning")
    parser.add_argument("--fp16", action="store_true", default=False, help="Training with floating point 16 ")
    parser.add_argument("--reuse_actors", action="store_true", help="Reusing Ray Actors should accelarate training")
    parser.add_argument("--metric_to_optimize", type=list_of_strings, action ="append", help="Metric which is used for deciding what a good trial is.")
    parser.add_argument("--modes", type=list_of_strings, action ="append", help="Modes of metrics (mix or max). Position 1 refers to metric 1 etc.")
    parser.add_argument("--eval_sample_size", type=int, default=10, help="Divides the evaluation set in eval_sample_size shards,"
                                                                         "and evaluates only a sub-sample after each eval step")
    # For Scheduler and Search algorithm specifically: https://docs.ray.io/en/latest/tune/api/schedulers.html#resourcechangingscheduler
    parser.add_argument("--search_schedule_mode", type=str, default="large_small_BOHB", choices=TUNE_CHOICES, help="Which Searcher Algorithm and Scheduler combination. See 'get_searcher_and_scheduler' function for details.")
    parser.add_argument("--reduction_factor", type=int, default=2, help="Factor by which trials are reduced after grace_period of time intervals")
    parser.add_argument("--grace_period", type=int, default=1, help="With grace_period=n you can force ASHA to train each trial at least for n time interval")
    # For Population Based Training (PBT): https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    parser.add_argument("--perturbation_interval", type=int, default=10, help="Models will be considered for perturbation at this interval of time_attr.")
    parser.add_argument("--burn_in_period", type=int, default=1, help="Grace Period for PBT")
    # which hyperparameter to search for for
    parser.add_argument('--hyperparameters', type=list_of_strings, action ="append", help="List of Hyperparameter to tune")

    # Other settings
    parser.add_argument("--run_on_local_machine", action="store_true", help="Store true if training is on local machine.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base directory where outputs are saved.")
    parser.add_argument("--storage_path", type=str, default="./output/scratch", help="Where to store ray tune results. ")
    parser.add_argument("--resume_training", action="store_true", help="Whether or not to resume training.")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--path_to_data", type=str, default="../data/datasets/fzh-wde0459_03_03",
                        help="Path to audio batch-prepared audio files if in debug mode. Otherwise: all data in datasets are loaded")
    parser.add_argument("--dataset_name", type=str, default="eg_dataset_subset_1000.h5",
                        help="Name of dataset")
    parser.add_argument("--random_seed", type=int, default=1337, help="Random Seed for reproducibility")
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Ray Tune Workers Configuration
    
    We break the configuration down in several steps
    1. We load and pre-process the data
    2. We define the TorchTrainer for each trial, including the resources (GPUs, CPUs, Workers) per trial
       https://docs.ray.io/en/latest/_modules/ray/train/torch/torch_trainer.html#TorchTrainer
    3. We define the searchers and schedulers (get_searcher_and_scheduler) used to find the optimizal paramters
    4. We set-up a ray Tuner Object defining the the hyperparameters, and other important Configs 
    """

    """STEP 1: Data laoding and pre-processing"""

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

    # if args.run_on_local_machine:
    #     args.storage_path = os.path.join(os.getcwd(),"output")
    #     ray.init()
    # else:
    #     ray.init("auto")

    # logger.info("Ray Nodes info: %s", ray.nodes())
    # logger.info("Ray Cluster Resources: %s", ray.cluster_resources())


    """STEP 1: Load the data
    
        We need to define a data_mode config with defining which type of data and collator is used.
    """

    path_to_data = os.path.join(args.path_to_data, args.dataset_name)

    # data_mode = {
    #     "train": {"type": "h5",
    #               "collator": "streaming",
    #               "path_to_ds": path_to_data},
    #     "val": {"type": "h5",
    #             "collator": "streaming",
    #             "path_to_ds": path_to_data}
        # "val": {"type": "parquet",
        #         "collator": "parquet",
        #         "path_to_ds": path_to_data}
        # "val": {"type": "h5",
        #                "collator": "streaming",
        #                "path_to_ds": path_to_data}
    # }

    dataset_kwargs = make_dataset_kwargs(args)
    if not args.load_ds_in_trainer:
        try:
            ray_datasets, data_collators = get_datasets_and_collators(dataset_kwargs)
            logger.info("Successfully loaded Dataset.")
        except Exception as e:
            ray_datasets, data_collators = {}, None
            logger.info(f"Could not load the ray_datasets: {e}")
    else:
        ray_datasets, data_collators = None, dataset_kwargs


    try:
        args.len_train_set = ray_datasets["train"].count()
        logger.info('len_train_set: %s', args.len_train_set)
    except Exception as e:
        logger.info(f"ray_datasets does not have key train or \n "
              f"was not able to .count() length: {e}")

    logger.info("Starting Finetuning for model %s", args.model_type)

    # get the relevant hyperparameter config and training kwargs (necessary for finetuning with ray)
    training_kwargs = make_training_kwargs(args)

    """STEP 2: Define the Ray Trainer
   
    Args: 
        train_model (function): the training function to execute on each trial.
                     The partial wrapper allows for additional arguments (config is required).  
        scaling_config (ScalingConfig): resources_per_worker...defines CPU and GPU requirements used by each worker.
                     num_workers...number of workers used for each trial
                     placement_strategy...how job is distributed across different nodes
       datasets (dict): Dataset dictionary. Train and eval with ray_dataset objects is required. 
    
    Reference: https://docs.ray.io/en/latest/_modules/ray/train/torch/torch_trainer.html#TorchTrainer
    """

    resources_per_trial={"CPU": args.cpus_per_trial, "GPU": args.gpus_per_trial}

    train_model = train_whisper_peft_model if args.peft else train_whisper_model

    trainer = TorchTrainer(
        partial(train_model,
                training_kwargs=training_kwargs,
                data_collators=data_collators,
                eval_sample_size=args.eval_sample_size),  # the training function to execute on each worker.
        scaling_config=ScalingConfig(num_workers=args.num_workers,
                                     use_gpu=args.use_gpu,
                                     resources_per_worker = resources_per_trial,
                                     placement_strategy="SPREAD"),
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

    """Step 4: Set-up a ray Tuner [1]

    The Tuner Object takes the trainer, defines the param_space, configures the searcher and scheduler, while observing the
    checkpointing config (more details below). 

    If resume_training = True, training is resumed from the previous interrupted training. Important: all the settings needs
    to be the same as the original run. We resume unfinished and errored trials. Terminated trials will not be resumed.
    Ref: [2] 

    More Details:
        param_space (dict)...defines the hyper-parameter space we search for optimal parameters. Requires a prior distribut.
                             over the hyperparameters. Instances of parameters will be passed to Seq2SeqTrainingArguments.
                             In principle, every optimial parameter there can be part of the param_space.           
        tune_config (Tune.Config) ... Config for deciding which metric to use for deifning a good trial (e.g. eval_loss or 
                                      eval_wer,  number of total trials/hyperparameter set-ups (num_samples), searcher, 
                                      scheduler. Reuse_actor optimizes resource usage so that all GPUs are always used.
        run_config (RunConfig): Runtime configuration that is specific to individual trials.
                    num_to_keep...how many checkpoints to keep (more requires more memory)
                    checkpoint_score_attribute...which metric to use for performance
                    checkpoint_score_order...min means the lower the checkpoint_score_attribute, the better

    References:
        [1] https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html
        [2] https://docs.ray.io/en/latest/tune/tutorials/tune-fault-tolerance.html#tune-fault-tolerance-ref 
    """


    if args.resume_training:
        tuner = Tuner.restore(os.path.join(args.storage_path, args.output_tag),
                              trainable=trainer,
                              resume_unfinished = True,
                              resume_errored = True)

    else:
        # pdb.set_trace()
        tuner = Tuner(
            trainer,
            param_space={
                "train_loop_config": get_hyperparameters(args)
            },
            tune_config=tune.TuneConfig(
                metric=args.metric_to_optimize[0][0],
                mode=args.modes[0][0],
                num_samples=args.num_samples, # num_samples (int) – Number of times to sample from the hyperparameter space..
                search_alg=tune_searcher,
                scheduler= tune_scheduler,
                reuse_actors=args.reuse_actors,
            ),
            # run_config
            run_config=RunConfig(
                name=args.output_tag,     # folder name where ray workers results are saved and basis for tensorboard viz.
                # storage_path=args.storage_path,
                checkpoint_config=CheckpointConfig(
                    num_to_keep= args.num_to_keep,
                    checkpoint_score_attribute=args.metric_to_optimize[0][0],
                    checkpoint_score_order=args.modes[0][0],
                ),
            ),
        )

    tune_results = tuner.fit()
    tune_results.get_dataframe().sort_values(args.metric_to_optimize)
    best_result = tune_results.get_best_result()

    logger.info('Best Result %s', best_result)

    # save best result into folder
    best_result_list = {}
    best_result_list["metrics"] = best_result.metrics
    best_result_list["path"] = best_result.path
    best_result_list["checkpoint"] = best_result.checkpoint
    np.save(os.path.join(args.output_dir,'best_result.npy'), best_result_list)

"""Training a specific Model.

For training whisper, we use the Huggingface (HF) transformers package. We base the training on the following tutorial for
finetuning using HF-Whisper: https://huggingface.co/blog/fine-tune-whisper

A high-level overview of the training procsess:
    1.  We define the model.
    2.  We prepare the dataloaders
    3.  We start the training of a specific model.

"""
import os

import pdb
from functools import partial
import pprint
import numpy as np
import json

from utils import (list_of_strings, create_ray_indexloader,
                   save_file, steps_per_epoch)

from utils import SimpleStreamingCollator
import h5py

# laod models
from transformers import set_seed

# For code organization and reporting
import configargparse
import logging

# For hyperparameter optimization
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
import ray.train
from ray.tune import Tuner
from models import get_whisper_models as get_models
from trainers import train_whisper_model as train_model
from trainers import train_single_whisper_model
from trainers import make_seq2seq_training_kwargs as make_training_kwargs
from utils_new import SimpleStreamingCollator

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
os.environ["RAY_VERBOSITY"] = "0"
ray.data.context.DataContext.get_current().enable_operator_progress_bars = False
ray.data.context.DataContext.get_current().enable_progress_bars = False

logger = logging.getLogger(__name__)

# options for different ray searchers and scheduler (see the get_searcher_and_scheduler function for details)
TUNE_CHOICES = ['small_small', 'large_small_BOHB', 'large_small_OPTUNA', 'large_large']


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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="increase by 2x for every 2x decrease in batch size")
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
    parser.add_argument("--dataloader_num_workers", type=int, default=1,
                        help="How many CPUs to allocate for dataloaders")
    parser.add_argument("--logging_steps", type=int, default=25, help="After how many steps to do some logging")

    # model settings
    parser.add_argument("--model_type", type=str, default="openai/whisper-tiny", help="Model to optimize")
    parser.add_argument("--target_language", type=str, default="german", help="Target Language")
    parser.add_argument("--return_timestamps", action="store_true", help="Return Timestemps mode for model")
    parser.add_argument("--peft", action="store_true", help="Whether or not to do Parameter Efficient Training")
    # https://github.com/Vaibhavs10/fast-whisper-finetuning?tab=readme-ov-file#evaluation-and-inference

    # Dataset settings
    parser.add_argument("--single_file", action="store_true", help="If data is in a single.h5 format. If False, assume "
                                                                   "folder structure of: training and talidation")

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

    # tune options: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.TuneConfig.html
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of times to sample from the hyperparameter space.")
    parser.add_argument("--num_to_keep", type=int, default=1, help="number of checkpoints to keep on disk")
    parser.add_argument("--max_t", type=int, default=10,
                        help="Max number of steps for finding the best hyperparameters")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of trials that can run at the same time")
    parser.add_argument("--cpus_per_trial", type=int, default=1, help="Number of CPUs per Ray actor")
    parser.add_argument("--gpus_per_trial", type=float, default=0, help="Number of GPUs per Ray actor")
    parser.add_argument("--use_gpu", action="store_true", help="If using GPU for the finetuning")
    parser.add_argument("--fp16", action="store_true", default=False, help="Training with floating point 16 ")
    parser.add_argument("--reuse_actors", action="store_true", help="Reusing Ray Actors should accelarate training")
    parser.add_argument("--metric_to_optimize", type=str, default="eval_loss",
                        help="Metric which is used for deciding what a good trial is.")

    # For Scheduler and Search algorithm specifically: https://docs.ray.io/en/latest/tune/api/schedulers.html#resourcechangingscheduler
    # parser.add_argument("--search_schedule_mode", type=str, default="large_small_BOHB", choices=TUNE_CHOICES,
    #                     help="Which Searcher Algorithm and Scheduler combination. See 'get_searcher_and_scheduler' function for details.")
    parser.add_argument("--reduction_factor", type=int, default=2,
                        help="Factor by which trials are reduced after grace_period of time intervals")
    parser.add_argument("--grace_period", type=int, default=1,
                        help="With grace_period=n you can force ASHA to train each trial at least for n time interval")
    # For Population Based Training (PBT): https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    parser.add_argument("--perturbation_interval", type=int, default=10,
                        help="Models will be considered for perturbation at this interval of time_attr.")
    parser.add_argument("--burn_in_period", type=int, default=1, help="Grace Period for PBT")
    # which hyperparameter to search for for
    parser.add_argument('--hyperparameters', type=list_of_strings, action="append",
                        help="List of Hyperparameter to tune")

    # Other settings
    parser.add_argument("--run_on_local_machine", action="store_true",
                        help="Store true if training is on local machine.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base directory where outputs are saved.")
    parser.add_argument("--storage_path", type=str, default="/scratch/USER/ray_results",
                        help="Where to store ray tune results. ")
    parser.add_argument("--resume_training", action="store_true", help="Whether or not to resume training.")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--path_to_data", type=str, default=None,
                        help="Path to audio batch-prepared audio files if in debug mode. Otherwise: all data in datasets are loaded")
    parser.add_argument("--dataset_name", type=str, default="eg_dataset_subset_1000.h5",
                        help="Name of dataset")
    parser.add_argument("--random_seed", type=int, default=1337, help="Random Seed for reproducibility")
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")

    args = parser.parse_args()
    return args


def collate_fn(batch):
    input_features_list = []
    labels_list = []

    input_features_batch = torch.stack([torch.from_numpy(x) for x in batch["input_features"]])
    labels_batch = torch.stack([torch.from_numpy(x) for x in batch["labels"]])

    return {
        "input_features": input_features_batch, #input_features_batch,
        "labels": labels_batch# labels_batch
    }


if __name__ == "__main__":

    """STEP 1: GETTING THE MODEL"""

    # Arguments and Logger
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", \
                        level=logging.DEBUG if args.debug else logging.INFO, \
                        filename=os.path.join(args.output_dir, 'memory_usage.log'))

    logger.info("Hi!")

    # set random seed for reproducibility
    set_seed(args.random_seed)

    # get models
    if args.run_on_local_machine:
        from models import get_whisper_models_local
        model, feature_extractor, tokenizer, processor = get_whisper_models_local(args.model_type, args.target_language,
                                                                    return_timestamps=args.return_timestamps,
                                                           load_in_8bit=args.peft)
    else:
        model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language,
                                                                return_timestamps=args.return_timestamps,
                                                                load_in_8bit=args.peft)
    """STEP 2: LO"""
    # Load the state dictionary from specific trial
    # Option B: Or specify a specific trial ID if you know which one you want
    specific_trial_id = "TorchTrainer_a9924cfa_69_learning_rate=0.0000,lr_scheduler_type=linear,per_device_train_batch_size=8,warmup_steps=24,weight_decay=_2025-04-28_20-55-27"  # Replace with your actual trial ID
    path_to_experiments= os.path.join("/scratch/usr/bemchrvt/ray_results/", "v3_large_feb")
    path_to_experiment = os.path.join(path_to_experiments, specific_trial_id)

    if os.path.exists(path_to_experiment):
        # hyper_parameters = {}

        with open(os.path.join(path_to_experiment, 'params.json'),'r') as f:
            parameter = json.load(f)
            hyper_parameters = {
            "learning_rate": parameter["train_loop_config"]["learning_rate"],
            "lr_scheduler_type": parameter["train_loop_config"]["lr_scheduler_type"],
            "warmup_steps": parameter["train_loop_config"]["warmup_steps"],
            "per_device_train_batch_size": parameter["train_loop_config"]["per_device_train_batch_size"],
            "weight_decay": parameter["train_loop_config"]["weight_decay"]
            }
            # learning_rate = parameter["train_loop_config"]["learning_rate"]
            # lr_scheduler_type = parameter["train_loop_config"]["lr_scheduler_type"]
            # warmup_steps = parameter["train_loop_config"]["warmup_steps"]
            # per_device_train_batch_size = parameter["train_loop_config"]["per_device_train_batch_size"]
            # weight_decay = parameter["train_loop_config"]["weight_decay"]

        # from transformers.models.whisper.convert_openai_to_hf import make_linear_from_emb
        # model_ckpt_path = os.path.join(path_to_experiment, "checkpoint_000000","checkpoint")
        # model = WhisperForConditionalGeneration.from_pretrained(model_ckpt_path)
        # model.generation_config.language = 'de'
        # model.generation_config.task = "transcribe"
        # model.generation_config.forced_decoder_ids = None
        # model.generation_config.return_timestamps = return_timestamps
        # # state_dict = safetensors.torch.load_file(os.path.join(model_ckpt_path, 'model.safetensors'))
        # # # Fix missing proj_out weights: https://github.com/openai/whisper/discussions/2302
        # # model.load_state_dict(state_dict, strict=False)
        # model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
        # logger.info('Whisper model from checkpoint %s loaded.', args.model_ckpt_path)


    else:
       logger.info(f"{path_to_experiment} does not exists or model could not be loaded.")


    """STEP 2: Load Dataset"""

    # Count trainable parameters
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # logger.info("Trainable Original Parameters: %s", trainable_params)

    # Initial memory usage
    #    log_memory_usage("before_ray_init")
    if args.run_on_local_machine:
        args.storage_path = os.path.join(os.getcwd(), "output")
        ray.init()
    else:
        ray.init("auto")

    #    log_memory_usage("after_ray_init")

    # Could help to connect to head note if connection fails frequently
    #     ip_head = os.getenv("ip_head")
    #     if not ip_head:
    #         raise RuntimeError("Head node address not found in environment variables.")
    #     ray.init(address=ip_head)
    #     username = os.getenv("USER")

    logger.info("Ray Nodes info: %s", ray.nodes())
    logger.info("Ray Cluster Resources: %s", ray.cluster_resources())

    path_to_data = os.path.join("/scratch/usr/", os.getenv(
        'USER') + "/data/eg_dataset_complete_v3_sharded") if args.path_to_data is None else args.path_to_data

    dataset_name = args.dataset_name
    train_h5_path = os.path.join(path_to_data, f"{dataset_name}_train.h5")
    val_h5_path = os.path.join(path_to_data, f"{dataset_name}_val.h5")

    train_loader, val_loader = create_ray_indexloader(train_h5_path), create_ray_indexloader(val_h5_path)

    # train_h5_path = os.path.join(path_to_data, "train_parquet")
    #
    # train_ds = ray.data.read_parquet(
    #                 train_h5_path,
    #                 )

    dataset_size = train_loader.count()
    len_train_set = dataset_size
    args.len_train_set = len_train_set
    logger.info('len_train_set: %s', len_train_set)

    # ray_datasets = {
    #     "train": train_loader,
    #     "validation": val_loader,
    # }

    # data_collators = {
    # "training": collate_fn,
    # "validation": SimpleStreamingCollator(val_h5_path, feature_extractor, tokenizer, num_workers=args.cpus_per_trial)
    # }

    logger.info('Ray datasets and collators prepared.')

    # Save Arguments for reproducibility
    config_ = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__))
    args.output_dir = os.path.join(args.output_dir, args.output_tag)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_file(config_, args.output_dir)

    # get the relevant hyperparameter config and training kwargs (necessary for finetuning with ray)
    training_kwargs = make_training_kwargs(args)
    hyper_parameters["training_kwargs"] = training_kwargs
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
    # args.cpus_per_trial
    resources_per_trial = {"CPU": args.cpus_per_trial, "GPU": args.gpus_per_trial}
    trainer = TorchTrainer(
        train_loop_per_worker = train_single_whisper_model,  # the training function to execute on each worker.
        train_loop_config=hyper_parameters,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu,
                                     resources_per_worker=resources_per_trial, placement_strategy="SPREAD"),
        datasets={
            "train": ray_datasets["train"],
            "eval": ray_datasets["validation"],
            # "test": ray_datasets["test"], # we don't need this, saves space
        },
        # run_config
        run_config=RunConfig(
            name=args.output_tag,  # folder name where ray workers results are saved and basis for tensorboard viz.
            # stop={"training_iteration": 100},  # after how many iterations to stop
            storage_path=args.storage_path,
            checkpoint_config=CheckpointConfig(
                num_to_keep=args.num_to_keep,
            ),
        ),
    )

    if args.resume_training:
        # args.reuse_actors = False # otherwise somehow buggy when resuming training
        tuner = trainer.restore(os.path.join(args.storage_path, args.output_tag), trainable=trainer)

    trainer.fit()
    # else:
    #     tuner = Tuner(
    #         trainer,
    #         param_space={
    #             "train_loop_config": get_hyperparameters(args)
    #         },
    #         tune_config=tune.TuneConfig(
    #             max_concurrent_trials=args.max_concurrent_trials,
    #             metric=args.metric_to_optimize,
    #             mode="min",
    #             num_samples=args.num_samples,
    #             # num_samples (int) – Number of times to sample from the hyperparameter space..
    #             search_alg=tune_searcher,
    #             scheduler=tune_scheduler,
    #             reuse_actors=args.reuse_actors,
    #         ),
    #         # run_config
    #         run_config=RunConfig(
    #             name=args.output_tag,  # folder name where ray workers results are saved and basis for tensorboard viz.
    #             # stop={"training_iteration": 100},  # after how many iterations to stop
    #             storage_path=args.storage_path,
    #             checkpoint_config=CheckpointConfig(
    #                 num_to_keep=args.num_to_keep,
    #                 checkpoint_score_attribute=args.metric_to_optimize,
    #                 checkpoint_score_order="min",
    #             ),
    #         ),
    #     )
    #
    # tune_results = tuner.fit()
    # tune_results.get_dataframe().sort_values(args.metric_to_optimize)
    # best_result = tune_results.get_best_result()
    #
    # logger.info('Best Result %s', best_result)
    #
    # # save best result into folder
    # best_result_list = {}
    # best_result_list["metrics"] = best_result.metrics
    # best_result_list["path"] = best_result.path
    # best_result_list["checkpoint"] = best_result.checkpoint
    # np.save(os.path.join(args.output_dir, 'best_result.npy'), best_result_list)

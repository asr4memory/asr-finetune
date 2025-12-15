import os
import json
import pdb
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
import socket
import psutil
import ray
from functools import partial
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.tune import Tuner
from models import get_whisper_models as get_models
import logging
from utils import create_ray_indexloader, SimpleStreamingCollator, MultiStreamingCollator
logger = logging.getLogger(__name__)

os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
os.environ["RAY_VERBOSITY"] = "0"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
ray.data.context.DataContext.get_current().enable_operator_progress_bars = False
ray.data.context.DataContext.get_current().enable_progress_bars = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", type=str, default="/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_complete_v3_sharded/train")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--model_type", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu_list", nargs='+', type=int, default=[2, 4, 8, 16, 24])
    parser.add_argument("--prefetch_list", nargs='+', type=int, default=[1, 4, 8])
    parser.add_argument("--prefetch_item", type=int, default=1)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--gpus_per_trial", type=float, default=0)
    parser.add_argument("--output_tag", type=str, default="dataloader_benchmark")
    parser.add_argument("--storage_path", type=str, default="./ray_results")
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--run_on_local_machine", action="store_true")
    parser.add_argument("--target_language", type=str, default="de")
    return parser.parse_args()


def monitor_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB


def run_trial(config, data_collators=None):
    # Get dataset shards from Ray Train
    train_ds = ray.train.get_dataset_shard("train")

    # Create iterable dataset with specified prefetch and batch size
    train_ds_iterable = train_ds.iter_torch_batches(
        prefetch_batches=config["prefetch_batches"],
        batch_size=config["batch_size"],
        collate_fn=data_collators["training"])

    # Record start time and memory
    start = time.time()
    mem_start = monitor_memory()
    batch_count = 0

    # Process the required number of batches
    max_batches = 100  # Set a reasonable limit to avoid infinite loops
    for batch in train_ds_iterable:
        # Process the batch (in real scenario, this would involve model inference)
        # Here we're just counting batches for benchmarking
        batch_count += 1
        if batch_count >= max_batches:
            break

    # Calculate metrics
    duration = time.time() - start
    mem_end = monitor_memory()

    samples_per_sec = (batch_count * config["batch_size"]) / duration
    peak_mem = max(mem_start, mem_end)
    # pdb.set_trace()
    # Create detailed metrics
    metrics = {
        "samples_per_sec": round(samples_per_sec, 2),
    }
    logger.info("Samples per sec: %s", samples_per_sec )
    # Report metrics to Ray Tune
    ray.train.report(metrics)

    return metrics

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", \
                        level=logging.INFO,
                        )

    logger.info("Hi! Lets Benchmark our dataloaders!")

    # Initialize Ray
    if args.run_on_local_machine:
        args.storage_path = os.path.join(os.getcwd(), "output")
        ray.init(local_mode=True)
    else:
        # ray.init(local_mode=True)
        ray.init("auto")


    logger.info("Ray Nodes info: %s", ray.nodes())
    logger.info("Ray Cluster Resources: %s", ray.cluster_resources())

    # get models
    if args.run_on_local_machine:
        from models import get_whisper_models_local
        model, feature_extractor, tokenizer, processor = get_whisper_models_local(args.model_type, args.target_language,
                                                                    return_timestamps=False,
                                                           load_in_8bit=False)
    else:
        model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language,
                                                                return_timestamps=False,
                                                                load_in_8bit=False)




    path_to_data = os.path.join("/scratch/usr/", os.getenv('USER') + "/data") if args.path_to_data is None else args.path_to_data

    h5_path = os.path.join(path_to_data, args.dataset_name + ".h5")  # "eg_dataset_complete_5sec.h5")
    train_h5_path =os.path.join(path_to_data,  args.dataset_name + "_train.h5")
    val_h5_path =os.path.join(path_to_data, args.dataset_name + "_train.h5")
    # Hack: Ray Tune requires a ray dataset object. However, converting log-mel spectogram formatare not supported
    # So we create a index ray dataset and the actual data-fetching is Wrapped in the
    train_loader, val_loader = create_ray_indexloader(train_h5_path), create_ray_indexloader(val_h5_path)
    dataset_size = train_loader.count()
    logger.info("Dataset size: %s", dataset_size)


    # Set up Ray datasets
    ray_datasets = {
        "train": train_loader,
        "validation": val_loader,
    }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "benchmark_results.json")

    # Store all results here
    all_results = []

    # Run benchmarks for different CPU and prefetch combinations
    for cpu in args.cpu_list:
        for prefetch in args.prefetch_list:
            logging.info(f"Running benchmark with {cpu} CPUs and prefetch={prefetch}")

            if args.simple:
                data_collators = {
                    "training": SimpleStreamingCollator(train_h5_path, feature_extractor, tokenizer,
                                                        num_workers=cpu),
                    "validation": SimpleStreamingCollator(val_h5_path, feature_extractor, tokenizer,
                                                          num_workers=cpu)
                    }
            else:
                # Create the parallel collator with 4 reader processes
                data_collators = {
                    "training": MultiStreamingCollator(train_h5_path, processor, feature_extractor, tokenizer,
                                                       num_workers=cpu),
                    "validation": MultiStreamingCollator(val_h5_path, processor, feature_extractor, tokenizer,
                                                         num_workers=cpu)
                    }

            # Get dataset shards from Ray Train
            train_ds = train_loader

            # Create iterable dataset with specified prefetch and batch size
            train_ds_iterable = train_ds.iter_torch_batches(
                prefetch_batches=prefetch,
                batch_size=args.batch_size,
                collate_fn=data_collators["training"])

            from ray_custom_pytorch import WhisperHDF5TorchDataset

            torch_ds = WhisperHDF5TorchDataset(
                                    hdf5_path=train_h5_path,
                                    feature_extractor=feature_extractor,
                                    tokenizer=tokenizer,
                                 )
            from torch.utils.data import DataLoader

            import torch
            from torch.nn.utils.rnn import pad_sequence
            from typing import Dict, List, Optional, Union, Tuple

            def whisper_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
                # Get max lengths for dynamic padding
                max_feature_length = max([item["input_features"].shape[1] for item in batch])

                # Prepare lists for stacking
                input_features_list = []
                labels_list = []

                # Process each item in the batch
                for item in batch:
                    # Handle input features padding if necessary
                    features = item["input_features"]
                    if features.shape[1] < max_feature_length:
                        # Pad features
                        padding = torch.zeros(features.shape[0], max_feature_length - features.shape[1],
                                              dtype=features.dtype, device=features.device)
                        features = torch.cat([features, padding], dim=1)
                    input_features_list.append(features)

                    # Add labels directly (already padded)
                    labels_list.append(item["labels"])

                # Stack tensors
                input_features_batch = torch.stack(input_features_list, dim=0)
                labels_batch = torch.stack(labels_list, dim=0)

                return {
                    "input_features": input_features_batch,
                    "labels": labels_batch
                }

            train_ds_iterable = DataLoader(torch_ds,
                                           prefetch_factor=prefetch,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=cpu,
                                           pin_memory=False)
                                           # collate_fn=whisper_collate_fn)
                                           # collate_fn=whisper_collate_fn)


            # Record start time and memory
            start = time.time()
            # mem_start = monitor_memory()
            batch_count = 0

            # Process the required number of batches
            max_batches = 100  # Set a reasonable limit to avoid infinite loops
            for batch in train_ds_iterable:
                # Process the batch (in real scenario, this would involve model inference)
                # Here we're just counting batches for benchmarking
                batch_count += 1
                if batch_count >= max_batches:
                    break

            # Calculate metrics
            duration = time.time() - start
            # mem_end = monitor_memory()
            samples_per_sec = (batch_count * args.batch_size) / duration

            logger.info("CPUs: %s",cpu)
            logger.info('samples per sec : %s',samples_per_sec)
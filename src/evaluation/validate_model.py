"""
validate_model_minimal_patched.py

Minimal patch validator script:
- DOES NOT touch your original trainers.py / Seq2SeqTrainerEvalSampling
- DOES NOT touch collate_parquet
- Keeps args/model loading/dataset loading logic as-is
- Fixes standalone evaluation by:
  (1) wrapping Ray iter_torch_batches iterator into a torch IterableDataset
  (2) using a local identity collator that unwraps HF's list-of-one
  (3) copying training_kwargs inside the loop (so deletions don't poison later shards)
- Adds nice printing + saves eval_loss + eval_wer per shard at the end
"""

# General
import os
import sys
import pprint

# Make src/ importable so sibling packages resolve whether this file is run via
# ``python -m evaluation.validate_model`` or directly as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils import list_of_strings, save_file
from transformers import set_seed

# For code organization and reporting
import configargparse
import logging

# For hyperparameter optimization and ray stuff
import ray.train
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, CheckpointConfig
from ray.tune import Tuner, RunConfig

from trainers.trainers import make_seq2seq_training_kwargs as make_training_kwargs
from trainers.trainers import train_whisper_model, train_whisper_peft_model
from searchers_and_schedulers.ray_searchers_and_schedulers import get_searcher_and_scheduler
from searchers_and_schedulers.ray_searchers_and_schedulers import get_whisper_hyperparameters as get_hyperparameters

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Datasets
from data_and_collator.datasets_and_collators import get_datasets_and_collators, make_dataset_kwargs
from projects_paths import DATA_PATH

from pathlib import Path
import csv
from typing import Optional, List, Union, Dict, Any, Tuple

# keep existing collator (DO NOT CHANGE)
# NOTE: this is whatever your get_datasets_and_collators returns as data_collators["val"]
# and you requested not to touch collate_parquet implementation.

# Logging control
os.environ["RAY_AIR_NEW_OUTPUT"] = "1"
os.environ["RAY_VERBOSITY"] = "1"
ray.data.context.DataContext.get_current().enable_operator_progress_bars = True
ray.data.context.DataContext.get_current().enable_progress_bars = True

logger = logging.getLogger(__name__)

TUNE_CHOICES = ['small_small', 'large_small_OPTUNA', 'large_large']
DATA_MODES = ['h5', 'parquet', 'parquet_h5']


def parse_args():
    parser = configargparse.ArgumentParser()

    # Training settings for Seq2SeqTrainingArguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_tag", type=str, default="whisper-tiny-de")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--generation_max_length", type=int, default=225)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--eval_delay", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=25)

    # model settings
    parser.add_argument("--model_type", type=str, default="openai/whisper-tiny")
    parser.add_argument("--target_language", type=str, default="german")
    parser.add_argument("--return_timestamps", action="store_true")
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--simple", action="store_true")

    # Dataset settings
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--h5", action="store_true")
    parser.add_argument("--data_mode", type=str, default="h5", choices=DATA_MODES)

    # Hyperparameter Optimization settings for Ray Tune
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_warmup_steps", type=int, default=10)
    parser.add_argument("--len_train_set", type=int, default=10)
    parser.add_argument("--max_concurrent_trials", type=int, default=1)
    parser.add_argument("--prefetch_batches", type=int, default=1)
    parser.add_argument("--optuna_db_path", type=str, default=None)
    parser.add_argument("--load_ds_in_trainer", action="store_true", default=False)
    parser.add_argument("--optuna_study_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_to_keep", type=int, default=1)
    parser.add_argument("--max_t", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--cpus_per_trial", type=int, default=1)
    parser.add_argument("--gpus_per_trial", type=float, default=0)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--reuse_actors", action="store_true")
    parser.add_argument("--metric_to_optimize", type=str, default="eval_loss")
    parser.add_argument("--wer_weight", type=float, default=1.0)
    parser.add_argument("--modes", type=list_of_strings, action="append")
    parser.add_argument("--eval_sample_fraction", type=float, default=1.0)
    parser.add_argument("--search_schedule_mode", type=str, default="large_small_BOHB", choices=TUNE_CHOICES)
    parser.add_argument("--reduction_factor", type=int, default=2)
    parser.add_argument("--grace_period", type=int, default=5000)
    parser.add_argument("--perturbation_interval", type=int, default=10)
    parser.add_argument("--burn_in_period", type=int, default=1)
    parser.add_argument('--hyperparameters', type=list_of_strings, action="append")

    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    # Other settings
    parser.add_argument("--run_on_local_machine", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--storage_path", type=str, default="./output/scratch")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--path_to_data", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="eg_dataset_subset_1000.h5")
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")

    return parser.parse_args()


import math
import ray
import torch
from typing import Optional, List, Dict
from torch.utils.data import Dataset


def partition_dataset(ds, fraction: float):
    total_len = ds.count()
    n_splits = math.ceil(1 / fraction)
    target_size = math.ceil(total_len * fraction)
    split_indices = [min((i + 1) * target_size, total_len) for i in range(n_splits - 1)]
    splits = ds.split_at_indices(split_indices)
    val_sets = {f"val_{i+1}": subset.materialize() for i, subset in enumerate(splits)}
    return val_sets


# ---------------------------
# MINIMAL PATCHES (LOCAL ONLY)
# ---------------------------

import torch
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer

class RayIterableDataset(torch.utils.data.IterableDataset):
    """Wraps a Python iterator so torch/HF treats it as an IterableDataset (no __len__)."""
    def __init__(self, it):
        self.it = it
    def __iter__(self):
        yield from self.it


def data_collator_id_unwrap_list_of_one(features):
    """
    HF DataLoader calls collate_fn with a LIST of dataset items.
    Our dataset items are already full batch dicts -> unwrap list-of-one.
    """
    if isinstance(features, list):
        # With get_eval_dataloader(batch_size=1), this will always be 1.
        if len(features) != 1:
            raise ValueError(
                f"[data_collator_id_unwrap_list_of_one] Expected list length 1, got {len(features)} "
                f"(HF is batching your batches)."
            )
        return features[0]
    return features


class Seq2SeqTrainerEvalSamplingPatched(Seq2SeqTrainer):
    """
    Local patched copy:
    - wraps Ray iterator in torch IterableDataset
    - forces HF eval DataLoader batch_size=1 to avoid batching batches
    """
    def __init__(self, *args, eval_sample_fraction=0.1, prefetch_batches=1, eval_collator=None, wer_weight=1.0,
                 validation_summary_csv: str = "./output/large_small_BOHB/large_v3_eval/validation_summary.csv", **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_batches = prefetch_batches
        self.eval_sample_fraction = eval_sample_fraction
        self.eval_collator = eval_collator
        self.wer_weight = wer_weight
        # Load CSV once into a lookup: {"val_17": 17.8019, ...}
        self._val_wer_lookup = self._load_validation_wer_lookup(validation_summary_csv)
        
    @staticmethod
    def _load_validation_wer_lookup(csv_path: str) -> Dict[str, float]:
        path = Path(csv_path)
        lookup: Dict[str, float] = {}

        if not path.exists():
            # Don’t crash training; just skip the diff metric if missing.
            print(f"[Eval]: WARNING: validation summary CSV not found: {path}")
            return lookup

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shard = row.get("shard")
                wer = row.get("eval_wer")
                if shard is None or wer is None:
                    continue
                try:
                    lookup[str(shard)] = float(wer)
                except ValueError:
                    continue
        return lookup

    def get_eval_dataloader(self, eval_dataset=None):
        """
        CRITICAL FIX:
        Our eval_dataset yields already-collated BATCH dicts.
        HF must NOT batch them again -> force batch_size=1 here.
        """
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            ds,
            batch_size=1,
            collate_fn=self.data_collator,   # will unwrap list-of-one
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval",
                 max_length=None, num_beams=None, eval_key=None):

        if eval_key is not None:
            shard_key = eval_key
        else:
            import random
            shard_key = random.choice(list(self.eval_shards.keys()))

        random_ds = self.eval_shards[shard_key]

        ds_iter = random_ds.iter_torch_batches(
            prefetch_batches=self.prefetch_batches,
            batch_size=self.args.per_device_eval_batch_size,   # this is the *real* batch size
            collate_fn=self.eval_collator,                     # your collate_parquet unchanged
        )

        ds = RayIterableDataset(ds_iter)
        self.eval_dataset = ds  # so get_eval_dataloader() uses it

        print(f"\n[Eval] Selected shard: {shard_key}")

        # nice peek
        try:
            peek_iter = iter(RayIterableDataset(random_ds.iter_torch_batches(
                prefetch_batches=1,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.eval_collator,
            )))
            first = next(peek_iter)
            shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v)) for k, v in first.items()}
            print("[Eval] First batch keys/shapes:", shapes)
        except Exception as e:
            print("[Eval] Could not peek first batch:", repr(e))

        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        metrics = super().evaluate(ds, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        if "eval_loss" in metrics and "eval_wer" in metrics:
            beta = self.wer_weight
            alpha = 1 - beta
            metrics["eval_loss_wer"] = alpha * metrics["eval_loss"] + beta * metrics["eval_wer"]
        
        if "eval_wer" in metrics:
            shard_key = shard_key #f"val_{int(shard_key)}"  # handles "17" or 17
            csv_eval_wer = self._val_wer_lookup.get(shard_key)

            if csv_eval_wer is not None:
                metrics["eval_wer_diff"] = float(metrics["eval_wer"]) - csv_eval_wer 
            else:
                # optional: keep it predictable if missing
                metrics["eval_wer_diff"] = float("nan")
                print(f"[Eval]: WARNING: shard {shard_key} not found in validation_summary.csv")
        
        if "eval_loss_wer" in metrics:
            print("eval_loss_wer", metrics["eval_loss_wer"])
        if "eval_wer_diff" in metrics:
            print("eval_wer_diff", metrics["eval_wer_diff"])
        
        print(f"[Eval] Done shard {shard_key} | eval_loss={metrics.get('eval_loss')} | eval_wer={metrics.get('eval_wer')}")
        return metrics

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    logger.info("Hi!")
    set_seed(args.random_seed)

    config_ = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__))
    args.output_dir = os.path.join(args.output_dir, args.search_schedule_mode, args.output_tag)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_file(config_, args.output_dir)

    print("Already initialized?", ray.is_initialized())
    ip_head = os.getenv("ip_head")
    print("ip_head:", repr(ip_head))
    if not ip_head:
        raise RuntimeError("Head node address not found in environment variables.")
    ray.init(address=ip_head)

    logger.info("Ray Nodes info: %s", ray.nodes())
    logger.info("Ray Cluster Resources: %s", ray.cluster_resources())

    # STEP 1: Data loading
    args.path_to_data = DATA_PATH
    args.data_mode = "parquet"
    dataset_kwargs = make_dataset_kwargs(args)

    ray_datasets_, data_collators = get_datasets_and_collators(dataset_kwargs)
    train_ds = ray_datasets_["train"]
    val_ds = ray_datasets_["val"]
    
    args.eval_sample_fraction = 0.001
    val_subsets = partition_dataset(val_ds, fraction=args.eval_sample_fraction)

    logger.info(
        f"Created {len(val_subsets)} validation subsets, "
        f"each about {args.eval_sample_fraction * 100:.1f}% of total."
    )

    # Model loading (unchanged)
    from models.whisper_models import get_whisper_models

    base_training_kwargs = make_training_kwargs(args)

    model, feature_extractor, tokenizer, processor = get_whisper_models(
        base_training_kwargs["model_type"],
        base_training_kwargs["target_language"],
        return_timestamps=base_training_kwargs["return_timestamps"],
        load_in_8bit=base_training_kwargs["peft"],
        local=base_training_kwargs["run_on_local_machine"],
    )

    logger.info("Starting Validation for model %s", args.model_type)

    from trainers.metrics import get_metric_to_optimize
    compute_metrics = get_metric_to_optimize("wer", tokenizer=tokenizer)

    results = {}

    # IMPORTANT: do not mutate base_training_kwargs across shards
    for val_key in val_subsets.keys():
        logger.info(f"[EVAL] Validation subset {val_key}")

        training_kwargs = dict(base_training_kwargs)  # minimal but critical

        # keep your deletions as-is (but on the per-iteration copy)
        del training_kwargs["model_type"]
        del training_kwargs["target_language"]
        del training_kwargs["return_timestamps"]
        del training_kwargs["run_on_local_machine"]
        del training_kwargs["len_train_set"]
        del training_kwargs["num_train_epochs"]
        prefetch_batches_ = training_kwargs["prefetch_batches"]
        wer_weight_ = training_kwargs["wer_weight"]
        del training_kwargs["prefetch_batches"]
        del training_kwargs["wer_weight"]
        del training_kwargs["peft"]

        training_kwargs["dataloader_num_workers"] = 0

        training_args = Seq2SeqTrainingArguments(
            eval_strategy="steps",
            save_strategy="steps",
            report_to=["tensorboard"],
            load_best_model_at_end=False,
            greater_is_better=False,
            push_to_hub=False,
            do_eval=True,
            dataloader_persistent_workers=False,
            dataloader_pin_memory=True,
            group_by_length=True,  # minimal safety for custom batch dicts
            **training_kwargs,
        )

        print("Data Collator:", data_collators["val"])

        trainer = Seq2SeqTrainerEvalSamplingPatched(
            eval_sample_fraction=args.eval_sample_fraction,
            prefetch_batches=args.prefetch_batches,
            eval_dataset=val_subsets,  # we use eval_shards instead (dict)
            eval_collator=data_collators["val"],  # keep as-is
            wer_weight=args.wer_weight,
            args=training_args,
            model=model,
            train_dataset=None,
            data_collator=data_collator_id_unwrap_list_of_one,  # LOCAL PATCHED ID COLLATOR
            compute_metrics=compute_metrics,
            callbacks=None,
        )

        trainer.eval_shards = val_subsets

        metrics = trainer.evaluate(eval_key=val_key)
        results[val_key] = metrics

    # ---------------------------
    # Print + save summary
    # ---------------------------
    print("\n====================")
    print("EVAL SUMMARY (per shard)")
    print("====================")

    summary_rows = []
    for k in sorted(results.keys(), key=lambda x: int(x.split("_")[-1]) if "_" in x else x):
        m = results[k]
        row = {
            "shard": k,
            "eval_loss": m.get("eval_loss", None),
            "eval_wer": m.get("eval_wer", None),
        }
        summary_rows.append(row)
        print(f"{k:>10} | eval_loss={row['eval_loss']} | eval_wer={row['eval_wer']}")

    # Save to disk in output_dir
    import json
    summary_path_json = os.path.join(args.output_dir, "validation_summary.json")
    summary_path_csv = os.path.join(args.output_dir, "validation_summary.csv")

    with open(summary_path_json, "w") as f:
        json.dump({"results": results, "summary": summary_rows}, f, indent=2)

    # simple csv
    with open(summary_path_csv, "w") as f:
        f.write("shard,eval_loss,eval_wer\n")
        for r in summary_rows:
            f.write(f"{r['shard']},{r['eval_loss']},{r['eval_wer']}\n")

    print("\nSaved:")
    print(" -", summary_path_json)
    print(" -", summary_path_csv)

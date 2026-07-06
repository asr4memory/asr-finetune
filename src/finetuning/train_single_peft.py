"""Single-run PEFT finetuning script for Whisper ASR.

Minimal-change version:
- keeps configargparse config style
- keeps Ray only for dataset handling
- removes Ray Tune / TorchTrainer / searcher / scheduler
- adds verbose progress prints
"""

import os
import sys
import math
import pprint
import logging
from pathlib import Path

# Make src/ importable so the finetuning package resolves whether this is run via
# ``python -m finetuning.train_single_peft`` or directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ray
import ray.data
import configargparse
from transformers import set_seed

from finetuning.utils import list_of_strings, save_file
from finetuning.data_and_collator.datasets_and_collators import get_datasets_and_collators, make_dataset_kwargs
from finetuning.projects_paths import DATA_PATH, VALIDATION_SUMMARY_CSV

from finetuning.trainers.trainers import make_seq2seq_training_kwargs as make_training_kwargs
from finetuning.trainers.trainers_single import train_whisper_peft_model_single

logger = logging.getLogger(__name__)

DATA_MODES = ["h5", "parquet", "parquet_h5"]


def parse_args():
    parser = configargparse.ArgumentParser()

    # training
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_tag", type=str, default="whisper-single-peft")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--generation_max_length", type=int, default=225)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--eval_delay", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=25)

    # fixed hyperparameters for single run
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--alpha", type=int, default=None,
                        help="Absolute lora_alpha. Required unless --alpha_coupled is set.")
    parser.add_argument("--alpha_coupled", type=int, default=None,
                        help="If set, lora_alpha = alpha_coupled * target_r (matches the "
                             "Phase-1 hailmary search space). Overrides --alpha.")
    parser.add_argument("--target_r", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout. Must match the source trial for faithful "
                             "reproduction; Phase-1 sampled {0.0, 0.05, 0.1}.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="Adapter EMA decay; applied at eval to mirror HPO trainer.")
    parser.add_argument("--ema_start_step", type=int, default=0,
                        help="Step after which the EMA shadow starts updating.")

    # model
    parser.add_argument("--model_type", type=str, default="openai/whisper-tiny")
    parser.add_argument("--target_language", type=str, default="german")
    parser.add_argument("--return_timestamps", action="store_true")
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--simple", action="store_true")

    # dataset
    parser.add_argument("--cpus_per_trial", type=int, default=2,
                    help="Used by dataset/collator loading code.")
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--h5", action="store_true")
    parser.add_argument("--data_mode", type=str, default="h5", choices=DATA_MODES)
    parser.add_argument("--len_train_set", type=int, default=10)
    parser.add_argument("--prefetch_batches", type=int, default=0)
    parser.add_argument("--load_ds_in_trainer", action="store_true", default=False)
    parser.add_argument("--eval_sample_fraction", type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--shard_schedule", type=str, default="deterministic",
                        choices=["deterministic", "random"])

    # metrics / eval
    parser.add_argument("--metric_to_optimize", type=list_of_strings, action="append")
    parser.add_argument("--wer_weight", type=float, default=1.0)

    # other
    parser.add_argument("--run_on_local_machine", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_from_output_dir", action="store_true", default=False,
                        help="When set with --resume_training, look in output_dir for the latest "
                             "checkpoint-N and use it instead of --resume_from_checkpoint. "
                             "Falls back to --resume_from_checkpoint if no local checkpoints exist.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--path_to_data", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="eg_dataset_subset_1000.h5")
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--validation_summary_csv", type=str,
                        default=None,
                        help="Per-shard whisper-large-v3 baseline CSV. Defaults to "
                             "<repo>/trainers/data/validation_summary.csv via projects_paths.")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")

    return parser.parse_args()


def partition_dataset(ds, fraction: float):
    print("[partition_dataset] counting validation dataset...")
    total_len = ds.count()
    n_splits = math.ceil(1 / fraction)
    target_size = math.ceil(total_len * fraction)
    print(f"[partition_dataset] total_len={total_len} fraction={fraction} "
          f"n_splits={n_splits} target_size={target_size}")
    split_indices = [min((i + 1) * target_size, total_len) for i in range(n_splits - 1)]
    splits = ds.split_at_indices(split_indices)
    out = {f"val_{i+1}": subset.materialize() for i, subset in enumerate(splits)}
    print(f"[partition_dataset] materialized {len(out)} validation subsets")
    return out


if __name__ == "__main__":
    print("[main] STEP 0: parse args")
    args = parse_args()
    if args.validation_summary_csv is None:
        args.validation_summary_csv = VALIDATION_SUMMARY_CSV

    logging.basicConfig(
        format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
        datefmt="%H:%M",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    logger.info("Hi!")
    print("[main] STEP 1: set seed")
    set_seed(args.random_seed)

    print("[main] STEP 2: prepare output dir")
    args.output_dir = os.path.join(args.output_dir, "single_run", args.output_tag)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[main] output_dir={args.output_dir}")

    if args.auto_resume_from_output_dir and args.resume_training:
        ckpt_dirs = []
        for name in os.listdir(args.output_dir):
            if not name.startswith("checkpoint-"):
                continue
            suffix = name[len("checkpoint-"):]
            if suffix.isdigit() and os.path.isdir(os.path.join(args.output_dir, name)):
                ckpt_dirs.append((int(suffix), name))
        if ckpt_dirs:
            ckpt_dirs.sort()
            latest_step, latest_name = ckpt_dirs[-1]
            latest_path = os.path.join(args.output_dir, latest_name)
            print(f"[auto-resume] Found {len(ckpt_dirs)} local checkpoints; "
                  f"using latest: {latest_path} (step {latest_step})")
            args.resume_from_checkpoint = latest_path
        else:
            print(f"[auto-resume] No local checkpoints in {args.output_dir}; "
                  f"falling back to config resume_from_checkpoint={args.resume_from_checkpoint}")

    print("[main] STEP 3: save config dump")
    config_dump = "Parsed args:\n{}\n\n".format(pprint.pformat(args.__dict__))
    save_file(config_dump, args.output_dir)

    print("[main] STEP 4: initialize/connect Ray")
    print(f"[main] run_on_local_machine={args.run_on_local_machine}")
    print(f"[main] ray.is_initialized() before init = {ray.is_initialized()}")

    if not ray.is_initialized():
        if args.run_on_local_machine:
            print("[main] starting local Ray instance")
            ray.init(ignore_reinit_error=True)
        else:
            ip_head = os.getenv("ip_head")
            print(f"[main] ip_head={repr(ip_head)}")
            if not ip_head:
                raise RuntimeError("Head node address not found in environment variables.")
            print("[main] connecting to existing Ray cluster")
            ray.init(address=ip_head)

    print(f"[main] ray.is_initialized() after init = {ray.is_initialized()}")
    logger.info("Ray Nodes info: %s", ray.nodes())
    logger.info("Ray Cluster Resources: %s", ray.cluster_resources())

    print("[main] STEP 5: set data path/mode")
    args.path_to_data = DATA_PATH
    args.data_mode = "parquet"
    print(f"[main] DATA_PATH={DATA_PATH}")
    print(f"[main] args.path_to_data={args.path_to_data}")
    print(f"[main] args.dataset_name={args.dataset_name}")
    print(f"[main] args.data_mode={args.data_mode}")

    print("[main] STEP 6: build dataset kwargs")
    dataset_kwargs = make_dataset_kwargs(args)
    print(f"[main] dataset_kwargs={dataset_kwargs}")

    print("[main] STEP 7: call get_datasets_and_collators")
    ray_datasets_, data_collators = get_datasets_and_collators(dataset_kwargs)
    print("[main] returned from get_datasets_and_collators")

    print("[main] STEP 8: extract train/val datasets")
    train_ds = ray_datasets_["train"]
    val_ds = ray_datasets_["val"]
    print(f"[main] train_ds={train_ds}")
    print(f"[main] val_ds={val_ds}")

    print("[main] STEP 9: partition validation dataset")
    val_subsets = partition_dataset(val_ds, fraction=args.eval_sample_fraction)
    ray_datasets = {"train": train_ds, **val_subsets}

    logger.info(
        "Created %s validation subsets, each about %.1f%% of total.",
        len(val_subsets), args.eval_sample_fraction * 100.0
    )

    print("[main] STEP 10: count train dataset")
    args.len_train_set = ray_datasets["train"].count()
    logger.info("len_train_set: %s", args.len_train_set)

    print("[main] STEP 11: count first few validation subsets")
    for name in list(val_subsets.keys())[:3]:
        logger.info("%s size: %s", name, ray_datasets[name].count())

    logger.info("Starting single PEFT finetuning for model %s", args.model_type)

    print("[main] STEP 12: build training kwargs")
    training_kwargs = make_training_kwargs(args)
    print(f"[main] training_kwargs keys={list(training_kwargs.keys())}")

    print("[main] STEP 13: start training function")
    if args.alpha_coupled is not None:
        effective_alpha = int(args.alpha_coupled) * int(args.target_r)
        print(f"[main] alpha_coupled={args.alpha_coupled} target_r={args.target_r} "
              f"-> lora_alpha={effective_alpha} (overriding --alpha={args.alpha})")
    elif args.alpha is not None:
        effective_alpha = int(args.alpha)
    else:
        raise ValueError("Pass either --alpha or --alpha_coupled "
                         "(lora_alpha = alpha_coupled * target_r).")

    train_whisper_peft_model_single(
        args=args,
        config={
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "alpha": effective_alpha,
            "target_r": args.target_r,
            "lr_scheduler_type": args.lr_scheduler_type,
            "lora_dropout": args.lora_dropout,
        },
        training_kwargs=training_kwargs,
        ray_datasets=ray_datasets,
        data_collators=data_collators,
        eval_names=list(val_subsets.keys()),
    )

    print("[main] STEP 14: training finished")

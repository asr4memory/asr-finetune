import json
import os
from typing import Dict, Any, Optional, List

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Seq2SeqTrainingArguments

from models.whisper_models import get_whisper_models
from .custom_seq2seq_trainers_single import Seq2SeqTrainerEvalSamplingPeftSingle
from .utils import data_collator_id, SavePeftModelCallback, LoadAdapterFromSubdirCallback
from .ema_callback import AdapterEMACallback
from utils import steps_per_epoch

import numpy as np
from torch.utils.data import IterableDataset


class RayRowIterableDataset(IterableDataset):
    """Wrap a Ray Dataset as a torch IterableDataset that yields one row dict at a time.

    HF Trainer's DataLoader then batches with per_device_train_batch_size and applies
    the collate_fn. Crucially: no rows_to_skip - we rely on ignore_data_skip=True so
    each new epoch starts from row 0.
    """

    def __init__(self, ray_dataset):
        self.ray_dataset = ray_dataset

    def __iter__(self):
        return iter(self.ray_dataset.iter_rows())


def collate_parquet_for_trainer(batch):
    """Collate a list of row dicts (numpy arrays from Ray) into batched tensors."""
    input_features_batch = torch.stack([
        torch.from_numpy(row["input_features"]) if isinstance(row["input_features"], np.ndarray)
        else torch.as_tensor(row["input_features"])
        for row in batch
    ])
    labels_batch = torch.stack([
        torch.from_numpy(row["labels"]) if isinstance(row["labels"], np.ndarray)
        else torch.as_tensor(row["labels"])
        for row in batch
    ])
    return {"input_features": input_features_batch, "labels": labels_batch}


def _load_checkpoint_trainer_state(checkpoint_path: str) -> Dict[str, Any]:
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(
            f"Could not find trainer_state.json in checkpoint: {trainer_state_path}"
        )

    with open(trainer_state_path, "r") as f:
        return json.load(f)


def _load_checkpoint_training_args(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    training_args_path = os.path.join(checkpoint_path, "training_args.bin")
    if not os.path.exists(training_args_path):
        return None

    try:
        training_args = torch.load(training_args_path, map_location="cpu")
    except Exception as e:
        print(f"[resume-debug] Failed to load training_args.bin: {e}")
        return None

    if hasattr(training_args, "to_dict"):
        return training_args.to_dict()
    if hasattr(training_args, "__dict__"):
        return dict(training_args.__dict__)
    return None


def _get_resume_row_offset(
    checkpoint_path: str,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    trainer_state = _load_checkpoint_trainer_state(checkpoint_path)
    global_step = int(trainer_state.get("global_step", 0) or 0)
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)

    rows_per_optimizer_step = (
        per_device_train_batch_size * gradient_accumulation_steps * world_size
    )
    rows_to_skip = global_step * rows_per_optimizer_step

    print(
        "[resume] global_step=", global_step,
        "world_size=", world_size,
        "rows_per_optimizer_step=", rows_per_optimizer_step,
        "rows_to_skip=", rows_to_skip,
    )

    return rows_to_skip


def _print_resume_debug(checkpoint_path: str, training_args: Seq2SeqTrainingArguments) -> None:
    checkpoint_training_args = _load_checkpoint_training_args(checkpoint_path)
    current_training_args = training_args.to_dict()

    keys_to_compare = [
        "per_device_train_batch_size",
        "train_batch_size",
        "gradient_accumulation_steps",
        "world_size",
        "n_gpu",
        "local_rank",
        "deepspeed",
        "fp16",
        "max_steps",
    ]

    print("[resume-debug] Comparing checkpoint training args with current training args")
    for key in keys_to_compare:
        checkpoint_value = None if checkpoint_training_args is None else checkpoint_training_args.get(key)
        current_value = current_training_args.get(key)
        print(
            f"[resume-debug] {key}: "
            f"checkpoint={checkpoint_value!r} current={current_value!r}"
        )

def train_whisper_peft_model_single(
    args,
    config: Dict[str, Any],
    training_kwargs: Optional[Dict[str, Any]] = None,
    ray_datasets: Optional[Dict[str, Any]] = None,
    data_collators: Optional[Dict[str, Any]] = None,
    eval_names: Optional[List[str]] = None,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model, feature_extractor, tokenizer, processor = get_whisper_models(
        training_kwargs["model_type"],
        training_kwargs["target_language"],
        return_timestamps=training_kwargs["return_timestamps"],
        load_in_8bit=False,
        local=training_kwargs["run_on_local_machine"],
    )

    model = prepare_model_for_kbit_training(model)

    def make_inputs_require_grad(module, input, output):
        output = output.to(input[0].device)
        output.requires_grad_(True)
        return output

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    training_kwargs["max_steps"] = (
        steps_per_epoch(
            training_kwargs["len_train_set"],
            config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_kwargs["gradient_accumulation_steps"],
        )
        * training_kwargs["num_train_epochs"]
    )

    if args.max_steps:
        training_kwargs["max_steps"] = args.max_steps

    resume_from_checkpoint = args.resume_from_checkpoint if args.resume_training else None

    if resume_from_checkpoint:
        ckpt_training_args = _load_checkpoint_training_args(resume_from_checkpoint)
        if ckpt_training_args is not None:
            for key in ("max_steps", "warmup_ratio", "warmup_steps",
                        "lr_scheduler_type", "learning_rate"):
                if key in ckpt_training_args and ckpt_training_args[key] is not None:
                    new_val = ckpt_training_args[key]
                    old_val = training_kwargs.get(key, config.get(key))
                    if old_val != new_val:
                        print(f"[resume] Pinning {key}: {old_val!r} -> {new_val!r} (from checkpoint training_args.bin)")
                    if key in training_kwargs:
                        training_kwargs[key] = new_val
                    if key in config:
                        config[key] = new_val
                    if key == "max_steps":
                        training_kwargs["max_steps"] = new_val

    # Rank-coupled alpha (mirrors the HPO path in trainers.py).
    if "alpha_coupled" in config:
        config["alpha"] = int(config["alpha_coupled"]) * int(config["target_r"])
        del config["alpha_coupled"]

    lora_dropout = float(config.get("lora_dropout", 0.05))
    # Plain LoRA + DoRA. See trainers.py:train_whisper_peft_model for the full
    # rationale on why this replaces AdaLoraConfig and why PiSSA is currently
    # back-pocket (CPU SVD over 384 q/k/v/o weights hangs trial init for >6 min).
    lora_config = LoraConfig(
        r=int(config["target_r"]),
        lora_alpha=int(config["alpha"]),
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_dora=True,
    )

    model = get_peft_model(model, lora_config)
    model.to(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.print_trainable_parameters()
    model.config.use_cache = False

    train_ds = ray_datasets["train"]
    eval_shards = {name.split("_")[-1]: ray_datasets[name] for name in eval_names}

    train_ds_iterable = RayRowIterableDataset(train_ds)

    callbacks_ = [SavePeftModelCallback]
    if resume_from_checkpoint:
        callbacks_.append(LoadAdapterFromSubdirCallback(resume_from_checkpoint))

    # Adapter EMA mirrors the HPO path (trainers.py:594-599). Read kwargs out of
    # training_kwargs before they're stripped, then bind the callback onto the
    # trainer below so custom evaluate() can apply/restore the shadow.
    ema_decay_ = float(training_kwargs.pop("ema_decay", 0.99))
    ema_start_step_ = int(training_kwargs.pop("ema_start_step", 0))
    shard_schedule_ = training_kwargs.pop("shard_schedule", "deterministic")
    ema_callback = AdapterEMACallback(decay=ema_decay_, start_step=ema_start_step_)
    callbacks_.append(ema_callback)

    del training_kwargs["model_type"]
    wer_weight_ = training_kwargs["wer_weight"]
    del training_kwargs["wer_weight"]
    target_language_ = training_kwargs["target_language"]
    del training_kwargs["target_language"]
    del training_kwargs["return_timestamps"]
    del training_kwargs["run_on_local_machine"]
    del training_kwargs["len_train_set"]
    # Keep num_train_epochs in training_kwargs: with max_steps>0 it does NOT
    # control termination (max_steps wins per HF docs), but it IS used by HF
    # Trainer to compute num_update_steps_per_epoch, which in turn drives the
    # resume fast-forward arithmetic (epochs_trained, steps_trained_in_current_
    # _epoch). Dropping it makes HF default to 3.0 and skews the fast-forward
    # so multi-slot resume re-trains rows already seen in slot 1. See
    # ignore_data_skip=False below.
    prefetch_batches_ = training_kwargs["prefetch_batches"]
    del training_kwargs["prefetch_batches"]
    del training_kwargs["peft"]
    # `_user_max_steps` is stashed by make_seq2seq_training_kwargs for the HPO
    # path's late override; the single path uses args.max_steps directly above,
    # so this key must be dropped before reaching Seq2SeqTrainingArguments.
    training_kwargs.pop("_user_max_steps", None)

    config = dict(config)
    del config["alpha"]
    del config["target_r"]
    config.pop("lora_dropout", None)

    if "dataloader_num_workers" in training_kwargs:
        del training_kwargs["dataloader_num_workers"]
    if "deepspeed" in training_kwargs:
        del training_kwargs["deepspeed"]
        
    training_args = Seq2SeqTrainingArguments(
        eval_strategy="steps",
        save_strategy="steps",
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        greater_is_better=False,
        push_to_hub=False,
        do_eval=True,
        dataloader_pin_memory=True,
        group_by_length=False,
        # ignore_data_skip=False -> on resume, HF Trainer iterates the
        # dataloader to fast-forward past batches already seen in slot 1, so
        # each parquet row is trained on exactly once across all slots. Cost
        # is ~5-10 min of slot-2 startup iterating-and-discarding rows; the
        # parquet stores pre-computed mel features so no audio decode happens.
        ignore_data_skip=False,
#        remove_unused_columns=False,
#        label_names=["labels"],
#        predict_with_generate=False,
#        gradient_checkpointing=False,
#        metric_for_best_model="eval_wer_diff",
#        dataloader_num_workers=0,
#        dataloader_persistent_workers=False,
        # IMPORTANT: no deepspeed here
        **config,
        **training_kwargs,
    )

    trainer = Seq2SeqTrainerEvalSamplingPeftSingle(
        processor=processor,
        tokenizer=tokenizer,
        eval_sample_fraction=args.eval_sample_fraction,
        prefetch_batches=0,
        eval_dataset=eval_shards["1"],
        eval_collator=data_collators["val"],
        wer_weight=wer_weight_,
        validation_summary_csv=args.validation_summary_csv,
        language=target_language_,
        task="transcribe",
        max_eval_batches=args.max_eval_batches,
        args=training_args,
        model=model,
        train_dataset=train_ds_iterable,
        data_collator=collate_parquet_for_trainer,
        compute_metrics=None,
        callbacks=callbacks_,
    )

    trainer.eval_shards = eval_shards
    trainer.shard_schedule = shard_schedule_
    trainer.ema_callback = ema_callback

    if resume_from_checkpoint:
        # AdapterEMACallback._maybe_load_snapshot reads RESUME_CKPT_DIR as a
        # fallback because HF TrainingArguments doesn't propagate the resume
        # path through to callbacks. Mirrors trainers.py:713-718.
        os.environ["RESUME_CKPT_DIR"] = str(resume_from_checkpoint)
        _print_resume_debug(resume_from_checkpoint, training_args)
        print(f"Resuming from checkpoint {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    return trainer

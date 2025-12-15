"""
Module: trainers.py

This module provides utilities for training Whisper ASR models using Hugging Face Transformers and Ray Tune.
It supports both standard training and parameter-efficient fine-tuning (PEFT) techniques such as LoRA and AdaLoRA.

Key Features:
-------------
- Supports full and PEFT-based fine-tuning of Whisper models.
- Integrates with Ray Tune for scalable hyperparameter search.
- Handles model checkpointing, evaluation sampling, and distributed training.
- Custom trainer classes for evaluation with WER weighting and eval-set sub-sampling.
- Compatible with Ray Datasets and Hugging Face Seq2SeqTrainer.

Main Functions:
---------------
- `train_whisper_model(...)`:
    Standard training loop using Hugging Face Transformers (no PEFT).

- `train_whisper_peft_model(...)`:
    Training loop with PEFT support (e.g., AdaLoRA), including gradient hooks and LoRA rank scheduling.

Notes:
------
- Assumes pre-tokenized and preprocessed datasets.
- Identity collator is used for efficiency since input is already collated.

Dependencies:
-------------
- Hugging Face Transformers
- Ray (Tune, Train, Datasets)
- PEFT (LoRA/AdaLoRA)
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import random
from typing import Optional, List, Union, Dict, Any, Tuple

# ray
import ray
import ray.train
from ray import tune
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback

# get models
from models.whisper_models import get_whisper_models
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, LoraModel, AdaLoraConfig, get_peft_model

# get data
from data_and_collator.datasets_and_collators import get_datasets_and_collators
from data_and_collator.hf_to_ray_custom_utils import prepare_trainer_custom

# HuggingFace
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# torch
from torch.utils.data import DataLoader, Dataset

# internal stuff
from .utils import load_checkpoints, StepSyncCallback, SavePeftModelCallback, data_collator_id
from .metrics import get_metric_to_optimize
from utils import  steps_per_epoch, normalize

from transformers.trainer_utils import EvalLoopOutput
import random

class Seq2SeqTrainerEvalSampling(Seq2SeqTrainer):
    """
    Custom Hugging Face Seq2SeqTrainer class that supports:
    - Evaluating on a random subset of the evaluation dataset.
    - Weighted combination of loss and Word Error Rate (WER) as a custom metric.
    
    Args:
        eval_sample_fraction (float): Fraction of evaluation dataset to sample (between 0 and 1).
        prefetch_batches (int): Number of batches to prefetch for dataloader performance.
        eval_collator (Callable): Collation function used during evaluation.
        wer_weight (float): Weight given to the WER metric in combined loss.
    """
    def __init__(self, *args, eval_sample_fraction=0.1, prefetch_batches = 1, eval_collator=None, wer_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        print("AAAAAAAAAAAAAAA")
        self.prefetch_batches = prefetch_batches
        self.eval_sample_fraction = eval_sample_fraction
        self.eval_collator = eval_collator
        self.wer_weight = wer_weight

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics, optionally computing a custom eval_loss_wer.
        """
        print("I AM HERE!")
#        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
#        
#        # Your dictionary
#        eval_shards = {"1": eval_ds_1, "2": eval_ds_2, "3": eval_ds_3}

        # Pick a random key
        random_key = random.choice(list(self.eval_shards.keys()))

        # Get the corresponding dataset
        random_ds = self.eval_shards[random_key]
        ds = random_ds.iter_torch_batches(
                        prefetch_batches = self.prefetch_batches,
                        batch_size=self.args.per_device_eval_batch_size,
                        collate_fn=self.eval_collator
                        )

        print(f"[Eval]: Selected shard: {random_key}")

        print("[Eval]: Print eval dataset ", ds)
#        print("Length eval dataset ", ds.count())
        
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        metrics = super().evaluate(ds, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        if "eval_loss" in metrics and "eval_wer" in metrics:
            beta = self.wer_weight
            alpha = 1 - beta
            metrics["eval_loss_wer"] = alpha * metrics["eval_loss"] + beta * metrics["eval_wer"]
            
        print("eval_loss_wer", metrics["eval_loss_wer"])
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self.log(metrics)
        
        return metrics
        

import torch
import torch.nn as nn

from .custom_seq2seq_trainers import Seq2SeqTrainerEvalSamplingPeft

class Seq2SeqTrainerEvalSamplingPeft_old(Seq2SeqTrainer):

    def __init__(self, *args, eval_sample_fraction=0.1, prefetch_batches = 1, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        print("AAAAAAAAAAAAAAA")
        self.prefetch_batches = prefetch_batches
        self.eval_sample_fraction = eval_sample_fraction
        self.eval_collator = eval_collator
        
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            forced_decoder_ids=forced_decoder_ids
            
            print("where are we?")
            if is_sagemaker_mp_enabled():
                print("A")
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    print("Aa")
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    print("Ab")
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                print("B")
                if has_labels:
                    print("Ba")
                    if self.use_amp:
                        with autocast():
                            loss, outputs = self.compute_loss(model, inputs, return_outputs=True, eval=True)
                    else:
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True, eval=True)
                    loss = loss.mean().detach()
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    print("Bb")
                    loss = None
                    if self.use_amp:
                        with autocast():
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        print("I AM HERE!!!")

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if self.tokenizer is not None:
            generation_inputs = {k: v for k, v in inputs.items() if k in self.tokenizer.model_input_names}
            # very ugly hack to make it work
            generation_inputs["input_ids"] = generation_inputs.pop(self.tokenizer.model_input_names[0])
        else:
            generation_inputs = inputs["input_ids"]

        generated_tokens = self.model.generate(
            **generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                print("use amp")
                with autocast():
                    forced_decoder_ids = [(1, 50261), (2, 50360), (3, 50364)]
                    outputs = model(**inputs,forced_decoder_ids=forced_decoder_ids)
#                    outputs = model(**inputs)
            else:
                print("no use amp")
                forced_decoder_ids = [(1, 50261), (2, 50360), (3, 50364)]
                outputs = model(**inputs,forced_decoder_ids=forced_decoder_ids)
#                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)
        
        
#    def compute_loss(self, model, inputs, return_outputs=False, eval=False):
#        """
#        How the loss is computed by Trainer. By default, all models return the loss in the first element.
#
#        Subclass and override for custom behavior.
#        """
#        if self.label_smoother is not None and "labels" in inputs:
#            labels = inputs.pop("labels")
#        else:
#            labels = None
#        
#        if eval:
#            forced_decoder_ids = [(1, 50261), (2, 50360), (3, 50364)]
#            outputs = model(**inputs,forced_decoder_ids=forced_decoder_ids)
#        else:
#            outputs = model(**inputs)
#            
#        # Save past state if it exists
#        # TODO: this needs to be fixed and made cleaner later.
#        if self.args.past_index >= 0:
#            self._past = outputs[self.args.past_index]
#
#        if labels is not None:
#            loss = self.label_smoother(outputs, labels)
#        else:
#            # We don't use .loss here since the model may return tuples instead of ModelOutput.
#            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#
#        return (loss, outputs) if return_outputs else loss

# only those hyperparameter which should be optimized
def make_seq2seq_training_kwargs(args):
    """
    Generate training arguments for Hugging Face Seq2SeqTrainer, based on Whisper defaults and Ray Tune configuration.

    Args:
        args (Namespace): Arguments from CLI or Ray Tune trial config.
    
    Returns:
        dict: Filtered and properly formatted training arguments.
    """

    #    # Load DeepSpeed config as a dictionary
    #    with open("/home/chrvt/asr-finetune-main/finetune_peft/deepspeed_config.json", "r") as f:
    #        ds_config = json.load(f)
    if args.peft:
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": False
            },
            "fp16": {
                "enabled": "auto"
            },
            "zero_force_ds_cpu_optimizer": False,  # This is crucial to avoid DeepSpeed's custom CPU optimizer
            "zero_allow_untested_optimizer": True,  # This allows using PyTorch optimizers with ZeRO
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            }

        #    # Update gradient_accumulation_steps to "auto"
        #    ds_config["gradient_accumulation_steps"] = "auto"

        #    # If there are other parameters that might conflict, set them to "auto" too
        #    # Common ones include:
        #    ds_config["train_batch_size"] = "auto"
        #    ds_config["train_micro_batch_size_per_gpu"] = "auto"
        # Important: Initialize DeepSpeed properly within Ray
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # Update DeepSpeed config with local_rank
        ds_config["local_rank"] = local_rank

    training_kwargs = {
        "output_dir": args.output_dir,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "len_train_set": args.len_train_set,
        "num_train_epochs": args.num_train_epochs,   # will be overriden if max_steps is given
        "max_steps": 0,  # will be determined in the train function
        "generation_max_length": args.generation_max_length,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "eval_delay": args.eval_delay, #int(args.max_steps/10),
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "seed": args.random_seed,
        "fp16": args.fp16,
        "model_type": args.model_type,
        "target_language": args.target_language,
        "return_timestamps": args.return_timestamps,
        "prefetch_batches": args.prefetch_batches,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "run_on_local_machine": args.run_on_local_machine,
        "wer_weight": args.wer_weight,
        # PEFT STUFF
        "peft": args.peft,
        "remove_unused_columns": False if args.peft else True,
        "label_names": ["labels"] if args.peft else None,
        "predict_with_generate": False if args.peft else True,
        "gradient_checkpointing": False if args.peft else True,
        "metric_for_best_model": "eval_loss" if args.peft else args.metric_to_optimize[0][0],
        "deepspeed": ds_config if args.peft else None,
        # "torch_empty_cache_steps": 1, # This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about 10% slower performance.
    }
    return training_kwargs


def train_whisper_peft_model(config, training_kwargs=None, data_collators=None, eval_sample_fraction=1.0, eval_names=None):
    """
    Main training function for Whisper model using PEFT (e.g. LoRA or AdaLoRA) and Ray Tune.

    This function is meant to be run inside a Ray Tune trial, performing the following:
        1. Load the Whisper model and tokenizer (optionally quantized).
        2. Apply LoRA or AdaLoRA PEFT configuration.
        3. Load model checkpoint (if resuming).
        4. Load datasets (Ray Data shards or fallback to HF utils).
        5. Configure and instantiate HuggingFace Trainer.
        6. Start training and report progress via Ray.

    Args:
        config (dict): Hyperparameter configuration for Ray Tune (e.g., alpha, rank).
        training_kwargs (dict): Trainer and model-specific kwargs such as output_dir, batch sizes, etc.
        data_collators (dict): Dictionary containing "train" and "val" collators for Ray Datasets.
        eval_sample_fraction (float): Fraction (0 < x <= 1.0) of eval dataset to use during evaluation.
    """
    # log_memory_usage("worker_start")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    starting_step = 0
    resume_from_checkpoint = None

    ###############################################################
    # 1. LOAD & PREPARE MODEL
    ###############################################################

    model, feature_extractor, tokenizer, processor = get_whisper_models(training_kwargs["model_type"],
                                          training_kwargs["target_language"],
                                          return_timestamps = training_kwargs["return_timestamps"],
                                          load_in_8bit = training_kwargs["peft"],
                                          local = training_kwargs["run_on_local_machine"]
                                          )

    # Prepare model for 8-bit LoRA-style training
    model = prepare_model_for_kbit_training(model)
    
    # register hook to ensure gradients can be computed for conv1
    def make_inputs_require_grad(module, input, output):
        output = output.to(input[0].device)
        output.requires_grad_(True)
        return output

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    # Compute training steps for AdaLoRA config
    training_kwargs["max_steps"] = (steps_per_epoch(training_kwargs["len_train_set"],
                                               config["per_device_train_batch_size"])
                                * training_kwargs["num_train_epochs"])
    
    # Setup AdaLoRA config
    lora_config = AdaLoraConfig(
        init_r=config["rank"],  # Initial rank (same as your old LoRA r)
        target_modules=["q_proj", "v_proj"],
        lora_alpha=config["alpha"],
        lora_dropout=0.05,
        tinit=int(training_kwargs["max_steps"] * 0.1),  # Step to start decreasing rank
        tfinal=int(training_kwargs["max_steps"] * 0.8),  # Step to finalize rank pruning
        deltaT=10,  # Rank update interval
        beta1=0.85,  # AdaLoRA-specific: Adam beta1
        beta2=0.95,  # AdaLoRA-specific: Adam beta2
        orth_reg_weight=0.8,  # Regularization term for low-rank adaptation
        total_step=training_kwargs["max_steps"]
    )
    
    # Apply LoRA adaptation
    model = get_peft_model(model, lora_config)
    model.to(f"cuda:{local_rank}")
    model.print_trainable_parameters()

    ###############################################################
    # 2. LOAD CHECKPOINT (IF AVAILABLE)
    ###############################################################
    try:
        checkpoint_dir = tune.get_checkpoint()
    except Exception as e:
        print(f"Error calling tune.get_checkpoint(): {e}", flush=True)
        checkpoint_dir = None

    if checkpoint_dir:
        trainer_state, starting_step, resume_from_checkpoint = load_checkpoints(checkpoint_dir)

    # Avoid caching in PEFT models
    model.config.use_cache = False

    ###############################################################
    # 3. LOAD DATASETS FROM RAY OR FALLBACK TO LOCAL
    ###############################################################
    ###############################################################
    # 4. LOAD DATASETS FROM RAY OR LOCAL FALLBACK
    ###############################################################
    if ray.train.get_dataset_shard("train") is not None:
        train_ds = ray.train.get_dataset_shard("train")
            # Collect validation shards dynamically
        eval_shards = {
            name.split("_")[-1]: ray.train.get_dataset_shard(name)
            for name in eval_names
        }
            
        print("Using pre-loaded Ray dataset shards.")
#        print("Using pre-loaded Ray dataset shards.")
#        train_ds = ray.train.get_dataset_shard("train")
#        
#        print("Getting eval dataset withing trainer.")
#        ray_datasets, data_collators_ = get_datasets_and_collators(data_collators["val"])
#        
#        eval_ds = ray_datasets["val"]
#        data_collators["val"] = data_collators_["val"]
#        eval_ds = ray.train.get_dataset_shard("val")
    else:
        print(f"Loading dataset within the trainer.")
        try:
            dataset_kwargs = data_collators
            ray_datasets, data_collators = get_datasets_and_collators(dataset_kwargs)
            train_ds = ray_datasets["train"]
            eval_ds = ray_datasets["val"]
            del dataset_kwargs
            print("Data successfully loaded within the trainer")
        except Exception as e:
            print(f"Could not load data: {e}")

    # Wrap Ray datasets for PyTorch
    train_ds_iterable = train_ds.iter_torch_batches(
        prefetch_batches = training_kwargs["prefetch_batches"],
        batch_size=config["per_device_train_batch_size"], collate_fn=data_collators["train"])

#    eval_ds_iterable = eval_ds.iter_torch_batches(
#        prefetch_batches = training_kwargs["prefetch_batches"],
#        batch_size=training_kwargs["per_device_eval_batch_size"], collate_fn=data_collators["val"])
    
    
    
#    if ray.train.get_dataset_shard("train") is not None:
#        print("Fetching dataset shards.")
#        train_ds = ray.train.get_dataset_shard("train")
#        eval_ds = ray.train.get_dataset_shard("val")
#    else:
#        print(f"Loading dataset within the trainer.")
#        try:
#            dataset_kwargs = data_collators
#            ray_datasets, data_collators = get_datasets_and_collators(dataset_kwargs)
#            train_ds = ray_datasets["train"]
#            eval_ds = ray_datasets["val"]
#            del dataset_kwargs
#            print("Data successfully loaded within the trainer")
#        except Exception as e:
#            print(f"Could not load data: {e}")


    ###############################################################
    # 4. CLEAN UP ARGS AND INITIALIZE TRAINER
    ###############################################################
    callbacks_ = [SavePeftModelCallback]
    compute_metrics = None
#    compute_metrics = get_metric_to_optimize("wer", tokenizer=tokenizer)

    # Remove unused entries from kwargs
    del training_kwargs["model_type"]
    wer_weight_ = training_kwargs["wer_weight"]
    del training_kwargs["wer_weight"]
    del training_kwargs["target_language"]
    del training_kwargs["return_timestamps"]
    del training_kwargs["run_on_local_machine"]
    del training_kwargs['len_train_set']  # we remove this as its not part of Seq2Seq
    del training_kwargs['num_train_epochs']
    prefetch_batches_ = training_kwargs['prefetch_batches']
    del training_kwargs['prefetch_batches']
    del training_kwargs['peft']
    del config['alpha']
    del config['rank']

    training_args = Seq2SeqTrainingArguments(
        eval_strategy="steps",
        save_strategy="steps",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        push_to_hub=False,
        do_eval=True,
        # Add these optimizations
        # dataloader_num_workers=4,  # Use more workers for dataloading
        dataloader_pin_memory=True,  # Speed up GPU transfers
        group_by_length=True,  # Group similar length sequences for efficiency
        **config,
        **training_kwargs,
    )

    ###############################################################
    # 5. INITIALIZE CUSTOM OR STANDARD TRAINER
    ###############################################################
    if eval_sample_fraction < 1:
        print(f"Evaluating on random fraction {eval_sample_fraction}%.")

        trainer = Seq2SeqTrainerEvalSamplingPeft(
                        processor=processor,
                        tokenizer=tokenizer,
                        eval_sample_fraction=eval_sample_fraction,
                        prefetch_batches=prefetch_batches_,
                        eval_dataset=eval_shards["1"],
                        eval_collator=data_collators["val"],
                        wer_weight = wer_weight_,
                        args=training_args,
                        model=model,
                        train_dataset=train_ds_iterable,
                        data_collator=data_collator_id,
                        compute_metrics=compute_metrics,
                        callbacks=callbacks_
                        # tokenizer=tokenizer,
                  )
        
        trainer.eval_shards = eval_shards

    else:
        trainer = Seq2SeqTrainer(
                    args=training_args,
                    model=model,
                    train_dataset=train_ds_iterable,
                    eval_dataset=eval_ds_iterable,
                    data_collator=data_collator_id,
                    compute_metrics=compute_metrics,
                    callbacks=callbacks_
                    # tokenizer=tokenizer,  we don't need this as we do the pre-processing before
                    )

    trainer.add_callback(RayTrainReportCallback())

    if starting_step > 0:
        trainer.add_callback(StepSyncCallback(starting_step))

    trainer = prepare_trainer(trainer)
    
    print("process_index:", trainer.args.process_index,
      "world_size:", trainer.args.world_size,
      "should_save:", trainer.args.should_save,
      "output_dir:", trainer.args.output_dir)
    print("save_strategy:", trainer.args.save_strategy, "save_steps:", trainer.args.save_steps)

    print("save_strategy:", trainer.args.save_strategy,
      "save_steps:", trainer.args.save_steps,
      "eval_steps:", trainer.args.eval_steps,
      "max_steps:", trainer.args.max_steps)
    print("global_step:", trainer.state.global_step)

    ###############################################################
    # 6. START TRAINING
    ###############################################################
    if resume_from_checkpoint:
        print(f"Resuming from Checkpoint {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()


def train_whisper_model(config, training_kwargs=None, data_collators=None, eval_sample_fraction=1.0, eval_names=None):
    """
    Main training function for Whisper model without PEFT (LoRA/AdaLoRA), using Hugging Face + Ray Tune.

    This function is typically executed as part of a Ray Tune trial. It supports training Whisper models using
    Hugging Face's Seq2SeqTrainer on Ray datasets, with optional checkpointing and partial evaluation.

    Steps:
        1. Load Whisper model and tokenizer.
        2. Optionally resume training from a Ray Tune checkpoint.
        3. Load training and evaluation datasets (Ray or local fallback).
        4. Build Seq2SeqTrainingArguments and Trainer.
        5. Train the model and log metrics via Ray.

    Args:
        config (dict): Hyperparameter configuration dictionary (e.g., batch size, learning rate).
        training_kwargs (dict): Training argument dictionary (e.g., steps, output_dir, warmup).
        data_collators (dict): Dictionary containing "train" and "val" collator functions.
        eval_sample_fraction (float): Optional fraction (0 < x ≤ 1) of eval set to use (for faster feedback).
    """
    # log_memory_usage("worker_start")
    ###############################################################
    ####LOAD MODELS############# ##################################
    starting_step = 0
    resume_from_checkpoint = None
    print("lets start the training")

    ###############################################################
    # 1. LOAD MODEL AND TOKENIZER
    ###############################################################
    model, feature_extractor, tokenizer, processor = get_whisper_models(training_kwargs["model_type"],
                                          training_kwargs["target_language"],
                                          return_timestamps = training_kwargs["return_timestamps"],
                                          load_in_8bit = training_kwargs["peft"],
                                          local = training_kwargs["run_on_local_machine"]
                                          )

    print("model loaded")

    ###############################################################
    # 2. LOAD CHECKPOINT IF AVAILABLE
    ###############################################################
    try:
        checkpoint_dir = tune.get_checkpoint()
    except Exception as e:
        print(f"Error calling tune.get_checkpoint(): {e}", flush=True)
        checkpoint_dir = None

    if checkpoint_dir:
        trainer_state, starting_step, resume_from_checkpoint = load_checkpoints(checkpoint_dir)

    ###############################################################
    # 3. DEFINE METRICS
    ###############################################################
    print("Initializing evaluation metric: WER")
    compute_metrics = get_metric_to_optimize("wer", tokenizer=tokenizer)

    ###############################################################
    # 4. LOAD DATASETS FROM RAY OR LOCAL FALLBACK
    ###############################################################
    if ray.train.get_dataset_shard("train") is not None:
        train_ds = ray.train.get_dataset_shard("train")
            # Collect validation shards dynamically
        eval_shards = {
            name.split("_")[-1]: ray.train.get_dataset_shard(name)
            for name in eval_names
        }
            
        print("Using pre-loaded Ray dataset shards.")
#        ctx = ray.train.get_context()
#        eval_shards = {
#            name.split("_")[-1]: ray.train.get_dataset_shard(name)
#            for name in ctx.get_dataset_config().keys()
#            if name.startswith("val_")
#        }
#        eval_ds_1 = ray.train.get_dataset_shard("val_1")
#        eval_ds_2 = ray.train.get_dataset_shard("val_2")
#        eval_ds_3 = ray.train.get_dataset_shard("val_3")
#        eval_shards = {"1": eval_ds_1, "2": eval_ds_2, "3": eval_ds_3}
        
        
#        print("Getting eval dataset within trainer.")
#        ray_datasets, data_collators_ = get_datasets_and_collators(data_collators["val"])
#        
#        eval_ds = ray_datasets["val"]
#        print("eval ds", eval_ds.count())
#        data_collators["val"] = data_collators_["val"]
#        eval_ds = ray.train.get_dataset_shard("val")
    else:
        print(f"Loading dataset within the trainer.")
        try:
            dataset_kwargs = data_collators
            ray_datasets, data_collators = get_datasets_and_collators(dataset_kwargs)
            train_ds = ray_datasets["train"]
            eval_ds = ray_datasets["val"]
            del dataset_kwargs
            print("Data successfully loaded within the trainer")
        except Exception as e:
            print(f"Could not load data: {e}")

    # Wrap Ray datasets for PyTorch
    train_ds_iterable = train_ds.iter_torch_batches(
        prefetch_batches = training_kwargs["prefetch_batches"],
        batch_size=config["per_device_train_batch_size"], collate_fn=data_collators["train"])

#    eval_ds_iterable = eval_ds.iter_torch_batches(
#        prefetch_batches = training_kwargs["prefetch_batches"],
#        batch_size=training_kwargs["per_device_eval_batch_size"], collate_fn=data_collators["val"])

    ###############################################################
    # 5. CLEAN UP TRAINING KWARGS AND CONFIG
    ###############################################################
    # Compute total max steps based on dataset size and epochs
    training_kwargs["max_steps"] = steps_per_epoch(training_kwargs["len_train_set"],config["per_device_train_batch_size"]) * training_kwargs["num_train_epochs"]

    del training_kwargs["model_type"]
    del training_kwargs["target_language"]
    del training_kwargs["return_timestamps"]
    del training_kwargs["run_on_local_machine"]
    del training_kwargs['len_train_set']        # we remove this as its not part of Seq2Seq
    del training_kwargs['num_train_epochs']
    prefetch_batches_ = training_kwargs['prefetch_batches']
    wer_weight_ = training_kwargs["wer_weight"]
    del training_kwargs['prefetch_batches']
    del training_kwargs['wer_weight']
    del training_kwargs['peft']

    ###############################################################
    # 6. CREATE TRAINING ARGUMENTS
    ###############################################################
    training_args = Seq2SeqTrainingArguments(
        eval_strategy="steps",
        save_strategy = "steps",
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        push_to_hub=False,
        do_eval=True,
        # Add these optimizations
        dataloader_pin_memory=True,  # Speed up GPU transfers
        group_by_length=True,  # Group similar length sequences for efficiency
        **config,
        **training_kwargs,
    )

    ###############################################################
    # 7. INITIALIZE THE TRAINER (WITH OR WITHOUT SUBSAMPLING)
    ###############################################################
    if eval_sample_fraction < 1:
        print(f"Evaluating on random fraction {eval_sample_fraction}%.")

        trainer = Seq2SeqTrainerEvalSampling(
                        eval_sample_fraction=eval_sample_fraction,
                        prefetch_batches=prefetch_batches_,
                        eval_dataset=eval_shards["1"], #eval_ds_iterable, eval_ds #,
                        eval_collator=data_collators["val"],
                        wer_weight = wer_weight_,
                        args=training_args,
                        model=model,
                        train_dataset=train_ds_iterable,
                        data_collator=data_collator_id,
                        compute_metrics=compute_metrics,
                        callbacks=None
                        # tokenizer=tokenizer,
                  )
        trainer.eval_shards = eval_shards
        trainer.eval_parquet_dir = "/scratch/usr/bemchrvt/data/eg_dataset_complete_v3_sharded/val_parquet"

    else:
        # Should not be Seq2SeqTrainer!
        trainer = Seq2SeqTrainer(
                    args=training_args,
                    model=model,
                    train_dataset=train_ds_iterable,
                    eval_dataset=eval_ds_iterable,
                    data_collator=data_collator_id,
                    compute_metrics=compute_metrics,
                    callbacks=None
                    # tokenizer=tokenizer,  we don't need this as we do the pre-processing before
                    )

    ###############################################################
    # 8. SETUP RAY TUNE CALLBACKS
    ###############################################################
    trainer.add_callback(RayTrainReportCallback())

    if starting_step > 0:
        trainer.add_callback(StepSyncCallback(starting_step))
    
    trainer = prepare_trainer(trainer)
#    trainer = prepare_trainer_custom(trainer)
    
    print("process_index:", trainer.args.process_index,
      "world_size:", trainer.args.world_size,
      "should_save:", trainer.args.should_save,
      "output_dir:", trainer.args.output_dir)
    print("save_strategy:", trainer.args.save_strategy, "save_steps:", trainer.args.save_steps)

    print("save_strategy:", trainer.args.save_strategy,
      "save_steps:", trainer.args.save_steps,
      "eval_steps:", trainer.args.eval_steps,
      "max_steps:", trainer.args.max_steps)
    print("global_step:", trainer.state.global_step)


    ###############################################################
    # 9. BEGIN TRAINING
    ###############################################################
    if resume_from_checkpoint:
        print(f"Resuming from Checkpoint {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()



# def collate_fn(batch):
#     input_features_list = []
#     labels_list = []
#
#     input_features_batch = torch.stack([torch.from_numpy(x) for x in batch["input_features"]])
#     labels_batch = torch.stack([torch.from_numpy(x) for x in batch["labels"]])
#
#     return {
#         "input_features": input_features_batch, #input_features_batch,
#         "labels": labels_batch# labels_batch
#     }
#
#
# def train_single_whisper_model(config):
#     """Main training function for one specific Hyper Parameter configuration.
#
#     Each ray-worker (each hyperparameter configuration) executes this function. The training data needs to be in a ray
#     training iter object which requires a data_collator. However, since we already collated the data in the pre-
#     processing (see DataCollatorSpeechSeq2SeqWithPadding), we simply use the identity as collator.
#
#     The reporting and logging is done by ray tune automatically. To adopt this function for another fine-tuning project,
#     follow similar steps:
#     1. define the required models
#     2. define the evaluation metric
#     3. load the data into ray tune iter
#     4. define trainer Instance (here: Seq2SeqTrainer)
#     5. End with:
#         trainer.add_callback(RayTrainReportCallback())
#         trainer = prepare_trainer(trainer)
#         trainer.train()
#
#     Requires:
#        get_models (function): A function loading the necessary models for training and evaluation
#        compute_metrics (function): A function which computes the metrics (WER in our case)
#
#     Args:
#        config (tune.TuneConfig): Config File with Hyperparameter instances (automaticall generated by ray)
#        training_kwargs (dict): Dictionary of training arguments for the Hugging Face Seq2SeqTrainer
#     """
#     # get models
#     training_kwargs = config["training_kwargs"]
#     ###############################################################
#     ####LOAD MODELS############# ##################################
#     model, feature_extractor, tokenizer, processor = get_whisper_models(training_kwargs["model_type"],
#                                           training_kwargs["target_language"],
#                                           return_timestamps = training_kwargs["return_timestamps"],
#                                           load_in_8bit = training_kwargs["peft"],
#                                           local = training_kwargs["run_on_local_machine"]
#                                           )
#
#     print("model loaded")
#     print("peft :", training_kwargs["peft"])
#     if training_kwargs["peft"]:
#         model = prepare_model_for_kbit_training(model)  # , output_embedding_layer_name="proj_out")
#
#         def make_inputs_require_grad(module, input, output):
#             output.requires_grad_(True)
#
#         model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
#
#         lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
#                                  bias="none")
#         model = get_peft_model(model, lora_config)
#         model.print_trainable_parameters()
#
#     del training_kwargs["model_type"]
#     del training_kwargs["target_language"]
#     del training_kwargs["return_timestamps"]
#     del training_kwargs["run_on_local_machine"]
#
#     print("loading the metric")
#     # Define metric for evaluation
#     try:
#         print("trying")
#         metric = evaluate.load("/home/bexkompi/wer.py")
#     except Exception as e:
#         print(f"Evaluate.load failed: {e}", flush=True)
#
#     #    def compute_metrics(pred):
#     #        return {"wer": 1.0}
#
#     def compute_metrics(pred):
#         """Performance Metric calculator, here: Word Error Rate (WER)
#
#         Note: 'Normalizes' the strings before calculating the WER.
#
#         Requires:
#             Initialized Tokenizer for decoded the predicitions and labels into human language
#             WER metric from the evaluate package
#         Args:
#             pred (dict): a dictionary with keys "predictions" and "label_ids"
#         Returns:
#             (dict): A dictionary with key "wer" and the corresponding value
#         """
#         pred_ids = pred.predictions
#         label_ids = pred.label_ids
#
#         # replace -100 with the pad_token_id
#         label_ids[label_ids == -100] = tokenizer.pad_token_id
#
#         # we do not want to group tokens when computing the metrics
#         pred_str = normalize(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
#         label_str = normalize(tokenizer.batch_decode(label_ids, skip_special_tokens=True))
#         wer = 100 * metric.compute(predictions=pred_str, references=label_str)
#
#         return {"wer": wer}
#
#     print("getting the data shards")
#
#     path_to_data = r"/scratch/usr/bemchrvt/data/eg_dataset_complete_v3_sharded/"
#     train_h5_path = os.path.join(path_to_data, "train_parquet")
#     # val_h5_path = os.path.join(path_to_data, "val_parquet")
#
#     #
#     #
#     #
#     train_ds = ray.data.read_parquet(
#         train_h5_path,
#     )  # Limit resources)
#
#     # eval_ds = ray.train.get_dataset_shard("eval")
#     val_h5_path = os.path.join(path_to_data, f"{dataset_name}_val.h5")
#     eval_ds = create_ray_indexloader(val_h5_path)
#     #    ray.data.read_parquet(
#     #                    val_h5_path,
#     #                    )
#
#     # this is a hack - as train_ds from Ray requires the data_collotor, so does Seq2SeqTrainer from HF
#     # but collating twice does not make sense, therefore we introduce the indentity collator
#     def data_collator_id(input):
#         """Identity Collator"""
#         return input
#
#     #    train_ds_ = ray.train.get_dataset_shard("train")
#     #    eval_ds = ray.train.get_dataset_shard("eval")
#
#     print("got it!")
#
#     #    log_memory_usage("after_data_load")
#     #
#
#     ########### load checkpoint###########
#     try:
#         checkpoint_dir = tune.get_checkpoint()
#     except Exception as e:
#         print(f"Error calling tune.get_checkpoint(): {e}", flush=True)
#         checkpoint_dir = None
#     #        ray.train.report(metrics={"eval_loss": 0})
#
#     print('-----------------------------------------------------------------')
#     print("CHECKPOINT: ", checkpoint_dir, flush=True)
#     #    ray.train.get_checkpoint().to_directory()
#     ##    ray.train.get_checkpoint().to_directory()
#     resume_from_checkpoint = None
#     starting_step = 0
#     if checkpoint_dir:
#         try:
#             # Load trainer state to get current step
#             trainer_state_path = os.path.join(checkpoint_dir.path, "checkpoint/trainer_state.json")
#             if os.path.exists(trainer_state_path):
#                 with open(trainer_state_path, 'r') as f:
#                     trainer_state = json.load(f)
#                     starting_step = trainer_state["global_step"]
#                     #                    train_ds = train_ds_
#                     #                    .filter(lambda row: row["idx"] >= starting_step)
#                     print(f"Will resume from step {starting_step}", flush=True)
#                     # Set this to tell the trainer where to load from
#                     resume_from_checkpoint = os.path.join(checkpoint_dir.path, "checkpoint")
#             else:
#                 print(f"Path does not exists: {trainer_state_path}", flush=True)
#
#         except Exception as e:
#             print(f"Error synchronizing iterator state: {e}", flush=True)
#     #    else:
#     #        train_ds = train_ds_
#
#     train_ds_iterable = train_ds.iter_torch_batches(
#         prefetch_batches=training_kwargs["prefetch_batches"],
#         batch_size=config["per_device_train_batch_size"], collate_fn=collate_fn)
#
#     eval_ds_iterable = eval_ds.iter_torch_batches(
#         prefetch_batches=training_kwargs["prefetch_batches"],
#         batch_size=config["per_device_train_batch_size"],
#         collate_fn= SimpleStreamingCollator(val_h5_path,
#                                             feature_extractor,
#                                             tokenizer,
#                                             num_workers=2))
#
#     # from utils import RayIterableDataset
#
#     # train_ds_iterable = RayIterableDataset(train_ds)
#     # eval_ds_iterable = RayIterableDataset(eval_ds)
#
#     # eval_ds_iterable = iter(eval_ds,batch_size=config["per_device_train_batch_size"],
#     #                          shuffle = True)
#
#     training_kwargs["max_steps"] = steps_per_epoch(training_kwargs["len_train_set"],
#                                                    config["per_device_train_batch_size"]) * training_kwargs[
#                                        "num_train_epochs"]
#
#     if training_kwargs["peft"]:
#         model.config.use_cache = False
#         callbacks_ = [SavePeftModelCallback]
#     else:
#         callbacks_ = None
#
#     del training_kwargs['len_train_set']  # we remove this as its not part of Seq2Seq
#     del training_kwargs['num_train_epochs']
#     del training_kwargs['prefetch_batches']
#     del training_kwargs['peft']
#     del config["training_kwargs"]
#     training_args = Seq2SeqTrainingArguments(
#         gradient_checkpointing=True,
#         eval_strategy="steps",
#         save_strategy="steps",
#         predict_with_generate=True,  # True if not training_kwargs["peft"] else False,
#         report_to=["tensorboard"],
#         load_best_model_at_end=True,
#         metric_for_best_model="wer",
#         greater_is_better=False,
#         push_to_hub=False,
#         do_eval=True,
#         # Add these optimizations
#         # dataloader_num_workers=4,  # Use more workers for dataloading
#         dataloader_pin_memory=True,  # Speed up GPU transfers
#         group_by_length=True,  # Group similar length sequences for efficiency
#         **config,
#         **training_kwargs,
#     )
#
#     trainer = Seq2SeqTrainer(
#         args=training_args,
#         model=model,
#         train_dataset=train_ds_iterable,
#         eval_dataset=eval_ds_iterable,
#         data_collator=data_collator_id,
#         compute_metrics=compute_metrics,
#         callbacks=callbacks_
#         # tokenizer=tokenizer,  we don't need this as we do the pre-processing before
#     )
#
#     #     Create a custom callback to handle step synchronization
#     class StepSyncCallback(TrainerCallback):
#         def __init__(self, starting_step):
#             self.starting_step = starting_step
#             self.has_synced = False
#
#         def on_train_begin(self, args, state, control, **kwargs):
#             if self.starting_step > 0 and not self.has_synced:
#                 print(f"Synchronizing step counter to {self.starting_step}")
#                 # Update the trainer's step counter
#                 state.global_step = self.starting_step
#                 self.has_synced = True
#
#     trainer.add_callback(RayTrainReportCallback())
#
#
#     if starting_step > 0:
#         trainer.add_callback(StepSyncCallback(starting_step))
#
#     trainer = prepare_trainer(trainer)
#     #    trainer.train()
#     if resume_from_checkpoint:
#         print(f"Resuming from Checkpoint {resume_from_checkpoint}")
#         trainer.train(resume_from_checkpoint=resume_from_checkpoint)
#     else:
#         trainer.train()

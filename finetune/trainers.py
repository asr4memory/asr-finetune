"""Collection of different trainers and corresponding training Arguments.
"""
import os

from utils import  steps_per_epoch, normalize, log_memory_usage
import evaluate

# laod transformers
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer


# For hyperparameter optimization
import ray
import ray.train
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback

# get models
from models import get_whisper_models as get_models
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, LoraModel, LoraConfig, get_peft_model
import json

# only those hyperparameter which should be optimized
def make_seq2seq_training_kwargs(args):
    """Training Arguments Filter for the train_model function.

    This is not stricly required as we can also pass args into the train_model. However, it serves as an overview of the
    relevant training arguments for the Seq2Seq Trainer:
    https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments

    In addition to the provided args, we set some default values based on the original Whisper Hyperparameters:
    https://cdn.openai.com/papers/whisper.pdf, in particular the beta1 and beta 2 values of the AdamW optimizer (which
    is the default optimzier) differ to the default parameters.

    Args:
        args (dict): dictionary of keyboard arguments
    Returns:
       training_kwargs (dict): dictionary of relevant training arguments
    """

#    # Load DeepSpeed config as a dictionary
#    with open("/home/chrvt/asr-finetune-main/finetune_peft/deepspeed_config.json", "r") as f:
#        ds_config = json.load(f)

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
#
#    # If there are other parameters that might conflict, set them to "auto" too
#    # Common ones include:
#    ds_config["train_batch_size"] = "auto"
#    ds_config["train_micro_batch_size_per_gpu"] = "auto"
    # Important: Initialize DeepSpeed properly within Ray
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # Update DeepSpeed config with local_rank
    ds_config["local_rank"] = local_rank

    training_kwargs = {
        "output_dir": args.output_dir,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "len_train_set": args.len_train_set,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": 0,
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
        "dataloader_num_workers": args.dataloader_num_workers,
        "run_on_local_machine": args.run_on_local_machine,
        # PEFT STUFF
        "peft": args.peft,
        "remove_unused_columns": False if args.peft else True,
        "label_names": ["labels"] if args.peft else None,
        "predict_with_generate": False if args.peft else True,
        "gradient_checkpointing": False if args.peft else True,
        "metric_for_best_model": "eval_loss" if args.peft else "wer",
        "deepspeed": ds_config if args.peft else None,
        "local_rank": local_rank,
        # "torch_empty_cache_steps": 1, # This can help avoid CUDA out-of-memory errors by lowering peak VRAM usage at a cost of about 10% slower performance.
    }
    return training_kwargs

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def train_whisper_model(config, training_kwargs=None, data_collators=None):
    """Main training function for one specific Hyper Parameter configuration.

    Each ray-worker (each hyperparameter configuration) executes this function. The training data needs to be in a ray
    training iter object which requires a data_collator. However, since we already collated the data in the pre-
    processing (see DataCollatorSpeechSeq2SeqWithPadding), we simply use the identity as collator.

    The reporting and logging is done by ray tune automatically. To adopt this function for another fine-tuning project,
    follow similar steps:
    1. define the required models
    2. define the evaluation metric
    3. load the data into ray tune iter
    4. define trainer Instance (here: Seq2SeqTrainer)
    5. End with:
        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)
        trainer.train()

    Requires:
       get_models (function): A function loading the necessary models for training and evaluation
       compute_metrics (function): A function which computes the metrics (WER in our case)

    Args:
       config (tune.TuneConfig): Config File with Hyperparameter instances (automaticall generated by ray)
       training_kwargs (dict): Dictionary of training arguments for the Hugging Face Seq2SeqTrainer
    """
    log_memory_usage("worker_start")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # get models
    if training_kwargs["run_on_local_machine"]:
        from models import get_whisper_models_local
        model, feature_extractor, tokenizer, processor = get_whisper_models_local(training_kwargs["model_type"],
                                                                training_kwargs["target_language"],
                                                                return_timestamps=training_kwargs["return_timestamps"],
                                                                     load_in_8bit=training_kwargs["peft"]
                                                                                  )
    else:
        model, feature_extractor, tokenizer, processor = get_models(training_kwargs["model_type"],
                                                                training_kwargs["target_language"],
                                                                return_timestamps=training_kwargs["return_timestamps"],
                                                                     load_in_8bit=training_kwargs["peft"])

    if training_kwargs["peft"]:
        model = prepare_model_for_kbit_training(model)
        #, output_embedding_layer_name="proj_out") not needed:  https://github.com/Vaibhavs10/fast-whisper-finetuning/issues/6

        def make_inputs_require_grad(module, input, output):
            output = output.to(input[0].device)
            output.requires_grad_(True)
            return output


        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
        # r=32, lora_alpha=64
        lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.05,
                                 bias="none")
        #, task_type="SEQ_2_SEQ_LM")
        model = get_peft_model(model, lora_config)
        model.to(f"cuda:{local_rank}")
        model.print_trainable_parameters()


    del training_kwargs["model_type"]
    del training_kwargs["target_language"]
    del training_kwargs["return_timestamps"]
    del training_kwargs["run_on_local_machine"]
    # Define metric for evaluation
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        """Performance Metric calculator, here: Word Error Rate (WER)

        Note: 'Normalizes' the strings before calculating the WER.

        Requires:
            Initialized Tokenizer for decoded the predicitions and labels into human language
            WER metric from the evaluate package
        Args:
            pred (dict): a dictionary with keys "predictions" and "label_ids"
        Returns:
            (dict): A dictionary with key "wer" and the corresponding value
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # # Check and fix nesting structure if needed
        # if isinstance(pred_ids[0], list):
        #     # If predictions are nested lists, take the first item
        #     # This is common when using beam search or similar techniques
        #     pred_ids = [p[0] if isinstance(p, list) else p for p in pred_ids]
        #
        # # Do the same for labels if needed
        # if isinstance(label_ids[0], list):
        #     label_ids = [l[0] if isinstance(l, list) else l for l in label_ids]

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = normalize(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
        label_str = normalize(tokenizer.batch_decode(label_ids, skip_special_tokens=True))
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    log_memory_usage("before_data_load")
    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("eval")

    # this is a hack - as train_ds from Ray requires the data_collotor, so does Seq2SeqTrainer from HF
    # but collating twice does not make sense, therefore we introduce the indentity collator
    def data_collator_id(batch):
        return {k: v.to(f"cuda:{local_rank}") if torch.is_tensor(v) else v for k, v in batch.items()}


    # the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.

    train_ds_iterable = train_ds.iter_torch_batches(
        prefetch_batches = training_kwargs["prefetch_batches"],
        batch_size=config["per_device_train_batch_size"], collate_fn=data_collators["training"])

    eval_ds_iterable = eval_ds.iter_torch_batches(
        prefetch_batches = training_kwargs["prefetch_batches"],
        batch_size=config["per_device_train_batch_size"], collate_fn=data_collators["validation"])

    log_memory_usage("after_data_load")

    training_kwargs["max_steps"] = steps_per_epoch(training_kwargs["len_train_set"],config["per_device_train_batch_size"]) * training_kwargs["num_train_epochs"]

    if training_kwargs["peft"]:
        model.config.use_cache = False
        callbacks_ = [SavePeftModelCallback]
        compute_metrics = None
    else:
        callbacks_ = None

    del training_kwargs['len_train_set']        # we remove this as its not part of Seq2Seq
    del training_kwargs['num_train_epochs']
    del training_kwargs['prefetch_batches']
    del training_kwargs['peft']

    training_args = Seq2SeqTrainingArguments(
        eval_strategy="steps",
        save_strategy = "steps",
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

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
        data_collator=data_collator_id,
        compute_metrics=compute_metrics,
        callbacks = callbacks_
        # tokenizer=tokenizer,  we don't need this as we do the pre-processing before
    )


    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()

    # processor.save_pretrained(training_args.output_dir) # TODO: is this really necessary?


"""Main Script for evaluating a fine-tuned or default Whisper Hugging Face model

A high-level overview of the script:
    1. We pre-process the data
    2. We call eval_model
    3. We load the model
    4. Iterate through test set
    5. store results in a dictionary

Functions are:
    parse_args...argument parser
    select_device...selects the device to evaluate on
    get_models...loads model, tokenizer, feature extractor
    eval_model...main evaluation function

For a description of what each function is doing, we refer to the docstrings of the very function.
"""
import sys
from pathlib import Path

# Make src/ importable so sibling packages (utils, models, data_and_collator, ...)
# resolve whether this file is run via ``python -m evaluation.evaluate`` or directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pprint
import os
import json

from utils import save_file, normalize

import evaluate
import safetensors
try:
    from transformers.models.whisper.convert_openai_to_hf import make_linear_from_emb
except ImportError:
    import torch.nn as _nn
    def make_linear_from_emb(emb):
        lin = _nn.Linear(emb.embedding_dim, emb.num_embeddings, bias=False)
        lin.weight = emb.weight
        return lin

import torch
from data_and_collator.streaming import create_ray_indexloader, SimpleStreamingCollator


def resolve_adapter_dir(path):
    """Return the directory that actually contains adapter_config.json.

    Training writes the real DoRA weights into <checkpoint-N>/adapter_model/
    while leaving a ZeRO-3 placeholder adapter_model.safetensors at the
    checkpoint root. Pointing PeftModel.from_pretrained at the root silently
    loads the empty placeholder -> base-only model -> wrong WER. Descend one
    level if the supplied path holds an `adapter_model/` subdir with a real
    adapter_config.json inside.
    """
    if not path:
        return path
    direct_cfg = os.path.join(path, "adapter_config.json")
    if os.path.isfile(direct_cfg):
        return path
    nested = os.path.join(path, "adapter_model")
    if os.path.isfile(os.path.join(nested, "adapter_config.json")):
        print(f"[adapter] resolved {path} -> {nested}")
        return nested
    return path

# laod models
from transformers import set_seed
# for loading from checkpoint

# For code organization and reporting
import configargparse
import logging

# For Dataset preparation

# get models
from models.whisper_models import get_whisper_models_from_dir, get_whisper_models_from_hub

logger = logging.getLogger(__name__)

# We define all the different parameters for the training, model, evaluation etc.
# Whisper model type choices: https://huggingface.co/models?search=openai/whisper
# openai/whisper-large-v3 sofa

# https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server

def parse_args():
    """ Parses command line arguments for the training.

    In particular:
            model_type...Whisper model type choices: https://huggingface.co/models?search=openai/whisper
            model_ckpt_path...path to model checkpoint to evaluate
    Important:
            test_split and random_seed should match the training setting (otherwise train set will be part of test set)
    """
    parser = configargparse.ArgumentParser()

    # Plotting

    # Training settings for Seq2SeqTrainingArguments
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--output_tag", type=str,
                        default="whisper-tiny-de",
                        help="Base directory where model is save.")
    # model settings
    parser.add_argument("--model_type", type=str, default="openai/whisper-tiny", help="Model to optimize")
    parser.add_argument("--target_language", type=str, default="german", help="Target Language")
    # parser.add_argument("--load_model", action="store_true", help="Load model from model_ckpt_path")   # TODO: enable restoring: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.restore.html#ray.tune.Tuner.restore
    parser.add_argument("--model_ckpt_path", type=str, default=None, help="Path to the fine-tuned checkpoint / adapter directory to evaluate (usually set in the .config file).")
    parser.add_argument("--return_timestamps", action="store_true", help="Return Timestemps mode for model")

    parser.add_argument("--num_workers", type=int, default=1, help="Number of trials that can run at the same time")
    parser.add_argument("--cpus_per_trial", type=int, default=1, help="Number of CPUs per Ray actor")
    parser.add_argument("--gpus_per_trial", type=float, default=0, help="Number of GPUs per Ray actor")
    parser.add_argument("--use_gpu", action="store_true", help="If using GPU for the finetuning")
    # parser.add_argument("--device", type=str, default="cpu",
    #                     help="Path to audio batch-prepared audio files.")

    # Dataset settings
    parser.add_argument("--test_split", type=float, default=0.2, help="Percentage of test data.")
    parser.add_argument("--h5", action="store_true", help="If data is in .h5 format")
    parser.add_argument("--fp16", action="store_true", default=False, help="Training with floating point 16 ")
    parser.add_argument("--peft", action="store_true", help="Whether or not to do Parameter Efficient Training")
    parser.add_argument("--prefetch_batches", type=int, default=1,
                        help="How many batches to prefetch data? Keep in mind: is using VRAM.")


    # Other settings
    parser.add_argument("--run_on_local_machine", action="store_true", help="Store true if training is on local machine.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base directory where outputs are saved.")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--path_to_data",
                        type=str,
                        default="../data/datasets/fzh-wde0459_03_03",
                        help="Path to audio batch-prepared audio files if in debug mode. Otherwise: all data in datasets are loaded")
    parser.add_argument("--dataset_name", type=str, default="eg_dataset_subset_1000.h5",
                        help="Name of dataset")
    parser.add_argument("--random_seed", type=int, default=1337, help="Random Seed for reproducibility")
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--push_to_hub", action="store_true", help="Push best model to Hugging Face")
    parser.add_argument("--resume_evaluation",
                        action="store_true",
                        help="Resume evaluation from the last checkpoint if available")
    parser.add_argument("--max_eval_batches", type=int, default=None,
                        help="If set, stop after this many batches. Smoke-test knob.")

    args = parser.parse_args()
    return args

# Function to select the device
def select_device():
    """Selects the device to evaluate on.

    Returns:
         torch.device("mps")/torch.device("cuda")/torch.device("cpu")
    :return:
    """
    # Check for CUDA availability with MPS support
    if torch.cuda.is_available():
        # Check if MPS is enabled and supported
        if torch.backends.mps.is_available():
            logger.info("Using CUDA MPS device")
            return torch.device("mps")
        else:
            logger.info("Using standard CUDA device")
            return torch.device("cuda")
    else:
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")

# def eval_model(args, data_collators=None):
#     """Evaluation of model (pre-oder fine-tuned) on eval_dict.
#
#     A model is loaded either from a checkpoint (args.model_ckpt_path) or the default HF model.
#     compute_metric stores the original and prediction text.
#
#     Note:
#         model.generation_config.return_timestamps = True had to be set to get same results as in asr-evaluate
#         This config seems to decrease halluzination
#
#     Todo:
#         * If model is loaded from a checkpoint, use the training-config file to load model settings and configs
#
#     Requires:
#        get_models (function): A function loading the necessary models for training and evaluation
#        compute_metrics (function): A function which computes the metrics (WER in our case)
#
#     Args:
#        args (dict): Argument parser. In particular, which model is used and if it is loaded from checkpoint.
#        eval_dict (DatasetDict): Dataset dictionary to evaluate the model on
#        data_collator (DataCollatorSpeechSeq2SeqWithPadding): Collator for data preparation
#
#     """
#     device = select_device()
#     logger.info('Device %s detected.', device)
#
#     # get models
#     model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language,
#                                                                 return_timestamps=args.return_timestamps,
#                                                                 load_in_8bit=args.peft)
#
#
#
#     # Load the state dictionary from the checkpoint
#     if len(args.model_ckpt_path)>0:
#
#         if args.peft:
#         #     from peft import PeftModel, PeftConfig
#         #     model = PeftModel.from_pretrained(model, args.model_ckpt_path)
#         #     model.config.use_cache = True
#         # else:
#             state_dict = safetensors.torch.load_file(os.path.join(args.model_ckpt_path, 'model.safetensors'))
#             # Fix missing proj_out weights: https://github.com/openai/whisper/discussions/2302
#             model.load_state_dict(state_dict, strict=False)
#             model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
#             model.config.use_cache = True
#             logger.info('Whisper model from checkpoint %s loaded.', args.model_ckpt_path)
#     else:
#         logger.info('Whisper model %s loaded.', args.model_type)
#
#     # Define metric for evaluation
#     metric = evaluate.load("wer")
#     def compute_metrics(pred_ids,label_ids):
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
#         label_ids[label_ids == -100] = tokenizer.pad_token_id
#
#         # we do not want to group tokens when computing the metrics
#         pred_str = normalize(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
#         label_str = normalize(tokenizer.batch_decode(label_ids, skip_special_tokens=True))
#
#         wer = 100 * metric.compute(predictions=pred_str, references=label_str)
#
#         logger.info("Original: %s", label_str)
#         logger.info("Model prediction: %s", pred_str)
#         # pred_str = model.tokenizer._normalize(pred_str)
#         # label_str = model.tokenizer._normalize(label_str)
#         return {"wer": wer, "original": label_str, "predictions":pred_str}
#
#     model.eval().to(device)        # eval mode for model pushed to device
#     batch_size =  args.eval_batch_size # 1 is more convenient for downstream processing of results as 1 data = 1 row
#
#     """Prepare test set with data_collator"""
#     test_ds = ray.train.get_dataset_shard("test")
#     test_ds_iterable = test_ds.iter_torch_batches(
#         batch_size=batch_size, collate_fn=data_collators["testing"]
#     )
#     eval_results = {}
#     count = -1
#     wer_average = 0
#     """Loop through test set"""
#     with torch.no_grad():
#         for batch in test_ds_iterable:
#             count += 1
#             label_ids = batch["labels"]
#             pred_ids = model.generate(batch["input_features"].to(device))
#             outputs = compute_metrics(pred_ids,label_ids)
#
#             eval_results[str(count)] = outputs
#             wer_average += outputs["wer"]
#
#             if count % 50:
#                 logger.info('WER for step %s...',count)
#                 logger.info('...%s',outputs["wer"])
#
#     logger.info("WER average on Test Set %s", wer_average/(count+1))
#
#     save_file(eval_results, args.output_dir, mode = 'json', file_tag='eval')

def eval_model(args = None, data_collators = None, test_set = None):
    """Evaluation of model (pre-oder fine-tuned) on eval_dict.

    A model is loaded either from a checkpoint (args.model_ckpt_path) or the default HF model.
    compute_metric stores the original and prediction text.

    Note:
        model.generation_config.return_timestamps = True had to be set to get same results as in asr-evaluate
        This config seems to decrease halluzination

    Todo:
        * If model is loaded from a checkpoint, use the training-config file to load model settings and configs

    Requires:
       get_models (function): A function loading the necessary models for training and evaluation
       compute_metrics (function): A function which computes the metrics (WER in our case)

    Args:
       args (dict): Argument parser. In particular, which model is used and if it is loaded from checkpoint.
       eval_dict (DatasetDict): Dataset dictionary to evaluate the model on
       data_collator (DataCollatorSpeechSeq2SeqWithPadding): Collator for data preparation

    """

    device = select_device()
    logger.info('Device %s detected.', device)

    # Create an evaluation checkpoint file path
    eval_checkpoint_path = os.path.join(args.output_dir, "eval_checkpoint.json")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(eval_checkpoint_path), exist_ok=True)

    # get models
    if args.run_on_local_machine:
        model, feature_extractor, tokenizer, processor = get_whisper_models_from_dir(args.model_type, args.target_language,
                                                                    return_timestamps=args.return_timestamps,
                                                                    load_in_8bit=args.peft)
    else:
        model, feature_extractor, tokenizer, processor = get_whisper_models_from_hub(args.model_type,args.target_language,
                                                                return_timestamps=args.return_timestamps,
                                                                load_in_8bit=args.peft)

    # Load the state dictionary from the checkpoint
    if len(args.model_ckpt_path) > 0:
        if args.peft:
            import peft as _peft_mod
            import transformers as _tf_mod
            from peft import PeftModel, PeftConfig
            from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

            adapter_dir = resolve_adapter_dir(args.model_ckpt_path)
            adapter_cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            if not os.path.isfile(adapter_cfg_path):
                raise FileNotFoundError(
                    f"adapter_config.json not found at {adapter_cfg_path}. "
                    f"Pass --model_ckpt_path pointing at the adapter_model/ dir or its parent."
                )
            with open(adapter_cfg_path) as _f:
                _adapter_meta = json.load(_f)
            logger.info(
                "[env] torch=%s transformers=%s peft=%s",
                torch.__version__, _tf_mod.__version__, _peft_mod.__version__,
            )
            logger.info(
                "[adapter] peft_type=%s r=%s lora_alpha=%s use_dora=%s target_modules=%s base=%s",
                _adapter_meta.get("peft_type"),
                _adapter_meta.get("r"),
                _adapter_meta.get("lora_alpha"),
                _adapter_meta.get("use_dora"),
                _adapter_meta.get("target_modules"),
                _adapter_meta.get("base_model_name_or_path"),
            )
            # PEFT < 0.9 silently drops `use_dora=True`. Refuse to run instead
            # of producing a WER number that secretly came from plain LoRA.
            if _adapter_meta.get("use_dora") and tuple(
                int(p) for p in _peft_mod.__version__.split(".")[:2]
            ) < (0, 9):
                raise RuntimeError(
                    f"Adapter sets use_dora=True but installed peft={_peft_mod.__version__} < 0.9 "
                    f"does not support DoRA. Upgrade peft on this env before evaluating."
                )

            peft_config = PeftConfig.from_pretrained(adapter_dir)
            model = PeftModel.from_pretrained(model, adapter_dir, config=peft_config)
            model.config.use_cache = True
            logger.info('Whisper PEFT model loaded from %s', adapter_dir)
        elif args.model_type == "whisperx":
            import whisperx
            model_dir = args.model_ckpt_path
            model = whisperx.load_model(model_dir, "cuda", compute_type="float16")
            
            
        else:
#            from transformers.utils import safe_load_file
            import glob
            safetensors_files = glob.glob(f"{args.model_ckpt_path}/*.safetensors")
            state_dict = {}
            
            for file_path  in safetensors_files:
                print(f"Loading model checkpoint from: {file_path}")
                partial_state_dict = safetensors.torch.load_file(file_path)
                state_dict.update(partial_state_dict)
                
            #  safetensors.torch.load_file(os.path.join(args.model_ckpt_path, 'model.safetensors'))
            # Fix missing proj_out weights: https://github.com/openai/whisper/discussions/2302
#            model.load_state_dict(state_dict, strict=False)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("missing", len(missing))
            print("unexpected", len(unexpected))
            print("example missing", missing[:20])
            print("example unexpected", unexpected[:20])

            model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
            logger.info('Whisper model from checkpoint %s loaded.', args.model_ckpt_path)
            
            
    else:
        logger.info('Whisper model %s loaded.', args.model_type)

    # Define metric for evaluation
    metric = evaluate.load("wer")

    def compute_metrics(pred_ids, label_ids):
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
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = normalize(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
        label_str = normalize(tokenizer.batch_decode(label_ids, skip_special_tokens=True))

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        # logger.info("Original: %s", label_str)
        # logger.info("Model prediction: %s", pred_str)
        return {"wer": wer, "original": label_str, "predictions": pred_str}

    model.eval().to(device)  # eval mode for model pushed to device
    batch_size = args.eval_batch_size

    """Prepare test set with data_collator"""
    test_ds_ = test_set["test"]

    # Check if we have an evaluation checkpoint to resume from
    start_count = 0
    eval_results = {}
    eval_results_batch = {}
    wer_average = 0

    if os.path.exists(eval_checkpoint_path) and args.resume_evaluation:
        try:
            with open(eval_checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                eval_results = checkpoint_data.get("eval_results", {})
                start_count = checkpoint_data.get("current_count", 0)
            
            test_ds_1, test_ds= test_ds_.split_at_indices([start_count])
            logger.info(f"Resuming evaluation from step {start_count}")

#            # Skip the already processed batches
#            for _ in range(start_count):
#                next(test_ds_iterable)

        except Exception as e:
            logger.warning(f"Failed to load evaluation checkpoint: {e}")
            logger.info("Starting evaluation from the beginning")
    
#    if start_count > 0:
#        # Skip directly to the position we need using Ray's skip method
#        test_ds = test_ds_.skip(start_count)
    else:
        test_ds = test_ds_
        
    test_ds_iterable = test_ds.iter_torch_batches(
        prefetch_batches=args.prefetch_batches,
        batch_size=batch_size, collate_fn=data_collators["testing"]
    )
    
    # Mirror training-eval generate() from
    # trainers/custom_seq2seq_trainers.py: greedy, no forced_decoder_ids, no
    # repetition guards. Language + task ride on model.generation_config
    # (set in models/whisper_models.py:get_whisper_models_from_dir).
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    logger.info(
        "[generate] pad_token_id=%s gen_cfg.language=%s gen_cfg.task=%s "
        "gen_cfg.return_timestamps=%s gen_cfg.forced_decoder_ids=%s",
        pad_token_id,
        getattr(model.generation_config, "language", None),
        getattr(model.generation_config, "task", None),
        getattr(model.generation_config, "return_timestamps", None),
        getattr(model.generation_config, "forced_decoder_ids", None),
    )
    max_eval_batches = getattr(args, "max_eval_batches", None)

    """Loop through test set"""
    count = start_count - 1  # Will be incremented to start_count on first iteration
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            try:
                for batch in test_ds_iterable:
                    count += 1
                    label_ids = batch["labels"]

                    if count < start_count:
                        continue

                    pred_ids = model.generate(
                        batch["input_features"].to(device),
                        pad_token_id=pad_token_id,
                    )

                    outputs = compute_metrics(pred_ids, label_ids)

                    eval_results[str(count)] = outputs
                    eval_results_batch[str(count)] = outputs
                    # Log progress every 10 steps
                    if count % 100 == 0:
                        logger.info("Original: %s", outputs["original"])
                        logger.info("Model prediction: %s", outputs["predictions"])
                        logger.info(f'WER for step {count}: {outputs["wer"]}')

                    if max_eval_batches is not None and (count - start_count + 1) >= max_eval_batches:
                        logger.info(
                            "[smoke] reached --max_eval_batches=%s; stopping early at count=%s",
                            max_eval_batches, count,
                        )
                        break

                    # Save checkpoint every 50 steps
                    if count % 500 == 0 or count == test_ds.count() - 1:
                        # Save intermediate results
                        checkpoint_data = {
                            "current_count": count + 1,  # +1 so we start from the next batch when resuming
                            "eval_results": eval_results,
                        }

                        # Save intermediate results - explicit error handling
                        try:
                            with open(eval_checkpoint_path, 'w') as f:
                                json.dump(checkpoint_data, f)

                            # Also save to a versioned file (in case checkpoint gets corrupted)
                            save_file(eval_results_batch, args.output_dir, mode='json', file_tag=f'eval_step_{count}')
                            logger.info(f"Saved evaluation checkpoint at step {count}")
                            eval_results_batch = {}
                        except Exception as save_error:
                            logger.error(f"Error saving checkpoint: {save_error}")
                            
                 

                        # # Save intermediate results
                        # with open(eval_checkpoint_path, 'w') as f:
                        #     json.dump(checkpoint_data, f)
                        #
                        # # Also save to a versioned file (in case checkpoint gets corrupted)
                        # # intermediate_file = os.path.join(args.output_dir, f"eval_results_step_{count}.json")
                        # save_file(eval_results, args.output_dir, mode='json', file_tag=f'eval_step_{count}')
                        #
                        logger.info(f"Saved evaluation checkpoint at step {count}")

            except Exception as e:
                logger.error(f"Evaluation interrupted at step {count}: {e}")
                # Save progress on error
                # checkpoint_data = {
                #     "current_count": count + 1,
                #     "eval_results": eval_results,
                # }
                # with open(eval_checkpoint_path, 'w') as f:
                #     json.dump(checkpoint_data, f)
                # logger.info(f"Saved evaluation state before error at step {count}")
    

    # Save final results
    save_file(eval_results, args.output_dir, mode='json', file_tag='eval_final')

    # Clean up the checkpoint file after successful completion
    # if os.path.exists(eval_checkpoint_path):
    #     os.remove(eval_checkpoint_path)

    return {"wer_average": outputs["wer"]}

if __name__ == "__main__":
    # Arguments and Logger
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False -- CUDA failed to initialize on this "
            "node/GPU allocation (driver/cgroup issue, not an OOM). Resubmit the job; "
            "if it keeps happening on the same node, request a different one."
        )
    print(torch.cuda.memory_summary())

    # set random seed for reproducibility
    set_seed(args.random_seed)

    # get models
    if args.run_on_local_machine:
        model, feature_extractor, tokenizer, processor = get_whisper_models_from_dir(args.model_type, args.target_language,
                                                                    return_timestamps=args.return_timestamps,
                                                           load_in_8bit=args.peft)
    else:
        model, feature_extractor, tokenizer, processor = get_whisper_models_from_hub(args.model_type,args.target_language,
                                                                return_timestamps=args.return_timestamps,
                                                                load_in_8bit=args.peft)


    if args.h5:
        # Fall back to $DATA_PATH when --path_to_data is not supplied via config.
        path_to_data = args.path_to_data if args.path_to_data is not None \
            else os.environ.get("DATA_PATH", "")

        dataset_name = args.dataset_name
        test_h5_path = os.path.join(path_to_data, f"{dataset_name}_test.h5")

        test_loader = create_ray_indexloader(test_h5_path)

        dataset_size = test_loader.count()
        len_eval_set= dataset_size
        ray_datasets = {
            "test": test_loader,
        }

        # Create the parallel collator with 4 reader processes
        data_collators = {
            "testing": SimpleStreamingCollator(test_h5_path, feature_extractor, tokenizer,
                                                num_workers=args.cpus_per_trial),
        }

    # else:
    #     path_to_data = args.path_to_data if args.debug else r"../data/datasets"
    #     dataset_dict, len_eval_set = None, None
    #     # TODO: fix load_and_prepare_data_from_folders
    #     # load_and_prepare_data_from_folders(path_to_data, feature_extractor, tokenizer,
    #     #                                                              test_size=args.test_split, seed=args.random_seed,
    #     #                                                              evaluate = True, debug = args.debug)
    #
    #     ray_datasets = {"test": ray.data.from_huggingface(dataset_dict["test"])}
    #
    #     # prepare dataset collator
    #     data_collators = {
    #         "testing": DataCollatorSpeechSeq2SeqWithPadding(
    #             processor=processor,
    #             decoder_start_token_id=model.config.decoder_start_token_id,
    #             )
    #     }

    # if args.run_on_local_machine:
    #     args.storage_path = os.path.join(os.getcwd(),"output")
    #     ray.init()
    # else:
    #     ray.init("auto")

    logger.info('len_eval_set: %s',len_eval_set)

    config_ = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__))
    args.output_dir = os.path.join(args.output_dir,  args.output_tag)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    save_file(config_, args.output_dir, file_tag = 'eval')

    resources_per_trial={"CPU": args.cpus_per_trial, "GPU": args.gpus_per_trial}


    # trainer = TorchTrainer(
    #     partial(eval_model, args=args, data_collators=data_collators),  # the training function to execute on each worker.
    #     scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu, resources_per_worker = resources_per_trial, placement_strategy="SPREAD"),
    #     datasets={
    #         "test": ray_datasets["test"],
    #     }
    # )
    # eval_results = trainer.fit()
    eval_results = eval_model(args=args,data_collators=data_collators,test_set = ray_datasets)
    logger.info("eval_results %s",eval_results)
    # eval_model(args, data_collators=data_collators)

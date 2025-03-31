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
import os
import pprint
import pdb

from utils import load_and_prepare_data_from_h5, load_and_prepare_data_from_folders, DataCollatorSpeechSeq2SeqWithPadding, save_file, normalize
import evaluate
import safetensors

import torch

# laod models
from transformers import set_seed
# for loading from checkpoint
from transformers.models.whisper.convert_openai_to_hf import make_linear_from_emb

# For code organization and reporting
import configargparse
import logging

# For Dataset preparation
import ray

# get models
from models import get_whisper_models as get_models

logger = logging.getLogger(__name__)

# We define all the different parameters for the training, model, evaluation etc.
# Whisper model type choices: https://huggingface.co/models?search=openai/whisper
# openai/whisper-large-v3 sofa

# https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
TUNE_CHOICES = ['small_small', 'large_small_BOHB', 'large_small_OPTUNA', 'large_large', '']
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
    parser.add_argument("--model_ckpt_path", type=str, default="", help="loads model from checkpoint training path")
    parser.add_argument("--return_timestamps", action="store_true", help="Return Timestemps mode for model")

    parser.add_argument("--search_schedule_mode", type=str, default="", choices=TUNE_CHOICES,
                        help="Which Searcher Algorithm and Scheduler combination. See 'get_searcher_and_scheduler' function for details.")

    # parser.add_argument("--device", type=str, default="cpu",
    #                     help="Path to audio batch-prepared audio files.")

    # Dataset settings
    parser.add_argument("--test_split", type=float, default=0.2, help="Percentage of test data.")
    parser.add_argument("--h5", action="store_true", help="If data is in .h5 format")
    parser.add_argument("--fp16", action="store_true", default=False, help="Training with floating point 16 ")
    parser.add_argument("--peft", action="store_true", help="Whether or not to do Parameter Efficient Training")

    # Other settings
    parser.add_argument("--run_on_local_machine", action="store_true", help="Store true if training is on local machine.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Base directory where outputs are saved.")
    parser.add_argument("--resume_training", action="store_true", help="Whether or not to resume training.")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--path_to_data", type=str, default="../data/datasets/fzh-wde0459_03_03", help="Path to audio batch-prepared audio files if in debug mode. Otherwise: all data in datasets are loaded")
    parser.add_argument("--random_seed", type=int, default=1337, help="Random Seed for reproducibility")
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--push_to_hub", action="store_true", help="Push best model to Hugging Face")

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

def eval_model(args, eval_dict, data_collator=None):
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

    # get models
    model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language,
                                                                return_timestamps=args.return_timestamps,
                                                                load_in_8bit=args.peft)



    # Load the state dictionary from the checkpoint
    if len(args.model_ckpt_path)>0:

        if args.peft:
        #     from peft import PeftModel, PeftConfig
        #     model = PeftModel.from_pretrained(model, args.model_ckpt_path)
        #     model.config.use_cache = True
        # else:
            state_dict = safetensors.torch.load_file(os.path.join(args.model_ckpt_path, 'model.safetensors'))
            # Fix missing proj_out weights: https://github.com/openai/whisper/discussions/2302
            model.load_state_dict(state_dict, strict=False)
            model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
            model.config.use_cache = True
            logger.info('Whisper model from checkpoint %s loaded.', args.model_ckpt_path)
    else:
        logger.info('Whisper model %s loaded.', args.model_type)

    # Define metric for evaluation
    metric = evaluate.load("wer")
    def compute_metrics(pred_ids,label_ids):
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

        logger.info("Original: %s", label_str)
        logger.info("Model prediction: %s", pred_str)
        # pred_str = model.tokenizer._normalize(pred_str)
        # label_str = model.tokenizer._normalize(label_str)
        return {"wer": wer, "original": label_str, "predictions":pred_str}

    model.eval().to(device)        # eval mode for model pushed to device
    batch_size =  args.eval_batch_size # 1 is more convenient for downstream processing of results as 1 data = 1 row

    """Prepare test set with data_collator"""
    test_ds = ray.data.from_huggingface(eval_dict["test"])
    test_ds_iterable = test_ds.iter_torch_batches(
        batch_size=batch_size, collate_fn=data_collator
    )
    eval_results = {}
    count = -1
    wer_average = 0
    """Loop through test set"""
    with torch.no_grad():
        for batch in test_ds_iterable:
            count += 1
            label_ids = batch["labels"]
            pred_ids = model.generate(batch["input_features"].to(device))
            outputs = compute_metrics(pred_ids,label_ids)

            eval_results[str(count)] = outputs
            wer_average += outputs["wer"]

            if count % 50:
                logger.info('WER for step %s...',count)
                logger.info('...%s',outputs["wer"])

    logger.info("WER average on Test Set %s", wer_average/(count+1))

    save_file(eval_results, args.output_dir, mode = 'json', file_tag='eval')


if __name__ == "__main__":
    # Arguments and Logger
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")

    # set random seed for reproducibility
    set_seed(args.random_seed)

    # get models for preprocessing
    model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language,return_timestamps=args.return_timestamps)

    path_to_data = args.path_to_data if args.debug else r"../data/datasets"

    if args.h5:
        # Create a Hugging Face dataset from generator
        from datasets import Dataset, DatasetDict
        from utils import h5_generator


        # dataset_dict = DatasetDict({
        #     "train": Dataset.from_generator(lambda: h5_generator(path_to_data,split="train",batch_size=1000,seed=args.random_seed)),
        #     "validation": Dataset.from_generator(lambda: h5_generator(path_to_data,split="validation",batch_size=1000,seed=args.random_seed)),
        # })
        dataset_dict, len_eval_set = load_and_prepare_data_from_h5(path_to_data,feature_extractor,tokenizer,
                                                                    debug = args.debug,
                                                                    evaluate=True,
                                                                    seed=args.random_seed)

    else:
        dataset_dict, len_eval_set = load_and_prepare_data_from_folders(path_to_data, feature_extractor, tokenizer,
                                                                     test_size=args.test_split, seed=args.random_seed,
                                                                     evaluate = True, debug = args.debug)

    if not args.run_on_local_machine:
        ray.init('auto')

    logger.info('len_eval_set: %s',len_eval_set)

    config_ = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__))
    args.output_dir = os.path.join(args.output_dir, args.search_schedule_mode, args.output_tag )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_file(config_,args.output_dir, file_tag = 'eval')

    # prepare dataset collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    eval_model(args, dataset_dict, data_collator=data_collator)
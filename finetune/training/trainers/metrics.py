"""
Metric Loader and Evaluator

This module defines a utility for loading and computing evaluation metrics
(such as WER – Word Error Rate) for Whisper ASR fine-tuning tasks. It ensures
normalized evaluation output and supports fallback loading from local directories.

Dependencies:
- HuggingFace `evaluate` package
- Local `wer.py` file containing the metric definition
- A tokenizer that can decode predictions and labels

Returns:
- A `compute_metrics` function usable in HuggingFace training loops.
"""
import evaluate
from utils import normalize
from pathlib import Path
import os
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from projects_paths import TRAINERS_PATH

# Define metric for evaluation

def get_metric_to_optimize(which_metric, tokenizer = None):
    """
    Returns a compute_metrics function based on the selected evaluation metric.

    Args:
        which_metric (str): Currently supports only "wer" (Word Error Rate)
        tokenizer (transformers.PreTrainedTokenizer): Required to decode predictions

    Returns:
        compute_metrics (Callable): A function to compute evaluation metrics
    """
    if which_metric == "wer":
        # Attempt to load WER metric from local project directory
        try:
            metric = evaluate.load(os.path.join(TRAINERS_PATH,"wer.py"))
        except Exception as e:
            print(f"Evaluate.load failed: {e}"
                  f"Trying to load wer metric locally", flush=True)
            try:
                metric = evaluate.load("wer.py")
            except Exception as e:
                print(f"Save the wer.py in the trainers dir: {e}", flush=True)

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

            # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            # we do not want to group tokens when computing the metrics
            pred_str = normalize(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
            label_str = normalize(tokenizer.batch_decode(label_ids, skip_special_tokens=True))
            wer = 100 * metric.compute(predictions=pred_str, references=label_str)

            return {"wer": wer}

        return compute_metrics
    
    elif which_metric == "evaluate_wer":
    
        try:
            metric = evaluate.load(os.path.join(TRAINERS_PATH,"wer.py"))
        except Exception as e:
            print(f"Evaluate.load failed: {e}"
                  f"Trying to load wer metric locally", flush=True)
            try:
                metric = evaluate.load("wer.py")
            except Exception as e:
                print(f"Save the wer.py in the trainers dir: {e}", flush=True)
        
        print("Succesfully retruend evaluate metric")
        return metric
        
    else:
        raise ValueError(f"Unsupported metric: {which_metric}")

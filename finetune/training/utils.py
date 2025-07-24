import torch
import os
import re
import json

def select_device():
    """
    Selects the best available device for PyTorch.

    Returns:
        torch.device: 'cuda', 'mps', or 'cpu' depending on availability.
    """
    # CUDA available
    if torch.cuda.is_available():
        # Prefer MPS (e.g., Apple Silicon)
        if torch.backends.mps.is_available():
            print("Using MPS device")
            return torch.device("mps")
        else:
            print("Using CUDA device")
            return torch.device("cuda")
    # Fallback to CPU
    print("Using CPU")
    return torch.device("cpu")


# ----------------------------
# CLI Argument Parsing
# ----------------------------

def list_of_strings(arg):
    """
    Converts a comma-separated argument string into a list of strings.

    Args:
        arg (str): Comma-separated string, e.g. 'a,b,c'

    Returns:
        list: ['a', 'b', 'c']
    """
    return arg.split(',')


# ----------------------------
# File Saving Utility
# ----------------------------

def save_file(file, output_dir, mode='config', file_tag=''):
    """
    Saves configuration or evaluation output to disk.

    Args:
        file (str or dict): File content (text or JSON-serializable dict).
        output_dir (str): Directory to write the file to.
        mode (str): 'config' saves as .txt, 'json' saves as .json.
        file_tag (str): Optional prefix or tag for filename.
    """
    if mode == 'config':
        config_path = os.path.join(output_dir, file_tag + 'config.txt')
        with open(config_path, 'a') as f:
            print(file, file=f)
    elif mode == 'json':
        eval_path = os.path.join(output_dir, file_tag + '.json')
        with open(eval_path, 'w') as f:
            json.dump(file, f)


# ----------------------------
# Text Normalization
# ----------------------------

def normalize(text):
    """
    Normalize text by lowercasing and removing punctuation.

    Args:
        text (str or list of str): Input string or list of strings.

    Returns:
        str or list of str: Normalized output.
    """
    def process_single_text(single_text):
        result = single_text.strip().lower()
        result = re.sub(r"[!\?\.,;]", "", result)
        return result

    if isinstance(text, list):
        return [process_single_text(t) for t in text]
    elif isinstance(text, str):
        return process_single_text(text)
    else:
        raise TypeError("Input must be a string or a list of strings.")


# ----------------------------
# Training Utility
# ----------------------------

    
def steps_per_epoch(len_train_set, batch_size):
    """Calculates the total number of gradient steps

    Assume gradient_accumulation_steps = 1.

    TODO:
        * Add gradient_accumulation_steps > 1
        * adjust train.py to allow for gradient accumulations

    Args:
        len_train_set (int): Total dataset length
        batch_size (int): batch size
    """
    if len_train_set % batch_size == 0:
        return int(len_train_set / batch_size)
    else:
       return int(len_train_set / batch_size) + 1


def calculate_grace_period(max_t, warmup_steps=0, warmup_ratio=0.0, max_warmup_steps=0):
    """
    Calculate warmup (grace) period for schedulers or early stopping.

    Args:
        max_t (int): Total number of training steps.
        warmup_steps (int): Fixed number of warmup steps.
        warmup_ratio (float): If warmup_steps is 0, use this ratio of max_t.
        max_warmup_steps (int): Override both warmup_steps and warmup_ratio if > 0.

    Returns:
        int: Number of warmup steps (grace period).
    """
    if max_warmup_steps > 0:
        return int(max_warmup_steps)

    if warmup_steps > 0:
        return int(warmup_steps)

    if warmup_ratio > 0:
        return int(round(max_t * warmup_ratio))

    return 0  # default if nothing is specified

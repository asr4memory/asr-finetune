import torch
import os
import re

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


# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')


def save_file(file,output_dir,mode='config',file_tag = ''):
    """Saves {config,eval_results} files.

    Args:
        file (txt,json): A text or json file to be saved.
        output_dir (str): Path to output directory where file will be stored
        mode (str): If `config`: saves config file. If `eval_results`: saves the output eval results as json.
    """
    if mode == 'config':
        config_path = os.path.join(output_dir, file_tag + 'config.txt')
        with open(config_path, 'a') as f:
            print(file, file=f)

    elif mode == 'json':
        eval_path = os.path.join(output_dir, file_tag + '.json')
        with open(eval_path, 'w') as f:
            json.dump(file, f)


def normalize(text):
    """
    Removes certain characters from text and lowers cases.

    Args:
        text (str or list of str): Single string or list of strings to be normalized.

    Returns:
        str or list of str: Normalized string or list of normalized strings.
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


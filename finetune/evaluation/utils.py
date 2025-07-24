""" Collection of utility functions/classes for pre-processing data, saving and more.

Functions are:
   save_file
   load_and_prepare_data_from_folders
   normalize
   steps_per_epoch

Classes are:
    DataCollatorSpeechSeq2SeqWithPadding
"""
import pdb
import json
import logging
import re

import h5py
import ray
import os
import multiprocessing
from multiprocessing import Pool
import time
import numpy as np
import psutil
logger = logging.getLogger(__name__)

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

def log_memory_usage(label=""):
    mem = psutil.virtual_memory()
    logging.info(
        f"MEMORY [{label}]: {mem.percent}% - Used: {mem.used / 1e9:.2f} GB, Available: {mem.available / 1e9:.2f} GB")

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


def steps_per_epoch(len_train_set,batch_size):
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

# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')


def create_ray_indexloader(file_path: str):
    """
    Create Ray dataset for a single HDF5 file.

    Args:
        file_path: Path to the HDF5 file

    Returns:
        Ray dataset
    """
    # Get number of samples in the file
    with h5py.File(file_path, 'r') as f:
        try:
            num_samples = len(f['audio'])
        except:
            num_samples = len(f['audio_waveforms'])

    num_samples = 128
    # Create items with indices
    items = [{"idx": idx} for idx in range(num_samples)]

    # Create dataset
    dataset = ray.data.from_items(items)

    return dataset


_shared_hdf5 = None

def _init_worker(hdf5_path):
    global _shared_hdf5
    _shared_hdf5 = h5py.File(hdf5_path, "r")

def _process_index_shared(idx):
    global _shared_hdf5
    try:
        audio = np.array(_shared_hdf5['audio'][idx], dtype=np.float32).copy()
        transcription = _shared_hdf5['transcription'][idx]
        if isinstance(transcription, bytes):
            transcription = transcription.decode('utf-8')
        return idx, audio, transcription
    except Exception as e:
        print(f"[ERROR] Index {idx}: {e}")
        return idx, None, None

def _extract_features_worker(args):
    audio, sampling_rate, feature_extractor = args
    features = feature_extractor(audio, sampling_rate=sampling_rate)
    return {"input_features": features.input_features[0]}

class SimpleStreamingCollator:
    def __init__(self, hdf5_path, processor, feature_extractor, tokenizer, num_workers=None):
        self.hdf5_path = hdf5_path
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        self.num_workers = num_workers or min(8, multiprocessing.cpu_count() - 1)
        self.pool = None  # Don't create it here

        # Performance tracking
        self.batch_times = []
        self.batch_count = 0

    def __call__(self, batch_dict):
        if self.pool is None:
            self.pool = multiprocessing.Pool(
                processes=self.num_workers,
                initializer=_init_worker,
                initargs=(self.hdf5_path,)
            )

        start = time.time()
        indices = batch_dict['idx']

        # Parallel data loading
        results = self.pool.map(_process_index_shared, indices)
        valid_results = [(idx, audio, trans) for idx, audio, trans in results if audio is not None]

        if not valid_results:
            raise RuntimeError(f"No valid data in batch: {indices}")

        _, audio_list, transcription_list = zip(*valid_results)

        # Feature extraction
        args_list = [(audio, 16000, self.feature_extractor) for audio in audio_list]
        with Pool(self.num_workers) as feat_pool:
            mel_features_list = feat_pool.map(_extract_features_worker, args_list)
        # mel_features_list = []
        # for audio in audio_list:
        #     features = self.feature_extractor(audio, sampling_rate=16000)
        #     mel_features_list.append({"input_features": features.input_features[0]})

        # Performance logging
        elapsed = time.time() - start
        self.batch_times.append(elapsed)
        self.batch_count += 1
        if self.batch_count % 5 == 0:
            avg_time = sum(self.batch_times[-5:]) / 5
            print(f"[Collator] Batch {self.batch_count}: {avg_time:.2f}s, {len(indices)/avg_time:.2f} samples/sec")

        return self._prepare_dataset(mel_features_list, transcription_list)


    def _prepare_dataset(self, mel_features_list, transcriptions):
        padded_features = self.feature_extractor.pad(
            mel_features_list,
            padding="longest",
            return_tensors="pt"
        )
        input_features = padded_features.input_features

        tokenized_labels = [
            self.tokenizer(text if isinstance(text, str) else str(text)).input_ids
            for text in transcriptions
        ]
        label_features = [{"input_ids": ids} for ids in tokenized_labels]
        labels_batch = self.tokenizer.pad(label_features,
                                          padding="max_length",
                                          max_length=448,
                                          return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return {"input_features": input_features, "labels": labels}

    def __del__(self):
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()


# Define a Ray actor to handle HDF5 files
@ray.remote
class HDF5Worker:
    def __init__(self, hdf5_path, feature_extractor):
        self.hdf5_path = hdf5_path
        self.feature_extractor = feature_extractor

    def process_index_with_features(self, idx):
        try:
            with h5py.File(self.hdf5_path, "r") as f:
                # pdb.set_trace()
                audio = f["audio"][idx]
                trans = f["transcription"][idx].decode('utf-8')
            # audio = np.array(_shared_hdf5['audio'][idx], dtype=np.float32).copy()
            # transcription = _shared_hdf5['transcription'][idx]
            # if isinstance(transcription, bytes):
            #     transcription = transcription.decode('utf-8')
            features = self.feature_extractor(audio, sampling_rate=16000)
            mel_features = {"input_features": features.input_features[0]}

            return idx, mel_features, trans
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            return idx, None, None


class MultiStreamingCollator:
    def __init__(self, hdf5_path, processor, feature_extractor, tokenizer, num_workers=None):
        """
        Initialize a streaming collator for a single HDF5 file.

        Args:
            hdf5_path: Path to the HDF5 file
            processor: The processor for the model
            feature_extractor: Feature extractor for audio
            tokenizer: Tokenizer for transcriptions
            num_workers: Number of parallel workers
        """
        self.hdf5_path = hdf5_path
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.num_workers = num_workers or 2  # Use a reasonable default

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create workers
        self.workers = [HDF5Worker.remote(hdf5_path, feature_extractor)
                        for _ in range(self.num_workers)]

        # Performance tracking
        self.batch_times = []
        self.batch_count = 0

    def __call__(self, batch_dict):
        """Process a batch of indices."""
        start = time.time()
        indices = batch_dict['idx']

        # Distribute work among workers round-robin
        futures = []
        for i, idx in enumerate(indices):
            worker = self.workers[i % len(self.workers)]
            futures.append(worker.process_index_with_features.remote(idx))

        # Get results
        results = ray.get(futures)
        valid_results = [(idx, mel_features, trans) for idx, mel_features, trans in results if mel_features is not None]

        if not valid_results:
            raise RuntimeError(f"No valid data in batch: {indices}")

        _, mel_features_list, transcription_list = zip(*valid_results)

        # Performance logging
        elapsed = time.time() - start
        self.batch_times.append(elapsed)
        self.batch_count += 1
        if self.batch_count % 5 == 0:
            avg_time = sum(self.batch_times[-5:]) / 5
            print(f"[Collator] Batch {self.batch_count}: {avg_time:.2f}s, {len(indices)/avg_time:.2f} samples/sec")

        return self._prepare_dataset(mel_features_list, transcription_list)

    def _prepare_dataset(self, mel_features_list, transcriptions):
        """Prepare the dataset for the model."""
        padded_features = self.feature_extractor.pad(
            mel_features_list,
            padding="longest",
            return_tensors="pt"
        )
        input_features = padded_features.input_features

        tokenized_labels = [
            self.tokenizer(text if isinstance(text, str) else str(text)).input_ids
            for text in transcriptions
        ]
        label_features = [{"input_ids": ids} for ids in tokenized_labels]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return {"input_features": input_features, "labels": labels}

    def __del__(self):
        """Clean up resources when the collator is garbage collected."""
        self.workers = {}  # Ray will automatically clean up the actors

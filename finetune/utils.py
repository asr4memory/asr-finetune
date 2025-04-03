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
import os
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


def split_indices(dataset_size, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Generate shuffled indices for dataset splitting."""
    np.random.seed(seed)
    indices = np.random.permutation(dataset_size)

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    train_size -= val_size  # adjust to avoid overlap
    # test = remaining

    return {
        "train": np.sort(indices[:train_size]),
        "validation": np.sort(indices[train_size:train_size + val_size]),
        "test": np.sort(indices[train_size + val_size:])
    }

### Claude input

from typing import Tuple

# Updated function to create dataloaders for all splits
def create_ray_indexloaders(
        split_indices: dict,
) -> Tuple[object, object, object]:

    # Get train, validation, and test indices using the provided function
    train_indices, val_indices, test_indices = split_indices["train"], split_indices["validation"], split_indices["test"]

    # Create Ray datasets for each split
    train_ds = ray.data.from_items([{"idx": i} for i in train_indices])#.repartition(num_blocks=num_parallel_tasks)
    val_ds = ray.data.from_items([{"idx": i} for i in val_indices])#.repartition(num_blocks=num_parallel_tasks)
    test_ds = ray.data.from_items([{"idx": i} for i in test_indices])#.repartition(num_blocks=num_parallel_tasks)

    return train_ds, val_ds, test_ds


import ray


import multiprocessing
import time
import numpy as np

_shared_hdf5 = None

def _init_worker(hdf5_path):
    global _shared_hdf5
    import h5py
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

class SimpleStreamingCollator:
    def __init__(self, hdf5_path, feature_extractor, tokenizer, num_workers=None):
        self.hdf5_path = hdf5_path
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
        mel_features_list = []
        for audio in audio_list:
            features = self.feature_extractor(audio, sampling_rate=16000)
            mel_features_list.append({"input_features": features.input_features[0]})

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
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        return {"input_features": input_features, "labels": labels}

    def __del__(self):
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                try:
                    self.pool.close()
                    self.pool.join()
                except Exception as e:
                    logger.warning(f"[SimpleStreamingCollator] Error closing/joining pool: {e}")
        except Exception as e:
            logger.warning(f"[SimpleStreamingCollator] Unexpected error in __del__: {e}")


import time
import numpy as np
import multiprocessing.dummy as mp_dummy  # ThreadPool version
import h5py
import torch

_shared_hdf5 = None
_shared_featurizer = None

def _init_worker(hdf5_path, feature_extractor):
    global _shared_hdf5, _shared_featurizer
    _shared_hdf5 = h5py.File(hdf5_path, "r", libver="latest", swmr=True)
    _shared_featurizer = feature_extractor

def _process_index_with_features(idx):
    global _shared_hdf5, _shared_featurizer
    try:
        audio = np.array(_shared_hdf5['audio_waveforms'][idx], dtype=np.float32).copy()
        transcription = _shared_hdf5['transcription'][idx]
        if isinstance(transcription, bytes):
            transcription = transcription.decode('utf-8')
        features = _shared_featurizer(audio, sampling_rate=16000).input_features[0]
        return idx, features, transcription
    except Exception as e:
        print(f"[ERROR] Index {idx}: {e}")
        return idx, None, None

class FastHDF5AudioCollator:
    def __init__(self, hdf5_path, feature_extractor, tokenizer, num_workers=None):
        self.hdf5_path = hdf5_path
        self.feature_extractor = feature_extractor  # needed only for init inside workers
        self.tokenizer = tokenizer

        self.num_workers = num_workers or min(8, multiprocessing.cpu_count() - 1)
        self.pool = None

        self.batch_times = []
        self.batch_count = 0

    def __call__(self, batch_dict):
        if self.pool is None:
            self.pool = mp_dummy.Pool(
                processes=self.num_workers,
                initializer=_init_worker,
                initargs=(self.hdf5_path, self.feature_extractor)
            )

        start = time.time()
        indices = batch_dict['idx']

        results = self.pool.map(_process_index_with_features, indices)
        valid_results = [(idx, features, trans) for idx, features, trans in results if features is not None]

        if not valid_results:
            raise RuntimeError(f"No valid data in batch: {indices}")

        _, input_features_list, transcription_list = zip(*valid_results)

        # Pad features into tensor batch
        padded_features = self.feature_extractor.pad(
            [{"input_features": f} for f in input_features_list],
            padding="longest",
            return_tensors="pt"
        )
        input_features = padded_features.input_features

        # Batched tokenization
        tokenized = self.tokenizer(
            list(transcription_list),
            padding=True,
            return_tensors="pt"
        )
        labels = tokenized["input_ids"].masked_fill(
            tokenized.attention_mask.ne(1), -100
        )

        # Performance tracking
        elapsed = time.time() - start
        self.batch_times.append(elapsed)
        self.batch_count += 1
        if self.batch_count % 5 == 0:
            avg_time = sum(self.batch_times[-5:]) / 5
            print(f"[FastCollator] Batch {self.batch_count}: {avg_time:.2f}s, {len(indices)/avg_time:.2f} samples/sec")

        return {"input_features": input_features, "labels": labels}

    def __del__(self):
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.close()
                self.pool.join()
        except Exception as e:
            print(f"[FastCollator] Error closing thread pool: {e}")

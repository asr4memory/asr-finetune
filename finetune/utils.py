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

#
# def load_and_prepare_data_from_folders(path,feature_extractor,tokenizer,test_size=0.2, seed = 0, evaluate = False,
#                                        debug = False, num_proc = 1):
#     """Loads and prepares data from a folder directory.
#
#     `Important`: Each folder needs to have a subfolder "data" (name not important) containing the .mav audio files AND
#                  a metadata.csv file with columns 'file_name' and 'transcription'. The file_name must match the file
#                  name in the data folder. The transcription is a string of the true transcription of the audio.
#
#     We
#         1. loop through subfolders of path-folder, load each folder as dataset, and concatenate datasets
#         2. Do the train, validation, and test splits
#         3. Resample the audio data and compute the log-Mel input features
#
#     Args:
#         path (str): Directory path of head data folder
#         feature_extractor (WhisperFeatureExtractor): Feature extractor calculates the log-Mel representations of inputs
#                                                      used for training and evaluation
#         tokenizer (WhisperTokenizer): The tokenizer is converting target-text into tokens.
#         test_size (float): Fraction of total data used for testing.
#         seed (int): random seed for reproducibility
#         evaluate (bool): If true, only the test set is created. Otherwise: train and validation set.
#         debug (bool): If true, does some statistics on the test set (if evaluate=True) or validation set (otherwise).
#                       Should result to the same value if one wants to compare two different models.
#     """
#     data_collection = []
#     first_level_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
#     num_rows = 0
#     """Step 1"""
#     for subfolder in first_level_subfolders:
#         dataset = load_dataset("audiofolder", data_dir=subfolder) # Laden des Datasets von der bereinigten CSV-Datei
#         data_collection += [dataset['train']]
#         num_rows += dataset['train'].num_rows
#
#     dataset = concatenate_datasets(data_collection)
#
#     assert dataset.num_rows == num_rows, "Some data got lost in the process of concatenation."
#
#     """Step 2: Dataset in Trainings- und Testsets aufteilen """
#     split_dataset = dataset.train_test_split(test_size=test_size, seed = seed)  # 20% für Testdaten
#     split_trainset = split_dataset['train'].train_test_split(test_size=0.1, seed = seed) # 10% of training for validation
#
#     # Erstellen eines DatasetDict-Objekts
#     if evaluate:
#         dataset_dict = DatasetDict({
#             'test': split_dataset['test']
#         })
#     else:
#         dataset_dict = DatasetDict({
#             'train': split_trainset['train'], #split_dataset['train'],
#             'validation': split_trainset['test'],
#         })
#
#     def prepare_dataset(batch):
#         # load and resample audio data from 48 to 16kHz
#         audio = batch["audio"]
#
#         # compute log-Mel input features from input audio array
#         batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
#
#         # encode target text to label ids
#         batch["labels"] = tokenizer(batch["transcription"]).input_ids
#         return batch
#
#     """Step 3: Apply prepare_dataset mapping to dataset_dict"""
#     dataset_dict = dataset_dict.map_batches(prepare_dataset,
#                    remove_columns=dataset_dict.column_names["test"] if evaluate else dataset_dict.column_names["train"],
#                                             num_proc=num_proc)
#
#     logger.info('len validation set: %s', split_trainset['test'].num_rows)
#     logger.info('len test set: %s', split_dataset['test'].num_rows)
#
#     return dataset_dict, split_trainset['train'].num_rows




# from dataclasses import dataclass
# from typing import Any
# import numpy as np
# @dataclass
# class DataCollatorSpeechSeq2SeqWithPadding:
#     """Data Collator for Speech Seq2Seq models with Padding
#
#     We used the collator suggested by the tutorial: https://huggingface.co/blog/fine-tune-whisper.
#     We had to slightly modify it due our data being in a different format as required by ray tune.
#
#     Attributes:
#        processor (WhisperProcessor): Processor used for padding (normalizing data to same length)
#        decoder_start_token_id (int): Token indicating the Beginning Of Setence (BOS)
#
#     Methods:
#        __call__(features): Processing a dictionary of input features
#     """
#     processor: Any
#     decoder_start_token_id: int
#
#     def __call__(self, features):
#         """ Processing a dictionary of input features.
#
#         Input features are padded to `longest` forms and pytorch tensors are returned.
#
#         Args:
#             features (dict): A dictionary with keys 'input_features' consiting of log-Mel features and tokenized
#                              'labels'
#         Returns"
#             batch (dict): Dictionary of padded `input_features` and `labels` as pytorch tensors
#         """
#         # split inputs and labels since they have to be of different lengths and need different padding methods
#         # first treat the audio inputs by simply returning torch tensors
#         # input_features = [{"input_features": np.vstack(list(feature))} for feature in features["input_features"]]
#         # batch = self.processor.feature_extractor.pad(input_features,  padding='longest', return_tensors="pt")
#         #
#         # #lab_feat = [{"input_ids": feature} for feature in features["labels"]]
#         # # pad the labels to max length
#         # labels_batch = lab_feat #self.processor.tokenizer.pad(lab_feat, return_tensors="pt")
#         #
#         #
#         # # replace padding with -100 to ignore loss correctly
#         # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#         #
#         # # if bos token is appended in previous tokenization step,
#         # # cut bos token here as it's append later anyways
#         # if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
#         #     labels = labels[:, 1:]
#         #
#         # batch["labels"] = labels
#         input_features = [{"input_features": feat["input_features"].reshape([128, 3000])} for feat in features["input_features"]]
#
#         batch = self.processor.feature_extractor.pad(input_features, padding='longest', return_tensors="pt")
#
#         # batch["input_features"] = features["input_features"]
#         batch["labels"] = features["labels"]
#         return batch
#



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
    test_size = dataset_size - train_size
    val_size = int(val_ratio * train_size)
    train_size -= val_size  # Adjust train size after validation split

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

import torch
from typing import Dict, List, Any
import time
from concurrent.futures import ProcessPoolExecutor
import os
import functools


# This needs to be a top-level function for multiprocessing to work properly
def _process_hdf5_index(hdf5_path, idx):
    """Process a single index from an HDF5 file"""
    import h5py

    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Get audio with proper dtype
            audio = np.array(f['audio'][idx], dtype=np.float32).copy()

            # Get transcription
            transcription = f['transcription'][idx]
            if isinstance(transcription, bytes):
                transcription = transcription.decode('utf-8')

        return idx, audio, transcription
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return idx, None, None


class SimpleMultiprocessingCollator:
    """
    A simple multiprocessing-based data collator that works with Ray.
    Uses Python's built-in multiprocessing to parallelize HDF5 reading.
    """

    def __init__(self, hdf5_path, processor, feature_extractor, tokenizer,
                 num_workers=None):
        self.hdf5_path = hdf5_path
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        # Use at most half of available CPUs to avoid resource contention
        self.num_workers = num_workers or max(1, min(4, multiprocessing.cpu_count() // 2))

        # Performance tracking
        self.batch_times = []
        self.batch_count = 0
        # Add prefetching
        self.prefetch_queue = Queue(maxsize=3)
        self.prefetch_thread = None
        self._start_prefetching()

    def __call__(self, batch_dict: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch using multiprocessing"""
        start_time = time.time()

        # Get indices
        indices = batch_dict['idx']

        # Create the process executor
        # Use at most 4 workers to avoid overwhelming the system
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Create a partial function with hdf5_path filled in
            process_func = functools.partial(_process_hdf5_index, self.hdf5_path)

            # Map indices to workers and collect results
            results = list(executor.map(process_func, indices))

        # Sort results by original index position
        # since they might come back in a different order
        index_dict = {idx: i for i, idx in enumerate(indices)}
        sorted_results = sorted(results, key=lambda x: index_dict[x[0]])

        # Extract audio and transcriptions
        audio_list = []
        transcription_list = []

        for _, audio, transcription in sorted_results:
            if audio is not None:  # Check for errors during processing
                audio_list.append(audio)
                transcription_list.append(transcription)

        # Process features in the main process
        mel_features_list = []
        for audio in audio_list:
            features = self.feature_extractor(audio, sampling_rate=16000)
            mel_features_list.append({"input_features": features.input_features[0]})

        # Prepare the final dataset
        result = self._prepare_dataset(mel_features_list, transcription_list)

        # Track performance
        elapsed = time.time() - start_time
        self.batch_times.append(elapsed)
        self.batch_count += 1

        # Print performance stats occasionally
        if self.batch_count % 5 == 0:  # Print more often
            recent_times = self.batch_times[-5:] if len(self.batch_times) >= 5 else self.batch_times
            avg_time = sum(recent_times) / len(recent_times)
            samples_per_sec = len(indices) / avg_time
            print(f"Batch {self.batch_count}: {avg_time:.2f}s, {samples_per_sec:.1f} samples/sec")

            # Keep batch_times from growing too large
            if len(self.batch_times) > 50:
                self.batch_times = self.batch_times[-25:]

        return result

    def _start_prefetching(self):
        """Start background thread for prefetching batches"""
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                daemon=True
            )
            self.prefetch_thread.start()

    def _prepare_dataset(self, mel_features_list, transcriptions):
        """Prepare the final dataset from processed features and transcriptions"""
        # Pad features
        padded_features = self.feature_extractor.pad(
            mel_features_list,
            padding='longest',
            return_tensors="pt"
        )

        # Get input features (keep as float32)
        input_features = padded_features.input_features

        # Process transcriptions
        tokenized_labels = []
        for text in transcriptions:
            if not isinstance(text, str):
                text = str(text)
            tokenized = self.tokenizer(text).input_ids
            tokenized_labels.append(tokenized)

        # Pad labels
        label_features = [{"input_ids": ids} for ids in tokenized_labels]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        return {
            "input_features": input_features,
            "labels": labels
        }


import threading
from queue import Queue
import multiprocessing
import numpy as np
import h5py


class SimpleMultiprocessingCollatorFast:
    """
    A multiprocessing collator that is properly serializable for Ray
    """

    def __init__(self, hdf5_path, processor, feature_extractor, tokenizer, num_workers=None):
        self.hdf5_path = hdf5_path
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        # Use more CPUs when available
        self.num_workers = min(8, multiprocessing.cpu_count() // 4)
        # self.num_workers = num_workers or max(2, multiprocessing.cpu_count() - 1)
        # Performance tracking
        self.batch_times = []
        self.batch_count = 0

        # No threading or locks that would cause serialization issues
        self._last_batch_indices = None

    def __call__(self, batch_dict):
        """Process a batch using multiprocessing"""
        start_time = time.time()

        indices = batch_dict['idx']
        self._last_batch_indices = indices  # Store for debugging

        # Process in parallel using a process pool
        # This avoids threading locks by using separate processes
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.map(
                self._process_index_wrapper,
                [(self.hdf5_path, idx) for idx in indices]
            )

        # Filter out any failed processings
        valid_results = [(idx, audio, trans) for idx, audio, trans in results if audio is not None]

        # if not valid_results:
        #     print(f"Warning: No valid results for batch with indices {indices}")
        #     # Return a minimal valid batch
        #     return {
        #         "input_features": torch.zeros((1, 80, 3000), dtype=torch.float16),
        #         "labels": torch.tensor([[-100]]) #, dtype=torch.long)
        #     }

        # Unpack results
        _, audio_list, transcription_list = zip(*valid_results)

        # Process features
        mel_features_list = []
        for audio in audio_list:
            features = self.feature_extractor(
                audio,
                sampling_rate=16000,
            )
            mel_features_list.append({
                "input_features": features.input_features[0]
            })

        # Track performance
        elapsed = time.time() - start_time
        self.batch_times.append(elapsed)
        self.batch_count += 1

        # Print performance stats occasionally
        if self.batch_count % 5 == 0:
            recent_times = self.batch_times[-5:] if len(self.batch_times) >= 5 else self.batch_times
            avg_time = sum(recent_times) / len(recent_times)
            samples_per_sec = len(indices) / avg_time
            print(f"Batch {self.batch_count}: {avg_time:.2f}s, {samples_per_sec:.1f} samples/sec")

            # Keep batch_times from growing too large
            if len(self.batch_times) > 50:
                self.batch_times = self.batch_times[-25:]

        return self._prepare_dataset(mel_features_list, transcription_list)

    @staticmethod
    def _process_index_wrapper(args):
        """Static method wrapper for multiprocessing"""
        hdf5_path, idx = args
        return SimpleMultiprocessingCollatorOptimized._process_index(hdf5_path, idx)

    @staticmethod
    def _process_index(hdf5_path, idx):
        """Process a single index - static method for better multiprocessing"""
        try:
            start = time.time()
            with h5py.File(hdf5_path, 'r') as f:
                audio = np.array(f['audio'][idx], dtype=np.float32).copy()
                transcription = f['transcription'][idx]
                if isinstance(transcription, bytes):
                    transcription = transcription.decode('utf-8')
            io_time = time.time() - start
            if idx % 100 == 0:
                print(f"[DEBUG] I/O time for idx {idx}: {io_time:.3f}s")

            return idx, audio, transcription
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return idx, None, None

    def _prepare_dataset(self, mel_features_list, transcriptions):
        """Prepare the final dataset"""
        # Pad features
        padded_features = self.feature_extractor.pad(
            mel_features_list,
            padding='longest',
            return_tensors="pt"
        )

        # Get input features
        input_features = padded_features.input_features

        # Process transcriptions
        tokenized_labels = []
        for text in transcriptions:
            if not isinstance(text, str):
                text = str(text)
            tokenized = self.tokenizer(text).input_ids
            tokenized_labels.append(tokenized)

        # Pad labels
        label_features = [{"input_ids": ids} for ids in tokenized_labels]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        return {
            "input_features": input_features,
            "labels": labels
        }

    def __getstate__(self):
        """
        Make sure we don't include any non-serializable objects like thread locks
        """
        state = self.__dict__.copy()
        # Clean up potentially problematic attributes
        for key in list(state.keys()):
            try:
                # If something might involve locks, remove it for serialization
                if any(s in str(type(state[key])).lower() for s in
                       ['thread', 'lock', 'queue', 'process', 'connection']):
                    del state[key]
            except:
                # If any issues when checking type, just delete it
                del state[key]
        return state

class SimpleMultiprocessingCollatorFast:
    """
    A multiprocessing collator that is properly serializable for Ray
    """
    _shared_hdf5_file = None

    def __init__(self, hdf5_path, processor, feature_extractor, tokenizer, num_workers=None):
        self.hdf5_path = hdf5_path
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        # Use more CPUs when available
        self.num_workers = min(8, multiprocessing.cpu_count() // 4)
        # self.num_workers = num_workers or max(2, multiprocessing.cpu_count() - 1)
        # Performance tracking
        self.batch_times = []
        self.batch_count = 0

        # No threading or locks that would cause serialization issues
        self._last_batch_indices = None

        self.pool = multiprocessing.Pool(processes=self.num_workers, initializer=init_worker,
                                         initargs=(hdf5_path,))

    def __call__(self, batch_dict):
        """Process a batch using multiprocessing"""
        start_time = time.time()

        indices = batch_dict['idx']
        self._last_batch_indices = indices  # Store for debugging

        # Process in parallel using a process pool
        # This avoids threading locks by using separate processes
        # with multiprocessing.Pool(processes=self.num_workers, initializer=init_worker,
        #                           initargs=(self.hdf5_path,)) as pool:
        results = selfpool.map(process_index_shared, indices)

        # Filter out any failed processings
        valid_results = [(idx, audio, trans) for idx, audio, trans in results if audio is not None]

        # if not valid_results:
        #     print(f"Warning: No valid results for batch with indices {indices}")
        #     # Return a minimal valid batch
        #     return {
        #         "input_features": torch.zeros((1, 80, 3000), dtype=torch.float16),
        #         "labels": torch.tensor([[-100]]) #, dtype=torch.long)
        #     }

        # Unpack results
        _, audio_list, transcription_list = zip(*valid_results)

        # Process features
        mel_features_list = []
        for audio in audio_list:
            features = self.feature_extractor(
                audio,
                sampling_rate=16000,
            )
            mel_features_list.append({
                "input_features": features.input_features[0]
            })

        # Track performance
        elapsed = time.time() - start_time
        self.batch_times.append(elapsed)
        self.batch_count += 1

        # Print performance stats occasionally
        if self.batch_count % 5 == 0:
            recent_times = self.batch_times[-5:] if len(self.batch_times) >= 5 else self.batch_times
            avg_time = sum(recent_times) / len(recent_times)
            samples_per_sec = len(indices) / avg_time
            print(f"Batch {self.batch_count}: {avg_time:.2f}s, {samples_per_sec:.1f} samples/sec")

            # Keep batch_times from growing too large
            if len(self.batch_times) > 50:
                self.batch_times = self.batch_times[-25:]

        return self._prepare_dataset(mel_features_list, transcription_list)

    def init_worker(hdf5_path):
        global _shared_hdf5_file
        import h5py
        _shared_hdf5_file = h5py.File(hdf5_path, 'r')

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

    def process_index_shared(idx):
        global _shared_hdf5_file
        try:
            audio = np.array(_shared_hdf5_file['audio'][idx], dtype=np.float32).copy()
            transcription = _shared_hdf5_file['transcription'][idx]
            if isinstance(transcription, bytes):
                transcription = transcription.decode('utf-8')
            return idx, audio, transcription
        except Exception as e:
            print(f"[ERROR] Index {idx} failed: {e}")
            return idx, None, None

    @staticmethod
    def _process_index_wrapper(args):
        """Static method wrapper for multiprocessing"""
        hdf5_path, idx = args
        return SimpleMultiprocessingCollatorOptimized._process_index(hdf5_path, idx)

    @staticmethod
    def _process_index(hdf5_path, idx):
        """Process a single index - static method for better multiprocessing"""
        try:
            start = time.time()
            with h5py.File(hdf5_path, 'r') as f:
                audio = np.array(f['audio'][idx], dtype=np.float32).copy()
                transcription = f['transcription'][idx]
                if isinstance(transcription, bytes):
                    transcription = transcription.decode('utf-8')
            io_time = time.time() - start
            if idx % 100 == 0:
                print(f"[DEBUG] I/O time for idx {idx}: {io_time:.3f}s")

            return idx, audio, transcription
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return idx, None, None

    def _prepare_dataset(self, mel_features_list, transcriptions):
        """Prepare the final dataset"""
        # Pad features
        padded_features = self.feature_extractor.pad(
            mel_features_list,
            padding='longest',
            return_tensors="pt"
        )

        # Get input features
        input_features = padded_features.input_features

        # Process transcriptions
        tokenized_labels = []
        for text in transcriptions:
            if not isinstance(text, str):
                text = str(text)
            tokenized = self.tokenizer(text).input_ids
            tokenized_labels.append(tokenized)

        # Pad labels
        label_features = [{"input_ids": ids} for ids in tokenized_labels]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        return {
            "input_features": input_features,
            "labels": labels
        }

    def __getstate__(self):
        """
        Make sure we don't include any non-serializable objects like thread locks
        """
        state = self.__dict__.copy()
        # Clean up potentially problematic attributes
        for key in list(state.keys()):
            try:
                # If something might involve locks, remove it for serialization
                if any(s in str(type(state[key])).lower() for s in
                       ['thread', 'lock', 'queue', 'process', 'connection']):
                    del state[key]
            except:
                # If any issues when checking type, just delete it
                del state[key]
        return state



import multiprocessing
import time
import torch
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperTokenizer

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
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

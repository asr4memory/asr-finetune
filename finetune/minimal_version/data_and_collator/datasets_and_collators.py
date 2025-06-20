import multiprocessing
import pdb
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
import torch
import os
import shutil
import h5py
import ray
from models.whisper_models import get_whisper_models
from .data_modes import get_data_modes

import logging
logger = logging.getLogger(__name__)

_shared_hdf5 = None


def make_dataset_kwargs(args):
    """Arguments Filter for the dataset loading.

    Args:
        args (dict): dictionary of keyboard arguments
    Returns:
       data_kwargs (dict): dictionary of relevant data fetching arguments
    """

    data_kwargs = {
        "cpus_per_trial": args.cpus_per_trial,
        "random_seed": args.random_seed,
        "model_type": args.model_type,
        "target_language": args.target_language,
        "return_timestamps": args.return_timestamps,
        "run_on_local_machine": args.run_on_local_machine,
        "path_to_data": args.path_to_data,
        "dataset_name": args.dataset_name,
        "peft": args.peft,
        "debug": args.debug,
        "data_mode": args.data_mode,
    }

    return data_kwargs


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
    def __init__(self, hdf5_path, feature_extractor, tokenizer, num_workers=None, copy_to_local=False):
        self.hdf5_path = self._copy_to_local(hdf5_path) if copy_to_local else hdf5_path
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        # Allow explicit num_workers=0 for single-process mode
        if num_workers == 0:
            self.num_workers = 0
        else:
            self.num_workers = min(num_workers or 4, multiprocessing.cpu_count() - 1, 8)

        self.pool = None
        self.h5file = None  # For single-process mode

        # Performance tracking
        self.batch_times = []
        self.batch_count = 0

    def __call__(self, batch_dict):
        start = time.time()
        indices = batch_dict['idx']

        if self.num_workers == 0:
            # Single-process mode
            if self.h5file is None:
                import h5py
                self.h5file = h5py.File(self.hdf5_path, "r")

            # Process indices sequentially
            results = []
            for idx in indices:
                try:
                    audio = np.array(self.h5file['audio'][idx], dtype=np.float32).copy()
                    transcription = self.h5file['transcription'][idx]

                    if isinstance(transcription, bytes):
                        transcription = transcription.decode('utf-8')

                    results.append((idx, audio, transcription))
                except Exception as e:
                    print(f"[ERROR] Index {idx}: {e}")
        else:
            # Multi-process mode
            if self.pool is None:
                self.pool = multiprocessing.Pool(
                    processes=self.num_workers,
                    initializer=_init_worker,
                    initargs=(self.hdf5_path,)
                )

            # Parallel data loading
            results = self.pool.map(_process_index_shared, indices)

            # Clean up right away
            self.pool.close()
            self.pool.join()
            self.pool = None

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
            print(f"[Collator] Batch {self.batch_count}: {avg_time:.2f}s, {len(indices) / avg_time:.2f} samples/sec")

        return self._prepare_dataset(mel_features_list, transcription_list)

    def _copy_to_local(self, path: str) -> str:
        """Copy HDF5 file to local storage for better performance."""
        fname = os.path.basename(path)
        local_dir = "/tmp"
        local_path = os.path.join(local_dir, fname)
        if not os.path.exists(local_path):
            try:
                print(f"[INFO] Copying {path} to {local_path} (node-local)...")
                start_time = time.time()
                shutil.copy2(path, local_path)
                elapsed = time.time() - start_time
                print(f"[INFO] Copy completed in {elapsed:.2f}s")
            except Exception as e:
                print(f"[WARNING] Failed to copy to local disk: {e}")
                return path
        return local_path

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

        def cleanup(self):
            """Explicitly clean up resources"""
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.close()
                self.pool.join()
                self.pool = None

            if hasattr(self, 'h5file') and self.h5file is not None:
                self.h5file.close()
                self.h5file = None

        def __del__(self):
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.close()
                self.pool.join()

            if hasattr(self, 'h5file') and self.h5file is not None:
                self.h5file.close()



def collate_parquet(batch):
    input_features_batch = torch.stack([torch.from_numpy(x) for x in batch["input_features"]])
    labels_batch = torch.stack([torch.from_numpy(x) for x in batch["labels"]])
    return {
        "input_features": input_features_batch, #input_features_batch,
        "labels": labels_batch# labels_batch
    }


def create_ray_indexloader(file_path: str, samples=0):
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

    if samples > 0:
        num_samples = samples
    # Create items with indices
    items = [{"idx": idx} for idx in range(num_samples)]

    # Create dataset
    dataset = ray.data.from_items(items)

    return dataset


from datasets import load_dataset, DatasetDict, concatenate_datasets
def load_and_prepare_data_from_folders(path,feature_extractor,tokenizer,test_size=0.2, seed = 0, mode = "train", debug = False):
    """Loads and prepares data from a folder directory.

    `Important`: Each folder needs to have a subfolder "data" (name not important) containing the .mav audio files AND
                 a metadata.csv file with columns 'file_name' and 'transcription'. The file_name must match the file
                 name in the data folder. The transcription is a string of the true transcription of the audio.

    We
        1. loop through subfolders of path-folder, load each folder as dataset, and concatenate datasets
        2. Do the train, validation, and test splits
        3. Resample the audio data and compute the log-Mel input features

    Args:
        path (str): Directory path of head data folder
        feature_extractor (WhisperFeatureExtractor): Feature extractor calculates the log-Mel representations of inputs
                                                     used for training and evaluation
        tokenizer (WhisperTokenizer): The tokenizer is converting target-text into tokens.
        test_size (float): Fraction of total data used for testing.
        seed (int): random seed for reproducibility
        evaluate (bool): If true, only the test set is created. Otherwise: train and validation set.
        debug (bool): If true, does some statistics on the test set (if evaluate=True) or validation set (otherwise).
                      Should result to the same value if one wants to compare two different models.
    """
    data_collection = []
    first_level_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    num_rows = 0
    """Step 1"""
    for subfolder in first_level_subfolders:
        dataset = load_dataset("audiofolder", data_dir=subfolder) # Laden des Datasets von der bereinigten CSV-Datei
        data_collection += [dataset['train']]
        num_rows += dataset['train'].num_rows

    dataset = concatenate_datasets(data_collection)

    assert dataset.num_rows == num_rows, "Some data got lost in the process of concatenation."

    """Step 2: Dataset in Trainings- und Testsets aufteilen """
    split_dataset = dataset.train_test_split(test_size=test_size, seed = seed)  # 20% für Testdaten
    split_trainset = split_dataset['train'].train_test_split(test_size=0.1, seed = seed) # 10% of training for validation

    # Erstellen eines DatasetDict-Objekts
    if mode == 'test':
        dataset_dict = DatasetDict({
            'test': split_dataset['test']
        })

    elif mode == 'train':
        dataset_dict = DatasetDict({
            'train': split_trainset['train']
        })

    elif mode == 'validation':
        dataset_dict = DatasetDict({
            'validation': split_trainset['test'],
        })

    else:
        raise TypeError(f"load_and_prepare_data_from_folders cannot return {mode} set")

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    """Step 3: Apply prepare_dataset mapping to dataset_dict"""
    dataset_dict = dataset_dict.map(prepare_dataset,
                                    remove_columns=dataset_dict.column_names["test"] if evaluate else dataset_dict.column_names["train"], num_proc=1)

    logger.info('len validation set: %s', split_trainset['test'].num_rows)
    logger.info('len test set: %s', split_dataset['test'].num_rows)

    # if debug:
    #     """Do some statistics for comparison to ensure correct splits were performed. Alternatively: just
    #        compare the .json eval outputs."""
    #     data_ = 'test' if evaluate else 'validation'
    #     logger.info('Sum of first 3 %s examples divided by total number of %s examples: %.2f',data_,data_,
    #             (sum(dataset_dict[data_]['input_features'][0][0]) +
    #                   sum(dataset_dict[data_]['input_features'][0][1]) +
    #                   sum(dataset_dict[data_]['input_features'][0][2])
    #                    )/dataset_dict[data_].num_rows
    #             )
    return dataset_dict #, split_trainset['train'].num_rows


from dataclasses import dataclass
from typing import Any
import numpy as np
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data Collator for Speech Seq2Seq models with Padding

    We used the collator suggested by the tutorial: https://huggingface.co/blog/fine-tune-whisper.
    We had to slightly modify it due our data being in a different format as required by ray tune.

    Attributes:
       processor (WhisperProcessor): Processor used for padding (normalizing data to same length)
       decoder_start_token_id (int): Token indicating the Beginning Of Setence (BOS)

    Methods:
       __call__(features): Processing a dictionary of input features
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        """ Processing a dictionary of input features.

        Input features are padded to `longest` forms and pytorch tensors are returned.

        Args:
            features (dict): A dictionary with keys 'input_features' consiting of log-Mel features and tokenized
                             'labels'
        Returns"
            batch (dict): Dictionary of padded `input_features` and `labels` as pytorch tensors
        """
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": np.vstack(list(feature))} for feature in features["input_features"]]
        batch = self.processor.feature_extractor.pad(input_features,  padding='longest', return_tensors="pt")

        lab_feat = [{"input_ids": feature} for feature in features["labels"]]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(lab_feat, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


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


def get_datasets_and_collators(args,
                               test_size=0.2):
    """Datasets fetcher for different scenarios

    1. If data is in a folder structure with a .csv overview file
    2. If data is in .h5 only
    3. If data is parquet file(s)

    Returns:
        Dictionary with keys: train, validation, testing
    """
    limit_samples = 100 if args["debug"] else 0
    num_workers = args["cpus_per_trial"]
    seed = args["random_seed"]
    path_to_ds = os.path.join(args["path_to_data"], args["dataset_name"])
    data_mode = get_data_modes(type=args["data_mode"])

    model, feature_extractor, tokenizer, processor = get_whisper_models(args["model_type"],
                                                                        args["target_language"],
                                                                        return_timestamps = args["return_timestamps"],
                                                                        load_in_8bit = args["peft"],
                                                                        local = args["run_on_local_machine"]
                                                                        )
    logger.debug("Tokenizer, Feature Extractor and Processor loaded")

    del model

    ray_datasets = {}
    data_collators = {}
    for dataset_mode, config in data_mode.items():
        logger.info("Getting dataset and collator for %s", dataset_mode)
        if config["type"] == "parquet":
            path_to_ds_ = os.path.join(path_to_ds,dataset_mode,"parquet")
            if os.path.exists(path_to_ds_):
                ds = ray.data.read_parquet(path_to_ds_)
            else:
                raise Exception(f"{path_to_ds_} does not exists for dataset mode {dataset_mode} and type parquet."
                       f"Make sure you save your parquet file in the format {dataset_mode}_parquet")

        elif config["type"] == "h5":
            path_to_ds_ = os.path.join(path_to_ds,"eg_dataset_subset_1000_testing_1337.h5")
                                                            # "dataset_mode+".h5")
            path_to_ds_ = r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_subset_1000_testing_1337.h5"
            if os.path.exists(path_to_ds_):
                ds = create_ray_indexloader(path_to_ds_, samples=limit_samples)
            else:
                raise Exception(f"{path_to_ds_} does not exists for dataset mode {dataset_mode} and type .h5"
                       f"Make sure you save your .h5 in the format {dataset_mode}.h5")

        elif config["type"] == "folder":
            logger.info("Train Test Split of 80 / 20 %")
            if os.path.exists(path_to_ds):
                ds = load_and_prepare_data_from_folders(config["path_to_ds"],
                                                    feature_extractor,
                                                    tokenizer,
                                                    test_size=test_size,
                                                    seed=seed,
                                                    mode=dataset_mode)
            else:
                raise Exception(f"The folder {config["type"]} does not exists for dataset mode {dataset_mode} and type folder"
                       f"Make sure you save your audio files in the folder with a corresponding .csv file")


        if config["collator"] == "parquet":
            collator = collate_parquet

        elif config["collator"] == "streaming":
            collator =  SimpleStreamingCollator(path_to_ds_, feature_extractor, tokenizer,
                                                num_workers=num_workers, copy_to_local=False)

        elif config["collator"] == "folder":
            collator = DataCollatorSpeechSeq2SeqWithPadding

        # pdb.set_trace()
        ray_datasets[dataset_mode] = ds.limit(limit_samples) if limit_samples>0 else ds
        data_collators[dataset_mode] = collator

    return ray_datasets, data_collators

#!/usr/bin/env python
# Simplified Ray Dataset Creation for Whisper Fine-tuning
"""
This script preprocesses a Whisper fine-tuning corpus stored in a single HDF5 file into
a set of sharded Parquet files using Ray Datasets, then demonstrates loading those
Parquet shards and iterating them as PyTorch batches.

Why this exists
---------------
Reading audio + text from HDF5 and running Whisper preprocessing (mel features + tokenization)
is CPU-heavy and I/O-heavy. Doing it online during training can bottleneck GPUs. This script
"materializes" preprocessing into Parquet so training can later read already-prepared arrays.

Key components
--------------
1) Shared-HDF5 multiprocessing loader
   - `_init_worker(hdf5_path)` opens the HDF5 file once per worker process and stores it in a
     process-global `_shared_hdf5`.
   - `_process_index_shared(idx)` reads one sample (audio array + transcription string) from
     that shared handle and returns (idx, audio, transcription).
   This avoids re-opening the HDF5 file for every sample and enables parallel loading.

2) `SimpleStreamingCollator`
   A custom collator that turns a batch of indices into model-ready tensors:
   - Input: a batch dict with column "idx" (e.g., {"idx": np.ndarray([...])})
   - Loads audio + transcription for each idx:
       * single-process mode if num_workers == 0 (keeps one h5py.File handle open)
       * multi-process mode otherwise (multiprocessing.Pool with per-worker shared HDF5 handle)
   - Runs Whisper feature extraction:
       * `feature_extractor(audio, sampling_rate=16000)` to produce log-mel features
       * pads features to the longest in the batch
   - Tokenizes transcriptions with the Whisper tokenizer, pads/truncates labels to a fixed
     max length (448), and masks padding tokens to -100 for loss computation.
   - Returns a dict of torch tensors:
       {"input_features": (B, ...), "labels": (B, 448)}

   The collator also logs rolling throughput every 5 batches.

3) Index sharding to avoid OOM
   - `create_index_shards(total_samples, num_shards=10)` creates multiple small Ray Datasets,
     each containing only {"idx": j} rows for a range of indices.
   - The main preprocessing loop processes each shard separately and writes it to:
       <output_path>/shard_<k>/
     This avoids building one huge in-memory indices list / dataset and makes output easier
     to manage.

4) Ray preprocessing pipeline
   For each shard:
   - `map_batches(process_batch, batch_size=..., batch_format="numpy")`:
       * Ray provides batches as dict-of-numpy-arrays (columnar), including "idx".
       * `process_batch` calls the collator to compute tensors.
       * Tensors are converted to NumPy arrays and attached back onto the batch dict under
         "input_features" and "labels".
   - The resulting dataset is repartitioned (currently to 1 file per shard) and written as
     Parquet.

5) Reload + iterate as PyTorch
   - `load_ray_dataset()` reads multiple shard Parquet directories into one Ray Dataset and
     optionally shuffles.
   - `get_torch_iterator()` uses `iter_torch_batches()` with a collate_fn that stacks the
     per-row NumPy arrays into torch tensors, yielding dicts with keys expected by Whisper
     training ("input_features", "labels").

Current state / assumptions
---------------------------
- Audio is assumed to be 16 kHz (hard-coded sampling_rate=16000).
- Labels are padded/truncated to max_length=448.
- Output is written as one Parquet partition per shard (`repartition(1)`), i.e. 10 Parquet
  files total if num_shards=10.
- The code contains remnants of earlier versions (duplicate imports, unused modules, and
  an imported `SimpleStreamingCollator` that is later redefined locally).
"""

import sys
from pathlib import Path

# Make src/ importable so sibling packages resolve whether this file is run via
# ``python -m prepare_data.materialize_dataset`` or directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.whisper_models import get_whisper_models_from_dir

import math

import h5py
import ray


import multiprocessing
import time
import numpy as np
import os
import shutil

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
        labels_batch = self.tokenizer.pad(label_features,
                                          padding="max_length",
                                          max_length=448,
                                          return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        return {"input_features": input_features, "labels": labels}

    def cleanup(self):
        """Explicitly clean up resources - call this before exiting."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        # Still have a __del__ as a fallback, but make it safe
        try:
            self.cleanup()
        except Exception:
            pass


def create_index_shards(total_samples, num_shards=10):
    """
    Create 10 equally sized Ray datasets of indices without causing OOM.

    Args:
        total_samples (int): Total number of samples
        num_shards (int, optional): Number of shards to create. Defaults to 10.

    Returns:
        List of Ray datasets, each containing indices for a shard
    """
    # Calculate samples per shard
    samples_per_shard = math.ceil(total_samples / num_shards)

    # Create sharded datasets
    sharded_datasets = []
    for i in range(num_shards):
        # Calculate start and end indices for this shard
        start_idx = i * samples_per_shard
        end_idx = min((i + 1) * samples_per_shard, total_samples)

        # Create dataset for this shard
        shard_indices = [{"idx": j} for j in range(start_idx, end_idx)]
        shard_ds = ray.data.from_items(shard_indices)
        sharded_datasets.append(shard_ds)

    return sharded_datasets


def create_ray_dataset(
        hdf5_path,
        output_path,
        model_type='openai/whisper-large-v3',
        batch_size=32,
        num_workers=8
):
    """
    Create and save a materialized Ray dataset using the existing SimpleStreamingCollator.

    Args:
        hdf5_path: Path to the HDF5 file containing audio and transcription data
        output_path: Path to save the Ray dataset
        processor_name: Name of the Whisper processor to use
        batch_size: Batch size for preprocessing
        num_workers: Number of workers for parallel processing
    """
    print(f"Initializing processor components from {model_type}")
    model, feature_extractor, tokenizer, processor = get_whisper_models_from_dir(model_type, 'de', return_timestamps=False,
                                                                              load_in_8bit=False)

    # Initialize the existing collator
    collator = SimpleStreamingCollator(
        hdf5_path=hdf5_path,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        num_workers=0
    )

    # Get the total number of samples from the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        total_samples = len(f['audio'])

    print(f"Processing {total_samples} samples with batch size {batch_size}")

    # total_samples = 500

    # Example usage
    # total_samples = 20131
    index_shards = create_index_shards(total_samples, num_shards=100)

    # Print out shard information
    for shard_idx, shard in enumerate(index_shards):
        print(f"Shard {shard_idx}: {shard.count()} indices")

        #        if shard_idx < 18:
        #            continue
        # Create a Ray dataset of indices
        indices_ds = shard  # ray.data.from_items([{"idx": i} for i in range(total_samples)])

        # Define a processor function that uses the collator to process a batch
        def process_batch(batch_dict):

            # Process the batch using the existing collator
            try:
                processed_batch = collator(batch_dict)
                print(f"Collator successful for batch with {len(batch_dict)} indices")
            except Exception as e:
                print(f"Error collating batch: {e}")
                return {"item": []}  # Return empty item list on error

            batch_dict["input_features"] = processed_batch["input_features"].numpy()
            batch_dict["labels"] = processed_batch["labels"].numpy()

            return batch_dict


        # Process the dataset in batches
        start_time = time.time()
        processed_ds = indices_ds.map_batches(
            process_batch,
            batch_size=batch_size,
            num_cpus=1, #num_workers
            batch_format="numpy"
        )

        # Materialize and save the dataset
        print(f"Materializing and saving dataset to {output_path}")
        # Control the number of partitions (parquet files)
        #    num_partitions = 1 # Adjust this number as needed
        shard_path = os.path.join(output_path, 'shard_' + str(shard_idx))
        if not os.path.exists(shard_path):
            os.makedirs(shard_path)

        processed_ds = processed_ds.repartition(1)
        processed_ds.write_parquet(shard_path)

        elapsed = time.time() - start_time
        print(f"Processing complete! {total_samples} samples in {elapsed:.2f}s")
        print(f"Average processing speed: {total_samples / elapsed:.2f} samples/sec")


#        return processed_ds


def load_ray_dataset(dataset_path, shuffle=False):
    """
    Load a preprocessed Ray dataset.

    Args:
        dataset_path: Path to the saved Ray dataset
        shuffle: Whether to shuffle the dataset

    Returns:
        The loaded Ray dataset
    """
    print(f"Loading dataset from {dataset_path}")

    dataset_paths = [os.path.join(dataset_path, 'shard_' + shard_idx) for shard_idx in range(100)]
    ds = ray.data.read_parquet(dataset_paths)

    if shuffle:
        ds = ds.random_shuffle()

    return ds


def get_torch_iterator(dataset, batch_size=16):
    """
    Create a PyTorch iterator from a Ray dataset.

    Args:
        dataset: Ray dataset
        batch_size: Batch size for batching

    Returns:
        PyTorch batch iterator
    """

    def collate_fn(batch):
        input_features_batch = torch.stack([torch.from_numpy(x) for x in batch["input_features"]])
        labels_batch = torch.stack([torch.from_numpy(x) for x in batch["labels"]])

        return {
            "input_features": input_features_batch,  # input_features_batch,
            "labels": labels_batch  # labels_batch
        }


    # # Return PyTorch iterator
    return dataset.iter_torch_batches(
        batch_size=batch_size,
        collate_fn=collate_fn
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Materialize an HDF5 corpus (audio + transcription) into sharded "
                    "Parquet files of pre-computed Whisper features (log-mel + token ids). "
                    "Run one split at a time.",
    )
    ap.add_argument("--hdf5_path", required=True,
                    help="Input HDF5 corpus with 'audio' and 'transcription' datasets, "
                         "e.g. $DATA_PATH/eg_dataset_complete_v3_train.h5")
    ap.add_argument("--output_path", required=True,
                    help="Output directory root; the --split sub-directory is created underneath.")
    ap.add_argument("--split", default="train_parquet",
                    help="Split sub-directory to write (train_parquet / val_parquet / test_parquet).")
    ap.add_argument("--model_type", default="whisper-large-v3",
                    help="Model sub-directory under MODEL_PATH whose feature extractor + "
                         "tokenizer are used for preprocessing (e.g. whisper-large-v3).")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    ray.init()
    create_ray_dataset(
        hdf5_path=args.hdf5_path,
        model_type=args.model_type,
        output_path=os.path.join(args.output_path, args.split),
        batch_size=args.batch_size,
    )
    ray.shutdown()


if __name__ == "__main__":
    main()

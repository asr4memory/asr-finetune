#!/usr/bin/env python
# Ray Dataset with Direct mmap Processing for Whisper

import os
import time
import h5py
import ray
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm


@ray.remote
class HDF5Worker:
    """Ray actor for memory-mapped HDF5 access"""

    def __init__(self, hdf5_path, processor_name="openai/whisper-small"):
        # Open HDF5 file in read-only mode (shared across all worker methods)
        self.hdf5_file = h5py.File(hdf5_path, "r")

        # Initialize processors
        self.processor = WhisperProcessor.from_pretrained(processor_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(processor_name)

    def process_sample(self, idx, max_label_length=448):
        """Process a single sample by index"""
        try:
            # Read audio and transcription using memory-mapping
            audio = np.array(self.hdf5_file['audio'][idx], dtype=np.float32)
            transcription = self.hdf5_file['transcription'][idx]

            # Decode transcription if needed
            if isinstance(transcription, bytes):
                transcription = transcription.decode('utf-8')

            # Extract features
            features = self.feature_extractor(audio, sampling_rate=16000)
            input_features = features.input_features[0]

            # Tokenize transcription with fixed max length
            tokenized = self.tokenizer(transcription, padding="max_length",
                                       max_length=max_label_length, return_tensors="pt")

            # Create labels with proper masking
            labels = tokenized.input_ids[0].numpy()
            attention_mask = tokenized.attention_mask[0].numpy()
            labels = np.where(attention_mask == 1, labels, -100)

            # Return as serializable dict
            return {
                "idx": idx,
                "input_features": input_features.tobytes(),
                "input_features_shape": input_features.shape,
                "input_features_dtype": str(input_features.dtype),
                "labels": labels.tobytes(),
                "labels_shape": labels.shape,
                "labels_dtype": str(labels.dtype)
            }
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return {"idx": idx, "error": str(e)}

    def process_batch(self, indices, max_label_length=448):
        """Process a batch of samples by indices"""
        results = []
        for idx in indices:
            result = self.process_sample(idx, max_label_length)
            if "error" not in result:
                results.append(result)
        return results

    def __del__(self):
        """Clean up when actor is destroyed"""
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()


def create_ray_dataset(
        hdf5_path,
        output_path,
        processor_name="openai/whisper-large-v3",
        batch_size=32,
        num_actors=4,
        max_label_length=448
):
    """
    Create a materialized Ray dataset using memory-mapped HDF5 access.

    Args:
        hdf5_path: Path to the HDF5 file with audio data
        output_path: Path to save the Ray dataset
        processor_name: Name of the Whisper processor
        batch_size: Batch size for processing
        num_actors: Number of Ray actors to create (defaults to CPU count)
        max_label_length: Maximum label length for padding
    """
    # Get dataset size from HDF5
    with h5py.File(hdf5_path, "r") as f:
        total_samples = len(f['audio'])

    # Determine number of actors
    if num_actors is None:
        import multiprocessing
        num_actors = min(8, multiprocessing.cpu_count())

    print(f"Processing {total_samples} samples with {num_actors} actors")

    total_samples = 512
    # Create Ray actors for parallel processing
    actors = [HDF5Worker.remote(hdf5_path, processor_name) for _ in range(num_actors)]

    # Create batches of indices
    batches = []
    for i in range(0, total_samples, batch_size):
        batches.append(list(range(i, min(i + batch_size, total_samples))))

    # Process batches in parallel using Ray actors
    start_time = time.time()

    # Submit tasks to actors in a round-robin fashion
    tasks = []
    for i, batch in enumerate(batches):
        actor_idx = i % num_actors
        tasks.append(actors[actor_idx].process_batch.remote(batch, max_label_length))

    # Create progress bar
    pbar = tqdm(total=len(batches), desc="Processing batches")

    # Collect results and build dataset incrementally
    all_samples = []
    for task_result in ray.get(tasks):
        all_samples.extend(task_result)
        pbar.update(1)

    pbar.close()

    # Create Ray dataset from processed samples
    ds = ray.data.from_items(all_samples)

    # Save to disk
    print(f"Saving dataset to {output_path}")
    ds.write_parquet(output_path)

    # Print stats
    elapsed = time.time() - start_time
    processed_count = ds.count()
    print(f"Processed {processed_count} samples in {elapsed:.2f}s")
    print(f"Processing speed: {processed_count / elapsed:.2f} samples/sec")

    return ds


def get_torch_iterator(dataset_path, batch_size=16, shuffle=True):
    """
    Load a dataset and create a PyTorch iterator.

    Args:
        dataset_path: Path to the saved Ray dataset
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
    """
    # Load the dataset
    ds = ray.data.read_parquet(dataset_path)
    print(f"Loaded dataset with {ds.count()} samples")

    # Shuffle if requested
    if shuffle:
        ds = ds.random_shuffle()

    # Define a collate function to reconstruct tensors
    def collate_fn(batch):
        input_features_list = []
        labels_list = []

        # Reconstruct tensors for each sample
        for item in batch:
            # Reconstruct input features
            input_shape = item["input_features_shape"]
            input_dtype = eval("np." + item["input_features_dtype"].split(".")[-1])
            input_features = np.frombuffer(item["input_features"], dtype=input_dtype).reshape(input_shape)

            # Reconstruct labels
            labels_shape = item["labels_shape"]
            labels_dtype = eval("np." + item["labels_dtype"].split(".")[-1])
            labels = np.frombuffer(item["labels"], dtype=labels_dtype).reshape(labels_shape)

            # Add to lists
            input_features_list.append(torch.tensor(input_features))
            labels_list.append(torch.tensor(labels))

        # Find max dimensions for padding input features (they may have different lengths)
        max_input_len = max(tensor.size(0) for tensor in input_features_list)
        feature_dim = input_features_list[0].size(1)

        # Create padded tensors for input features
        padded_inputs = torch.zeros(len(batch), max_input_len, feature_dim)

        # Fill with data
        for i, inputs in enumerate(input_features_list):
            padded_inputs[i, :inputs.size(0), :] = inputs

        # Stack labels (they should all have the same fixed length)
        labels_batch = torch.stack(labels_list)

        return {
            "input_features": padded_inputs,
            "labels": labels_batch
        }

    # Create iterator
    iterable = ds.iter_torch_batches(
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    return iter(iterable)  # Convert to iterator for next() support


def main():
    # Initialize Ray
    ray.init()

    try:
        # Define paths
        output_path = r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_complete_v3_sharded/val/"
        hdf5_path = os.path.join(output_path, r"eg_dataset_complete_v3_val.h5")

        dataset_path = output_path  # Output path

        # Create the dataset if it doesn't exist
        # if not os.path.exists(dataset_path):
        create_ray_dataset(
            hdf5_path=hdf5_path,
            output_path=dataset_path,
            batch_size=16,
            max_label_length=448,
            num_actors=None,
        )

        # Get torch iterator
        torch_iter = get_torch_iterator(dataset_path, batch_size=16)

        # Test the iterator
        try:
            batch = next(torch_iter)
            print(f"Successfully loaded batch:")
            print(f"  Input features shape: {batch['input_features'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")

            # Example training code
            print("\nExample training code:")
            print("""
            # In your training loop:
            for batch in torch_iter:
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            """)

        except Exception as e:
            print(f"Error loading batch: {e}")

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
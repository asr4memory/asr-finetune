#!/usr/bin/env python
# Simplified Ray Dataset Creation for Whisper Fine-tuning

import os
import pdb
import time
import h5py
import numpy as np
import ray
import torch
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from tqdm import tqdm
from typing import Dict

# Import the existing collator class
from utils import SimpleStreamingCollator
from models import get_whisper_models as get_models, get_whisper_models_local

def create_ray_dataset(
        hdf5_path,
        output_path,
        model_type= 'openai/whisper-large-v3',
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
    # processor = WhisperProcessor.from_pretrained(processor_name)
    # feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_name)
    # tokenizer = WhisperTokenizer.from_pretrained(processor_name)
    model, feature_extractor, tokenizer, processor = get_whisper_models_local(model_type, 'de',
                                                                              return_timestamps=False,
                                                                              load_in_8bit=False)

    # Initialize the existing collator
    collator = SimpleStreamingCollator(
        hdf5_path=hdf5_path,
        processor=processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        num_workers=num_workers
    )

    # Get the total number of samples from the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        total_samples = len(f['audio'])

    print(f"Processing {total_samples} samples with batch size {batch_size}")

    total_samples = 50
    # Create a Ray dataset of indices
    indices_ds = ray.data.from_items([{"idx": i} for i in range(total_samples)])

    # Define a processor function that uses the collator to process a batch
    def process_batch(batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # if isinstance(batch_dict, dict) and "idx" in batch_dict:
        #     # If batch_format="native" or batch_dict is already a dict with 'idx' key
        #     if isinstance(batch_dict["idx"], list):
        #         # Multiple indices in a single batch dict
        #         batch_indices = batch_dict["idx"]
        #     else:
        #         # Single index
        #         batch_indices = [batch_dict["idx"]]
        # elif isinstance(batch_dict, list):
        #     # If batch_dict is a list of dicts (default format)
        #     batch_indices = [item["idx"] for item in batch_dict]
        # elif hasattr(batch_dict, "idx"):
        #     # If batch_format="pandas"
        #     batch_indices = batch_dict.idx.tolist()
        # else:
        #     # Log the type to help diagnose
        #     print(f"Unexpected batch_dict type: {type(batch_dict)}")
        #     if isinstance(batch_dict, dict):
        #         print(f"Keys: {batch_dict.keys()}")
        #     # Return empty results when format is unknown
        #     return {"results": []}
        #
        # batch_indices = [item["idx"] for item in batch_dict]
        #
        # Prepare a batch input for the collator
        # batch_input = {"idx": batch_indices}

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
        # try:# Convert the processed batch to a list of individual samples
        #     results = {}
        #     results_list = []
        #     # pdb.set_trace()
        #     for i, idx in enumerate(batch_dict["idx"]):
        #         # Handle potential padding differences
        #         # if i < processed_batch["input_features"].shape[0] and i < processed_batch["labels"].shape[0]:
        #
        #             # Convert numpy arrays to bytes for storage
        #         input_features_bytes = processed_batch["input_features"][i].numpy() #.tobytes()
        #         labels_bytes = processed_batch["labels"][i].numpy() #.tobytes()
        #
        #         # Store the shapes for reconstruction
        #         input_shape = processed_batch["input_features"][i].shape
        #         labels_shape = processed_batch["labels"][i].shape
        #
        #
        #         # Create a dict with serialized arrays
        #         # results_list.append({
        #         #     "idx": idx,
        #         #     "input_features_bytes": input_features_bytes,
        #         #     "input_features_shape": str(input_shape),  # Store as string
        #         #     "input_features_dtype": str(processed_batch["input_features"][i].dtype),
        #         #     "labels_bytes": labels_bytes,
        #         #     "labels_shape": str(labels_shape),
        #         #     "labels_dtype": str(processed_batch["labels"][i].dtype)
        #         # })
        #
        #             # Convert tensors to numpy for storage
        #             # results["item"] = [
        #             #     "idx": batch_indices[i],
        #             #     "input_features": processed_batch["input_features"][i].numpy(),
        #             #     "labels": processed_batch["labels"][i].numpy()
        #             # }
        #             # results_list.append({
        #             #     "idx": batch_indices[i],
        #             #     "input_features": processed_batch["input_features"][i].numpy(),
        #             #     "labels": processed_batch["labels"][i].numpy()
        #             # })
        #
        #     # results["item"] = results_list
        #     print(f"Successfully processed {len(results_list)}/{i+1} samples in batch")
        #     return {"results": results_list}
        #     # return {"item": results_list}
        #         # results) #{"results": results}
        # except Exception as e:
        #

            # print(f"Error processing batch with indices {batch_dict}: {e}")
            # return {"item": [}

    # pdb.set_trace()
    # batch_idx = {'idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    #          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    #          56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    #          83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
    #          108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125]}
    # process_batch(batch_idx)
    # pdb.set_trace()
    # Process the dataset in batches
    start_time = time.time()
    processed_ds = indices_ds.map_batches(
        process_batch,
        batch_size=batch_size,
        num_cpus=num_workers,
        batch_format="numpy"
    )

    pdb.set_trace()
    # Flatten the nested results field
    # pdb.set_trace()
    # processed_ds = processed_ds.flat_map(lambda x: x["item"])
    # processed_ds = processed_ds.flat_map(lambda x: [{"item": item} for item in x["results"]]).map(lambda x: x["item"])
    # pdb.set_trace()
    # Materialize and save the dataset
    print(f"Materializing and saving dataset to {output_path}")
    num_partitions = 1  # Adjust this number as needed
    processed_ds = processed_ds.repartition(num_partitions)
    processed_ds.write_parquet(output_path)
    # pdb.set_trace()
    elapsed = time.time() - start_time
    print(f"Processing complete! {total_samples} samples in {elapsed:.2f}s")
    print(f"Average processing speed: {total_samples / elapsed:.2f} samples/sec")

    return processed_ds


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
    ds = ray.data.read_parquet(dataset_path)

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
    # pdb.set_trace()
    def collate_fn(batch):
        input_features_list = []
        labels_list = []
        print('batch keys', batch.keys())
        # pdb.set_trace()
        # for key, value in batch.items():
            # print(item)
            # Reconstruct input features
            # input_shape = eval(item["input_features_shape"])  # Convert string to tuple
            # input_dtype = eval("np." + item["input_features_dtype"].split(".")[-1])
            # input_features = np.frombuffer(item["input_features_bytes"], dtype=input_dtype).reshape(input_shape)
            #
            # # Reconstruct labels
            # labels_shape = eval(item["labels_shape"])
            # labels_dtype = eval("np." + item["labels_dtype"].split(".")[-1])
            # labels = np.frombuffer(item["labels_bytes"], dtype=labels_dtype).reshape(labels_shape)

            # Convert to tensors
            # input_features_list.append(torch.tensor(item["input_features"]))
            # labels_list.append(torch.tensor(item["labels"]))

        # pdb.set_trace()
        # Stack tensors
        input_features_batch = torch.stack([torch.from_numpy(x) for x in batch["input_features"]])
        labels_batch = torch.stack([torch.from_numpy(x) for x in batch["labels"]])
        # labels_batch = torch.stack(labels_list)

        return {
            "input_features": input_features_batch, #input_features_batch,
            "labels": labels_batch# labels_batch
        }

    # def collate_fn(batch):
    #     # Extract input features and labels
    #     input_features = torch.stack([torch.tensor(item["input_features"]) for item in batch["results"]])
    #     labels = torch.stack([torch.tensor(item["labels"]) for item in batch["results"]])
    #
    #     return {
    #         "input_features": input_features,
    #         "labels": labels
    #     }

    # # Return PyTorch iterator
    return dataset.iter_torch_batches(
        batch_size=batch_size,
        collate_fn=collate_fn
    )


def main():
    # Initialize Ray
    ray.init()

    # Define paths
    output_path = r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_complete_v3_sharded/val/"
    hdf5_path = os.path.join(output_path,r"eg_dataset_complete_v3_val.h5")  # Replace with your HDF5 path

    # "path/to/output/whisper_dataset"  # Replace with output path

    # Create and save the dataset
    # create_ray_dataset(
    #     hdf5_path=hdf5_path,
    #     output_path=os.path.join(output_path,'parquet/'),
    #     batch_size = 16
    # )
    #
    # Example of loading and using the dataset
    ds = load_ray_dataset(os.path.join(output_path,'parquet/'))

    # Get a PyTorch iterator
    torch_iter = get_torch_iterator(ds, batch_size=1)

    # pdb.set_trace()
    # Example of using the iterator
    for batch in torch_iter:
    # batch = next(iter(torch_iter))
        print(f"Input features shape: {torch.mean(batch['input_features'])}")
        # print(f"Labels shape: {torch.mean(batch['labels'])}")

    # Ray shutdown
    ray.shutdown()


if __name__ == "__main__":
    main()
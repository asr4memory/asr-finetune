import os
from utils import create_ray_indexloader
import pdb
import h5py
import torch
import ray
from typing import Dict, Any, Callable
from functools import partial
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
import numpy as np

# class H5Dataset(torch.utils.data.Dataset):
#     def __init__(self, path):
#         self.file_path = path
#         self.dataset = None
#         with h5py.File(self.file_path, 'r') as file:
#             self.dataset_len = len(file["audio"])
#
#     def __getitem__(self, idx):
#         if self.dataset is None:
#             audio = h5py.File(self.file_path, 'r')["audio"][idx]
#             trans = h5py.File(self.file_path, 'r')["transcription"][idx].decode('utf-8')
#         return audio, trans
#
#     def __len__(self):
#         return self.dataset_len

# Improved H5Dataset that handles file opening more efficiently
class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file_path = path
        # Just open once to get length
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["audio"])
        # We'll open the file in each worker properly

    def __getitem__(self, idx):
        # Open the file handle on first access in each worker process
        if not hasattr(self, '_file_handle'):
            self._file_handle = h5py.File(self.file_path, 'r')

        # Get data from already opened file handle
        audio = np.array(self._file_handle["audio"][idx], dtype=np.float32)
        trans = self._file_handle["transcription"][idx]
        if isinstance(trans, bytes):
            trans = trans.decode('utf-8')

        return {"idx": idx, "audio": audio, "transcription": trans}

    def __len__(self):
        return self.dataset_len

    def __del__(self):
        # Clean up file handle if it exists
        if hasattr(self, '_file_handle'):
            self._file_handle.close()

def hdf5_generator(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        for i in range(len(f["audio"])):
            audio = np.array(f["audio"][i], dtype=np.float32)
            trans = f["transcription"][i]
            if isinstance(trans, bytes):
                trans = trans.decode("utf-8")
            yield {"idx": i, "audio": audio, "transcription": trans}

# Feature extraction function for Ray Data transformation
def extract_features(batch: Dict[str, Any],
                     feature_extractor: Callable,
                     sampling_rate: int = 16000) -> Dict[str, Any]:
    """
    Process a batch of audio samples and extract features

    Args:
        batch: Dictionary containing "audio" arrays and "transcription" strings
        feature_extractor: The feature extractor function to use
        sampling_rate: The sampling rate of the audio

    Returns:
        Dictionary with extracted features added
    """
    results = []
    # pdb.set_trace()
    # Process each sample in the batch
    for i in range(len(batch["item"])):
        audio = batch["item"]["audio"][i]

        # Extract audio features
        features = feature_extractor(audio, sampling_rate=sampling_rate)

        # Create result dictionary
        result = {
            "idx": batch["item"]["idx"][i],
            "audio": audio,
            "transcription": batch["item"]["transcription"][i],
            "input_features": features.input_features[0]  # Assumes this is the format
        }
        results.append(result)

    # Convert list of dicts to dict of lists (Ray Dataset format)
    return {k: [d[k] for d in results] for k in results[0].keys()}


# Tokenization function for Ray Data transformation
def tokenize_transcriptions(batch: Dict[str, Any],
                            tokenizer: Callable) -> Dict[str, Any]:
    """
    Tokenize transcriptions in the batch

    Args:
        batch: Dictionary containing "transcription" strings
        tokenizer: The tokenizer to use

    Returns:
        Dictionary with tokenized transcriptions added
    """
    transcriptions = batch["transcription"]

    # Tokenize all transcriptions
    tokenized = [tokenizer(text).input_ids for text in transcriptions]

    # Add to batch
    batch["tokenized_transcription"] = tokenized

    return batch

def preprocess_batch(batch, feature_extractor, tokenizer):
    # print("Batch sample:", batch)
    # pdb.set_trace()
    # batch = batch["item"]
    # batch: list of dicts
    audios = batch["audio"] #[row["audio"] for row in batch]
    transcriptions = batch["transcription"].tolist() #[row["transcription"] for row in batch]

    # Feature extractiond
    feats = feature_extractor(audios, sampling_rate=16000, padding="longest", return_tensors="pt")
    input_features = feats["input_features"]

    # Tokenization
    tokenized = tokenizer(transcriptions, padding="longest", return_tensors="pt")
    labels = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    # Apply masking directly with PyTorch operations (more efficient)
    masked_labels = labels.masked_fill(attention_mask == 0, -100)
    # masked_labels = np.where(attention_mask == 1, labels, -100)

    # Apply -100 masking
    # labels = np.where(attention_mask == 1, labels, -100)

    return  {"input_features": input_features, "labels": masked_labels}
        # for feat, label in zip(input_features, masked_labels)


class FinalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Convert lists back to torch.Tensors
        input_features = [torch.tensor(example["input_features"]) for example in batch]
        labels = [torch.tensor(example["labels"]) for example in batch]

        # Pad features (assuming you're using a HuggingFace processor)
        padded_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_features": padded_features,
            "labels": padded_labels
        }

def h5_item_generator(path):
    with h5py.File(path, 'r') as f:
        total = len(f['audio'])
        print("Total: ",total)
        items = []
        for i in range(total):
            audio = np.array(f["audio"][i], dtype=np.float32)
            trans = f["transcription"][i]
            if isinstance(trans, bytes):
                trans = trans.decode('utf-8')
            items += [{"idx": i, "audio": audio, "transcription": trans}]
        return items


# path_to_data = os.path.join("/scratch/usr/", os.getenv('USER') + "/data") if args.path_to_data is None else args.path_to_data
#
# h5_path = os.path.join(path_to_data, args.dataset_name + ".h5")  # "eg_dataset_complete_5sec.h5")
# train_h5_path =os.path.join(path_to_data, args.dataset_name + "_train.h5")
# val_h5_path = os.path.join(path_to_data, args.dataset_name + "_train.h5")
model_type = 'openai/whisper-large-v3'
target_language = 'de'

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
tokenizer = WhisperTokenizer.from_pretrained(model_type, language=target_language, task="transcribe")

train_h5_path = r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_complete_v3_sharded/train/eg_dataset_complete_v3_train.h5"
    # r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_subset_1000_training_1337.h5"
# ray_ds = ray.data.from_items(h5_item_generator(train_h5_path))

ray_ds = create_ray_indexloader(train_h5_path)
from utils import SimpleStreamingCollator

collator = SimpleStreamingCollator(train_h5_path,feature_extractor, tokenizer)
ds_iter = ray_ds.iter_torch_batches(batch_size = 2,collate_fn=collator)

pdb.set_trace()

# preprocess_fn = partial(preprocess_batch, feature_extractor=feature_extractor, tokenizer=tokenizer)


# torchH5 = H5Dataset(train_h5_path)
# ray_ds = ray.data.from_torch(torchH5)
# pdb.set_trace()
# ds = ray.data.from_iterable(hdf5_generator(train_h5_path))


# Wrap for Ray


processed = ray_ds.limit(10).map_batches(preprocess_fn, batch_size=5, batch_format="default")
ds_iter = processed.iter_torch_batches(batch_size = 2) #,collate_fn=FinalDataCollator(tokenizer))

pdb.set_trace()

for batch in ds_iter:
    print("Batch keys:", batch["input_features"].shape)
    break
pdb.set_trace()
# Apply preprocessing in parallel batches
processed_ds = ds.map_batches(preprocess_fn, batch_size=32, num_gpus=1)
pdb.set_trace()

num_workers = 4
# Apply feature extraction
ray_ds = ray_ds.map_batches(
    partial(extract_features, feature_extractor=feature_extractor),
    batch_size=16,  # Adjust based on your memory constraints
    num_cpus=num_workers
)

pdb.set_trace()
# Apply tokenization
ray_ds = ray_ds.map_batches(
    partial(tokenize_transcriptions, tokenizer=tokenizer),
    batch_size=16,  # Can use larger batch size for tokenization
    num_cpus=num_workers
)

# Hack: Ray Tune requires a ray dataset object. However, converting log-mel spectogram formatare not supported
# So we create a index ray dataset and the actual data-fetching is Wrapped in the
# train_loader = create_ray_indexloader(train_h5_path)
# dataset_size = train_loader.count()
#
# train_ds_iterable = train_loader.iter_torch_batches(batch_size = 2)



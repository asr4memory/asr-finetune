import pdb

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from transformers import WhisperFeatureExtractor, WhisperTokenizer

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


class WhisperHDF5TorchDataset(Dataset):
    def __init__(
            self,
            hdf5_path: str,
            feature_extractor: "WhisperFeatureExtractor",
            tokenizer: "WhisperTokenizer",
            sampling_rate: int = 16000,
            limit: Optional[int] = None,
            debug: bool = False
    ):
        self.hdf5_path = hdf5_path
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate
        self.debug = debug
        self.dataset = None

        # Open file once to count samples
        with h5py.File(self.hdf5_path, "r") as f:
            self.total_samples = len(f["audio"])
        self.limit = limit if limit is not None else self.total_samples

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        # if self.dataset is None:
        #     self.dataset = h5py.File(self.hdf5_path,'r')
        with h5py.File(self.hdf5_path, "r") as f:
            audio = f["audio"][idx]  # [()]  # Use [()] to read the entire dataset into memory
            text = f["transcription"][idx]  # [()]
        #     self.dataset = h
        # audio = self.dataset["audio"][idx] #[()]  # Use [()] to read the entire dataset into memory
        # text = self.dataset["transcription"][idx] #[()]
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        # Ensure audio is a numpy array with correct dtype
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Process audio features
        input_features = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors=None  # Don't convert to tensor yet
        ).input_features[0]

        # Apply fixed padding to ensure consistent dimensions
        padded = np.zeros((input_features.shape[0], 3000), dtype=np.float32)
        length = min(input_features.shape[1], 3000)
        padded[:, :length] = input_features[:, :length]

        # Convert to tensor after padding
        input_features_tensor = torch.tensor(padded, dtype=torch.float32)

        # Process text with fixed padding
        tokenized = self.tokenizer(text, padding="max_length", max_length=448, truncation=True)

        # Convert to tensors
        labels_tensor = torch.tensor(tokenized.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(tokenized.attention_mask, dtype=torch.long)

        # Create attention mask for labels (important for loss calculation)
        labels = labels_tensor.masked_fill(attention_mask.ne(1), -100)

        return {
            "input_features": input_features_tensor,
            "labels": labels
        }


class WhisperHDF5TorchDataset2(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        sampling_rate: int = 16000,
        max_len: int = 3000,
        limit: int = None,
        debug: bool = False
    ):
        self.hdf5_path = hdf5_path
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate
        self.max_len = max_len
        self.debug = debug

        # Open file once to count samples
        with h5py.File(self.hdf5_path, "r") as f:
            self.total_samples = len(f["audio"])
        self.limit = limit if limit is not None else self.total_samples

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, "r") as f:
            audio = f["audio"][idx]
            text = f["transcription"][idx]
            if isinstance(text, bytes):
                text = text.decode("utf-8")

        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)

        # Process audio features
        input_features = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors=None  # Don't convert to tensor yet
        ).input_features[0]

        # Process text - we need to handle padding consistently
        tokenized = self.tokenizer(text, padding="max_length", max_length=448)

        # Convert to tensors
        input_features_tensor = torch.tensor(input_features, dtype=torch.float32)
        labels_tensor = torch.tensor(tokenized.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(tokenized.attention_mask, dtype=torch.long)

        # Create attention mask for labels (important for loss calculation)
        labels = labels_tensor.masked_fill(attention_mask.ne(1), -100)

        return {
            "input_features": input_features_tensor,
            "labels": labels
        }

        # # Feature extraction
        # features = self.feature_extractor(
        #     audio,
        #     sampling_rate=self.sampling_rate,
        #     return_tensors="pt"
        # ).input_features[0]
        #
        # # Tokenization
        # tokenized_labels = self.tokenizer(text, return_tensors="pt")
        #
        # # Return properly formatted features and labels
        # return {
        #     "input_features": features,  # Already a PyTorch tensor
        #     "labels": tokenized_labels.input_ids.squeeze(0)  # Get rid of batch dimension
        # }
        # Feature extraction
        # features = self.feature_extractor(audio, sampling_rate=self.sampling_rate).input_features[0]
        # # padded = np.zeros((80, self.max_len), dtype=np.float32)
        # # length = min(features.shape[1], self.max_len)
        # # padded[:, :length] = features[:, :length]
        # padded_features = self.feature_extractor.pad(
        #     {"input_features": features},
        #     padding="longest",
        #     return_tensors="pt"
        # )
        # input_features = padded_features.input_features.squeeze(0)
        #
        # # Tokenization
        # tokenized_labels = self.tokenizer(text, padding="longest", return_tensors="pt")
        # label_features = [{"input_ids": ids} for ids in tokenized_labels]
        # labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        # labels = labels_batch["input_ids"].masked_fill(
        #     labels_batch.attention_mask.ne(1), -100
        # )
        # # pdb.set_trace()
        # return {"input_features": input_features, "labels": tokenized_labels}


        return {
            "input_features": input_features,
            "labels": tokenized
        }

        # Tokenization
        # tokenized_labels = self.tokenizer(text, padding="longest", return_tensors="pt")
        # # label_features = [{"input_ids": ids} for ids in tokenized_labels]
        # # labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")
        # # labels = labels_batch["input_ids"].masked_fill(
        # #     labels_batch.attention_mask.ne(1), -100
        # # )
        # # pdb.set_trace()
        # return {"input_features": input_features, "labels": tokenized_labels}

        # input_ids = tokenized["input_ids"][0]
        # attention_mask = tokenized["attention_mask"][0]
        #
        # # Masked labels for Whisper
        # labels = np.where(attention_mask == 1, input_ids, -100)
        #
        # return {
        #     "input_features": torch.tensor(padded, dtype=torch.float32),
        #     "labels": torch.tensor(labels, dtype=torch.long),
        # }
        #
def get_whisper_dataloader(
    hdf5_path,
    batch_size=8,
    num_workers=4,
    limit=None,
    max_len=3000,
    shuffle=True,
    debug=False
):
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="de", task="transcribe")

    dataset = WhisperHDF5TorchDataset(
        hdf5_path=hdf5_path,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        sampling_rate=16000,
        max_len=max_len,
        limit=limit,
        debug=debug
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":
    hdf5_path = r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_complete_v3_sharded/train/eg_dataset_complete_v3_train.h5"

    dataloader = get_whisper_dataloader(hdf5_path, batch_size=2, limit=20)

    for batch in dataloader:
        print(batch["input_features"].shape)  # (B, 80, 3000)
        print(batch["labels"].shape)          # (B, T)
        break  # just preview the first batch

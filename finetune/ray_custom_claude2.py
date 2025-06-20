import os
import pdb
import time
import shutil
import h5py
import numpy as np
import pandas as pd
from typing import List, Dict, Iterator, Tuple, Optional, Any, Union
from ray.data.datasource import Datasource, ReadTask
from ray.data.block import BlockMetadata
import pyarrow as pa
from transformers import WhisperFeatureExtractor, WhisperTokenizer


def whisper_preprocess_batch(
    batch: Dict[str, List],
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    sampling_rate: int,
    debug: bool = False,
    max_len: int = 3000  # You can adjust this based on your typical audio length
) -> pd.DataFrame:
    """
    Preprocess a batch with fixed-length padding for Whisper training.
    Ensures input_features have uniform shape for Ray compatibility.

    Args:
        batch: Dict containing "audio", "transcription", and "idx"
        feature_extractor: HuggingFace WhisperFeatureExtractor
        tokenizer: HuggingFace WhisperTokenizer
        sampling_rate: Audio sampling rate
        debug: Whether to print debug info
        max_len: Fixed length to pad/truncate input_features to

    Returns:
        pd.DataFrame with keys: "idx", "input_features", "labels"
    """
    audios = batch["audio"]
    texts = batch["transcription"]
    idxs = batch["idx"]

    t_start = time.time() if debug else None

    features = []
    for i, audio in enumerate(audios):
        try:
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)

            extracted = feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
            padded = np.zeros((80, max_len), dtype=np.float32)
            length = min(extracted.shape[1], max_len)
            padded[:, :length] = extracted[:, :length]
            features.append(padded)
        except Exception as e:
            if debug:
                print(f"[DEBUG] Feature extraction failed at index {i}: {e}")
            features.append(np.zeros((80, max_len), dtype=np.float32))

    if debug:
        print(f"[DEBUG] Feature extraction + padding took {time.time() - t_start:.2f}s")

    tokenized = tokenizer(texts, padding="longest", return_tensors="np")
    masked_labels = np.where(tokenized["attention_mask"] == 1, tokenized["input_ids"], -100)

    return pd.DataFrame({
        "idx": idxs,
        "input_features": [f for f in features],  # List of np.array shape (80, 3000)
        "labels": [l for l in masked_labels]  # Also ensure this is a list, not stacked
    })


def make_reader_fn(
    hdf5_path: str,
    start: int,
    end: int,
    batch_size: int,
    debug: bool = False,
    preprocess: bool = False,
    preprocess_batch_fn=None,
):
    def reader() -> Iterator[pd.DataFrame]:
        if debug:
            print(f"[DEBUG] Reader running for range {start}:{end}")

        with h5py.File(hdf5_path, "r") as f:
            for batch_start in range(start, end, batch_size):
                batch_end = min(batch_start + batch_size, end)
                audios = f["audio"][batch_start:batch_end]
                texts = f["transcription"][batch_start:batch_end]

                if isinstance(texts[0], bytes):
                    texts = [t.decode("utf-8") for t in texts]

                batch_data = {
                    "idx": np.arange(batch_start, batch_end),
                    "audio": list(audios),
                    "transcription": texts,
                }

                if preprocess and preprocess_batch_fn:
                    yield preprocess_batch_fn(batch_data)
                else:
                    yield pd.DataFrame(batch_data)

    return reader


class WhisperHDF5Datasource(Datasource):
    """
    Optimized HDF5 datasource for Whisper model training.

    Features:
    - Fast vectorized HDF5 reading
    - Smart limit handling for preview operations
    - Optional lazy preprocessing
    - Local file caching for performance
    """

    def __init__(
            self,
            hdf5_path: str,
            batch_size: int = 32,
            feature_extractor: Optional[Any] = None,
            tokenizer: Optional[Any] = None,
            preprocess: bool = False,
            copy_to_local: bool = True,
            sampling_rate: int = 16000,
            target_language: str = "de",
            debug: bool = False
    ):
        """
        Initialize the datasource.

        Args:
            hdf5_path: Path to HDF5 file
            batch_size: Batch size for reading
            feature_extractor: Optional WhisperFeatureExtractor
            tokenizer: Optional WhisperTokenizer
            preprocess: Whether to preprocess during loading
            copy_to_local: Whether to copy the HDF5 file to local storage
            sampling_rate: Audio sampling rate
            target_language: Target language for transcription
            debug: Whether to print debug information
        """
        self.original_path = hdf5_path
        self.hdf5_path = self._copy_to_local(hdf5_path) if copy_to_local else hdf5_path
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.sampling_rate = sampling_rate
        self.target_language = target_language
        self.debug = debug

        # Initialize preprocessing components if needed
        if self.preprocess and (self.feature_extractor is None or self.tokenizer is None):
            print("[INFO] Initializing default feature extractor and tokenizer")
            model_type = 'openai/whisper-large-v3'
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
            self.tokenizer = WhisperTokenizer.from_pretrained(
                model_type, language=self.target_language, task="transcribe")

        # Profile dataset
        with h5py.File(self.hdf5_path, "r") as f:
            self.total_samples = len(f["audio"])
            if self.debug:
                print(f"[DEBUG] Total samples in dataset: {self.total_samples}")

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

    def preprocess_batch(
            self,
            batch: Union[pd.DataFrame, Dict[str, List]]
    ) -> pd.DataFrame:
        """
        Apply Whisper feature extraction and tokenization to a batch.

        Args:
            batch: DataFrame or dict with raw audio and text data

        Returns:
            DataFrame with processed features and labels
        """
        if self.feature_extractor is None or self.tokenizer is None:
            raise ValueError("Feature extractor and tokenizer must be provided for preprocessing")

        # Handle DataFrame input
        if isinstance(batch, pd.DataFrame):
            audios = batch["audio"].tolist()
            texts = batch["transcription"].tolist()
            idxs = batch["idx"].tolist()
        else:
            # Handle dict input
            audios = batch["audio"]
            texts = batch["transcription"]
            idxs = batch["idx"]

        t_start = time.time() if self.debug else None

        # Feature extraction
        features = []
        for audio in audios:
            try:
                # Handle both numpy arrays and lists
                if isinstance(audio, list):
                    audio = np.array(audio, dtype=np.float32)

                # Extract features
                extracted = self.feature_extractor(
                    audio, sampling_rate=self.sampling_rate).input_features[0]
                features.append({"input_features": extracted})
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error extracting features: {e}")
                # Create empty feature as fallback
                features.append({"input_features": np.zeros((80, 3000), dtype=np.float32)})

        # Pad the features
        padded = self.feature_extractor.pad(
            features, padding="longest", return_tensors="np")

        if self.debug:
            t_features = time.time()
            print(f"[DEBUG] Feature extraction completed in {t_features - t_start:.4f}s")

        # Tokenization
        tokenized = self.tokenizer(texts, padding="longest", return_tensors="np")

        # Apply masking
        masked_labels = np.where(
            tokenized["attention_mask"] == 1,
            tokenized["input_ids"],
            -100
        )

        if self.debug:
            t_tokens = time.time()
            print(f"[DEBUG] Tokenization completed in {t_tokens - t_features:.4f}s")

        # Create result DataFrame
        result = pd.DataFrame({
            "idx": idxs,
            "input_features": list(padded["input_features"]),
            "labels": list(masked_labels)
        })

        if self.debug:
            t_end = time.time()
            print(f"[DEBUG] Total preprocessing time: {t_end - t_start:.4f}s")

        return result

    def prepare_read(self, parallelism: int, **read_args) -> List[ReadTask]:
        """
        Prepare read tasks for Ray Data with early limit detection.

        Args:
            parallelism: Number of parallel tasks
            read_args: Additional read arguments

        Returns:
            List of read tasks
        """
        limit = read_args.get("_limit", None)
        if self.debug and limit is not None:
            print(f"[DEBUG] Detected limit: {limit}")

        # Handle small-limit fast path
        if limit is not None and limit < self.batch_size:
            adjusted_batch_size = max(limit, 1)

            metadata = BlockMetadata(
                num_rows=adjusted_batch_size,
                size_bytes=None,
                schema=self._get_schema(),
                input_files=[self.hdf5_path],
                exec_stats=None,
            )

            return [ReadTask(make_reader_fn(
                hdf5_path=self.hdf5_path,
                start=0,
                end=adjusted_batch_size,
                batch_size=adjusted_batch_size,
                debug=self.debug,
                preprocess=self.preprocess,
                preprocess_batch_fn=self.preprocess_batch
            ), metadata)]

        # Standard sharded reading
        shard_size = max(1, self.total_samples // parallelism)
        read_tasks = []

        for shard_id in range(parallelism):
            start = shard_id * shard_size
            end = self.total_samples if shard_id == parallelism - 1 else start + shard_size

            if limit is not None and end - start > limit:
                if shard_id > 0:
                    continue
                end = start + limit

            metadata = BlockMetadata(
                num_rows=end - start,
                size_bytes=None,
                schema=self._get_schema(),
                input_files=[self.hdf5_path],
                exec_stats=None,
            )

            read_tasks.append(ReadTask(
                make_reader_fn(
                    hdf5_path=self.hdf5_path,
                    start=start,
                    end=end,
                    batch_size=self.batch_size,
                    debug=self.debug,
                    preprocess=self.preprocess,
                    preprocess_batch_fn=self.preprocess_batch  # only pass the method
                ),
                metadata
            ))

        return read_tasks

    def _get_schema(self) -> pa.Schema:
        if self.preprocess:
            return pa.schema([
                ("idx", pa.int64()),
                ("input_features", pa.list_(pa.float32())),
                ("labels", pa.list_(pa.int64())),
            ])
        else:
            return pa.schema([
                ("idx", pa.int64()),
                ("audio", pa.list_(pa.float32())),
                ("transcription", pa.string()),
            ])


# Helper function for easier dataset creation
def read_whisper_hdf5(
        hdf5_path,
        limit=None,
        batch_size=32,
        parallelism=1,
        preprocess=False,
        feature_extractor=None,
        tokenizer=None,
        copy_to_local=True,
        sampling_rate=16000,
        target_language="de",
        debug=False
):
    """
    Read HDF5 file with optional preprocessing for Whisper model.

    Args:
        hdf5_path: Path to HDF5 file
        limit: Limit the number of samples (for preview)
        batch_size: Batch size for processing
        parallelism: Number of parallel tasks
        preprocess: Whether to apply Whisper preprocessing
        feature_extractor: Optional custom feature extractor
        tokenizer: Optional custom tokenizer
        copy_to_local: Whether to copy HDF5 file to local disk
        sampling_rate: Audio sampling rate
        target_language: Target language for transcription
        debug: Whether to print debug information

    Returns:
        Ray Dataset with raw or preprocessed data
    """
    from ray.data import read_datasource

    # Create datasource
    ds = read_datasource(
        WhisperHDF5Datasource(
            hdf5_path=hdf5_path,
            batch_size=batch_size,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            preprocess=preprocess,
            copy_to_local=copy_to_local,
            sampling_rate=sampling_rate,
            target_language=target_language,
            debug=debug
        ),
        parallelism=parallelism,
        _limit=limit  # Pass limit info early
    )

    # Apply limit again for safety
    if limit is not None:
        ds = ds.limit(limit)

    return ds

import torch

class WhisperTorchDataCollator:
    def __call__(self, batch: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch from Ray Data's iter_torch_batches format.

        Args:
            batch: Dict with 'input_features' and 'labels', each containing a list/array

        Returns:
            Dict with PyTorch tensors
        """
        return {
            "input_features": torch.tensor(batch["input_features"], dtype=torch.float32),
            "labels": torch.tensor(batch["labels"], dtype=torch.long)
        }

# Example usage
if __name__ == "__main__":
    import time

    hdf5_path = r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_complete_v3_sharded/train/eg_dataset_complete_v3_train.h5"

    # Fast preview without preprocessing
    print("\n=== Fast preview without preprocessing ===")
    start_time = time.time()
    raw_ds = read_whisper_hdf5(
        hdf5_path=hdf5_path,
        limit=4,
        debug=True
    )
    raw_ds.show()
    print(f"Raw preview completed in {time.time() - start_time:.2f}s")

    # Preview with preprocessing
    print("\n=== Preview with preprocessing ===")
    start_time = time.time()
    processed_ds = read_whisper_hdf5(
        hdf5_path=hdf5_path,
        limit=10,
        preprocess=True,
        debug=True
    )
    print(processed_ds.schema())
    # processed_ds.show()
    # print(f"Processed preview completed in {time.time() - start_time:.2f}s")
    torch_collator = WhisperTorchDataCollator()

    pdb.set_trace()
    for batch in processed_ds.iter_torch_batches(prefetch_batches=0, batch_size=2,collate_fn=torch_collator):
        print(batch["input_features"].shape)
        break
    pdb.set_trace()

    # Load full dataset for training (with preprocessing)
    print("\n=== Load full dataset with preprocessing for training ===")
    start_time = time.time()
    train_ds = read_whisper_hdf5(
        hdf5_path=hdf5_path,
        batch_size=32,
        parallelism=4,
        preprocess=True,
        debug=True
    )
    print(f"Training dataset created in {time.time() - start_time:.2f}s")
    print(f"Total training samples: {train_ds.count()}")

from typing import Callable, Iterator
import h5py

def make_reader_fn(
    hdf5_path: str,
    start: int,
    end: int,
    batch_size: int,
    debug: bool = False,
    preprocess: bool = False,
    feature_extractor=None,
    tokenizer=None,
    sampling_rate=16000,
    preprocess_fn: Callable = None
):
    def reader() -> Iterator[pd.DataFrame]:
        if debug:
            print(f"[DEBUG] Reader running for range {start}:{end}")

        with h5py.File(hdf5_path, "r") as f:
            for batch_start in range(start, end, batch_size):
                batch_end = min(batch_start + batch_size, end)
                if batch_end <= batch_start:
                    continue

                audios = f["audio"][batch_start:batch_end]
                texts = f["transcription"][batch_start:batch_end]

                if isinstance(texts[0], bytes):
                    texts = [t.decode("utf-8") for t in texts]

                batch_data = {
                    "idx": np.arange(batch_start, batch_end),
                    "audio": list(audios),
                    "transcription": texts,
                }

                if preprocess and preprocess_fn:
                    yield preprocess_fn(batch_data, feature_extractor, tokenizer, sampling_rate, debug)
                else:
                    yield pd.DataFrame(batch_data)

    return reader

import os
import time
import shutil
import numpy as np
import pyarrow as pa
from ray.data.datasource import Datasource, ReadTask
from ray.data.block import BlockMetadata
from transformers import WhisperFeatureExtractor, WhisperTokenizer

class WhisperHDF5Datasource(Datasource):
    def __init__(
        self,
        hdf5_path: str,
        batch_size: int = 32,
        feature_extractor: WhisperFeatureExtractor = None,
        tokenizer: WhisperTokenizer = None,
        preprocess: bool = False,
        copy_to_local: bool = True,
        sampling_rate: int = 16000,
        target_language: str = "de",
        debug: bool = False
    ):
        self.original_path = hdf5_path
        self.hdf5_path = self._copy_to_local(hdf5_path) if copy_to_local else hdf5_path
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.sampling_rate = sampling_rate
        self.target_language = target_language
        self.debug = debug

        if self.preprocess and (self.feature_extractor is None or self.tokenizer is None):
            print("[INFO] Loading default Whisper model: openai/whisper-large-v3")
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
            self.tokenizer = WhisperTokenizer.from_pretrained(
                "openai/whisper-large-v3", language=self.target_language, task="transcribe"
            )

        with h5py.File(self.hdf5_path, "r") as f:
            self.total_samples = len(f["audio"])
            if self.debug:
                print(f"[DEBUG] Total samples: {self.total_samples}")

    def _copy_to_local(self, path: str) -> str:
        fname = os.path.basename(path)
        local_path = os.path.join("/tmp", fname)
        if not os.path.exists(local_path):
            print(f"[INFO] Copying {path} to {local_path}...")
            shutil.copy2(path, local_path)
        return local_path

    def _get_schema(self) -> pa.Schema:
        if self.preprocess:
            return pa.schema([
                ("idx", pa.int64()),
                ("input_features", pa.list_(pa.list_(pa.float32()))),  # 2D padded
                ("labels", pa.list_(pa.int64())),
            ])
        else:
            return pa.schema([
                ("idx", pa.int64()),
                ("audio", pa.list_(pa.float32())),
                ("transcription", pa.string())
            ])

    def prepare_read(self, parallelism: int, **read_args) -> List[ReadTask]:
        limit = read_args.get("_limit", None)
        if self.debug and limit:
            print(f"[DEBUG] Limit detected: {limit}")

        shard_size = max(1, self.total_samples // parallelism)
        read_tasks = []

        for shard_id in range(parallelism):
            start = shard_id * shard_size
            end = self.total_samples if shard_id == parallelism - 1 else start + shard_size

            if limit and (end - start > limit):
                if shard_id > 0:
                    continue
                end = start + limit

            metadata = BlockMetadata(
                num_rows=end - start,
                size_bytes=None,
                schema=self._get_schema(),
                input_files=[self.hdf5_path],
                exec_stats=None,
            )

            read_tasks.append(ReadTask(make_reader_fn(
                hdf5_path=self.hdf5_path,
                start=start,
                end=end,
                batch_size=self.batch_size,
                debug=self.debug,
                preprocess=self.preprocess,
                preprocess_fn=whisper_preprocess_batch,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                sampling_rate=self.sampling_rate
            ), metadata))

        return read_tasks

if __name__ == "__main__":
    import time

    hdf5_path = r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_complete_v3_sharded/train/eg_dataset_complete_v3_train.h5"

    # Fast preview without preprocessing
    print("\n=== Fast preview without preprocessing ===")
    start_time = time.time()
    raw_ds = read_whisper_hdf5(
        hdf5_path=hdf5_path,
        limit=4,
        debug=True
    )
    raw_ds.show()
    print(f"Raw preview completed in {time.time() - start_time:.2f}s")

    # Preview with preprocessing
    print("\n=== Preview with preprocessing ===")
    start_time = time.time()
    processed_ds = read_whisper_hdf5(
        hdf5_path=hdf5_path,
        limit=10,
        preprocess=True,
        debug=True
    )
    print(processed_ds.schema())
    # processed_ds.show()
    # print(f"Processed preview completed in {time.time() - start_time:.2f}s")
    torch_collator = WhisperTorchDataCollator()

    pdb.set_trace()
    for batch in processed_ds.iter_torch_batches(prefetch_batches=0, batch_size=2,collate_fn=torch_collator):
        print(batch["input_features"].shape)
        break
    pdb.set_trace()

    # Load full dataset for training (with preprocessing)
    print("\n=== Load full dataset with preprocessing for training ===")
    start_time = time.time()
    train_ds = read_whisper_hdf5(
        hdf5_path=hdf5_path,
        batch_size=32,
        parallelism=4,
        preprocess=True,
        debug=True
    )
    print(f"Training dataset created in {time.time() - start_time:.2f}s")
    print(f"Total training samples: {train_ds.count()}")


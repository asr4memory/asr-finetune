import pdb

import h5py
import numpy as np
import ray
from ray.data.datasource import Datasource, ReadTask
from typing import List, Any, Dict, Optional, Iterator
import os
import pdb


import h5py
import numpy as np
from typing import List, Dict, Any, Iterator
from ray.data.datasource import Datasource, ReadTask

from ray.data.block import BlockMetadata
import pyarrow as pa

class HDF5Datasource(Datasource):
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path

    def prepare_read(self, parallelism: int, **read_args) -> List[ReadTask]:
        with h5py.File(self.hdf5_path, "r") as f:
            total_samples = len(f["audio"])

        shard_size = total_samples // parallelism
        read_tasks = []

        for shard_id in range(parallelism):
            start = shard_id * shard_size
            end = total_samples if shard_id == parallelism - 1 else start + shard_size

            def make_reader(start=start, end=end):
                def reader():
                    data = []
                    with h5py.File(self.hdf5_path, "r") as f:
                        for i in range(start, end):
                            audio = np.array(f["audio"][i], dtype=np.float32)
                            text = f["transcription"][i]
                            if isinstance(text, bytes):
                                text = text.decode("utf-8")
                            data.append({"idx": i, "audio": audio, "transcription": text})
                    return data  # ✅ return a list (not a generator)

                return reader

            schema = pa.schema([
                ("idx", pa.int64()),
                ("audio", pa.list_(pa.float32())),
                ("transcription", pa.string())
            ])

            metadata = BlockMetadata(
                num_rows=end - start,
                size_bytes=None,  # If you know, use os.path.getsize(...)
                schema=schema,
                input_files=[self.hdf5_path],
                exec_stats=None
            )

            read_tasks.append(ReadTask(make_reader(), metadata))

        return read_tasks


def preprocessing_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    audios = [b["audio"] for b in batch]
    texts = [b["transcription"] for b in batch]

    features = [{"input_features": feature_extractor(audio, sampling_rate=16000).input_features[0]}
                for audio in audios]
    padded = feature_extractor.pad(features, padding="longest", return_tensors="pt")

    labels = tokenizer(texts, padding="longest", return_tensors="pt")
    labels["input_ids"] = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)

    return {
        "input_features": padded.input_features,
        "labels": labels["input_ids"]
    }


class HDF5BatchReader:
    def __init__(self, path, start, end, batch_size):
        self.path = path
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self._file = None

    def __iter__(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        buffer = []
        for i in range(self.start, self.end):
            audio = np.array(self._file["audio"][i], dtype=np.float32)
            text = self._file["transcription"][i]
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            buffer.append((i, audio, text))
            if len(buffer) >= self.batch_size:
                yield pd.DataFrame(buffer, columns=["idx", "audio", "transcription"])
                buffer = []
        if buffer:
            yield pd.DataFrame(buffer, columns=["idx", "audio", "transcription"])


from ray.data.datasource import Datasource, ReadTask
import h5py
import numpy as np
import pyarrow as pa
from typing import Iterator, List, Dict, Any
import pandas as pd

class StreamingHDF5Datasource2(Datasource):
    # def __init__(self, hdf5_path: str, batch_size: int = 32):
    #     self.hdf5_path = hdf5_path
    #     self.batch_size = batch_size

    def __init__(self, hdf5_path: str, batch_size: int = 32):
        self.original_path = hdf5_path
        self.hdf5_path = self._copy_to_local(hdf5_path)
        self.batch_size = batch_size

    def _copy_to_local(self, path: str) -> str:
        """Copy the HDF5 file to a fast local dir if it isn't already."""
        fname = os.path.basename(path)
        local_dir = "/tmp"
        local_path = os.path.join(local_dir, fname)

        # Only one copy per node; use rank to avoid collisions
        if not os.path.exists(local_path):
            try:
                print(f"[INFO] Copying {path} to {local_path} (node-local)...")
                shutil.copy2(path, local_path)
                print(f"[INFO] Copy complete: {local_path}")
            except Exception as e:
                print(f"[WARNING] Failed to copy {path} to local disk: {e}")
                return path  # fall back to original
        return local_path


    def prepare_read(self, parallelism: int, **read_args) -> List[ReadTask]:

        def _rows_to_columnar_batch(rows: List[tuple]) -> pd.DataFrame:
            return pd.DataFrame(rows, columns=["idx", "audio", "transcription"])

        with h5py.File(self.hdf5_path, "r") as f:
            total_samples = len(f["audio"])

        shard_size = total_samples // parallelism
        read_tasks = []

        for shard_id in range(parallelism):
            start = shard_id * shard_size
            end = total_samples if shard_id == parallelism - 1 else start + shard_size

            def make_reader(start=start, end=end):
                def reader() -> Iterator[Dict[str, np.ndarray]]:
                    with h5py.File(self.hdf5_path, "r") as f:
                        buffer = []
                        for i in range(start, end):
                            audio = np.array(f["audio"][i], dtype=np.float32)
                            text = f["transcription"][i]
                            if isinstance(text, bytes):
                                text = text.decode("utf-8")
                            buffer.append((i, audio, text))
                            if len(buffer) >= self.batch_size:
                                yield _rows_to_columnar_batch(buffer)
                                buffer = []
                        if buffer:
                            yield _rows_to_columnar_batch(buffer)
                return reader

            schema = pa.schema([
                ("idx", pa.int64()),
                ("audio", pa.list_(pa.float32())),
                ("transcription", pa.string())
            ])

            metadata = BlockMetadata(
                num_rows=end - start,
                size_bytes=None,
                schema=schema,
                input_files=[self.hdf5_path],
                exec_stats=None
            )
            reader = lambda: HDF5BatchReader(self.hdf5_path, start, end, self.batch_size)
            read_tasks.append(ReadTask(reader, metadata))

        return read_tasks


import os
import time
import shutil
import h5py
import numpy as np
import pandas as pd
from typing import List, Dict, Iterator, Tuple
from ray.data.datasource import Datasource, ReadTask
import pyarrow as pa
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

class StreamingHDF5Datasource(Datasource):
    def __init__(self, hdf5_path: str, batch_size: int, feature_extractor, tokenizer):
        self.original_path = hdf5_path
        self.hdf5_path = self._copy_to_local(hdf5_path)
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def _copy_to_local(self, path: str) -> str:
        fname = os.path.basename(path)
        local_dir = "/tmp"
        local_path = os.path.join(local_dir, fname)
        if not os.path.exists(local_path):
            try:
                print(f"[INFO] Copying {path} to {local_path} (node-local)...")
                shutil.copy2(path, local_path)
            except Exception as e:
                print(f"[WARNING] Failed to copy to local disk: {e}")
                return path
        return local_path

    def _preprocess_batch(
            self, batch: List[Tuple[int, np.ndarray, str]]
    ) -> pd.DataFrame:
        idxs, audios, texts = zip(*batch)

        # Feature extraction
        features = [
            {"input_features": self.feature_extractor(audio, sampling_rate=16000).input_features[0]}
            for audio in audios
        ]
        padded = self.feature_extractor.pad(features, padding="longest", return_tensors="np")

        # Tokenization
        tokenized = self.tokenizer(list(texts), padding="longest", return_tensors="np")
        tokenized["input_ids"] = np.where(
            tokenized["attention_mask"] == 1, tokenized["input_ids"], -100
        )

        return pd.DataFrame({
            "idx": list(idxs),
            "input_features": list(padded["input_features"]),
            "labels": list(tokenized["input_ids"])
        })

    def prepare_read(self, parallelism: int, **read_args) -> List[ReadTask]:
        with h5py.File(self.hdf5_path, "r") as f:
            total_samples = len(f["audio"])

        shard_size = total_samples // parallelism
        read_tasks = []

        for shard_id in range(parallelism):
            start = shard_id * shard_size
            end = total_samples if shard_id == parallelism - 1 else start + shard_size

            def make_reader(start=start, end=end):
                def reader() -> Iterator[Dict[str, np.ndarray]]:
                    # Open HDF5 file once per reader
                    with h5py.File(self.hdf5_path, "r") as f:
                        # Pre-fetch indices to avoid reopening file
                        indices = list(range(start, end))

                        # Process in actual batches (not one at a time)
                        for batch_start in range(0, len(indices), self.batch_size):
                            batch_indices = indices[batch_start:batch_start + self.batch_size]
                            buffer = []

                            for i in batch_indices:
                                audio = np.array(f["audio"][i], dtype=np.float32)
                                text = f["transcription"][i]
                                if isinstance(text, bytes):
                                    text = text.decode("utf-8")
                                buffer.append((i, audio, text))

                            # Only process if we have items
                            if buffer:
                                yield self._preprocess_batch(buffer)
                return reader

            # Dummy schema (won’t be used much after preprocessing)
            schema = pa.schema([
                ("input_features", pa.list_(pa.float32())),
                ("labels", pa.list_(pa.int64())),
                ("idx", pa.int64())
            ])

            metadata = BlockMetadata(
                num_rows=end - start,
                size_bytes=None,
                schema=schema,
                input_files=[self.hdf5_path],
                exec_stats=None
            )

            read_tasks.append(ReadTask(make_reader(), metadata))

        return read_tasks


from ray.data import read_datasource

# hdf5_path = "/path/to/data.h5"
hdf5_path = r"/Users/chrvt/Documents/GitHub/asr-finetune/data_example/datasets/eg_dataset_complete_v3_sharded/train/eg_dataset_complete_v3_train.h5"
model_type = 'openai/whisper-large-v3'
target_language = 'de'
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
tokenizer = WhisperTokenizer.from_pretrained(model_type, language=target_language, task="transcribe")



ds = read_datasource(
    StreamingHDF5Datasource(
        hdf5_path=hdf5_path,
        batch_size=1,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    ),
    parallelism=1
)


# ds = read_datasource(
#     HDF5Datasource(hdf5_path),
#     parallelism=4  # Match number of CPUs/GPUs you want to use
# )

# ds = read_datasource(
#     StreamingHDF5Datasource(hdf5_path,batch_size=32),
#     parallelism=4  # Match number of CPUs/GPUs you want to use
# )


pdb.set_trace()

# processed_ds = ds.map_batches(preprocessing_collate, batch_size=8, batch_format="native")

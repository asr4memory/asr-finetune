"""
Ray ↔ Hugging Face Transformers Integration Utilities

This module provides classes and utilities to integrate Hugging Face's `Trainer` API
with Ray Train for distributed hyperparameter tuning and evaluation.

Features:
- RayTrainReportCallback: Reports Hugging Face checkpoints + metrics to Ray Tune
- RayTorchIterableDataset: Adapts Ray datasets to PyTorch IterableDatasets
- prepare_trainer_custom: Modifies a Hugging Face Trainer for Ray-compatible data loaders
"""
import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, Type

from torch.utils.data import DataLoader, Dataset, IterableDataset

import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.data.iterator import _IterableFromIterator
from ray.train import Checkpoint
from ray.util import PublicAPI

logger = logging.getLogger(__name__)


TRANSFORMERS_IMPORT_ERROR: Optional[ImportError] = None

try:
    import transformers.trainer
    from transformers import Trainer
    from transformers.trainer_callback import TrainerCallback
except ImportError as e:
    TRANSFORMERS_IMPORT_ERROR = e
    TrainerCallback = object

from torch.utils.data import DataLoader
from typing import Optional
import random
import os
import glob
import ray

@PublicAPI(stability="beta")
class RayTrainReportCallback(TrainerCallback):
    """A simple callback to report checkpoints and metrics to Ray Train.

    This callback is a subclass of `transformers.TrainerCallback
    <https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.TrainerCallback>`_
    and overrides the `TrainerCallback.on_save()` method. After
    a new checkpoint get saved, it fetches the latest metric dictionary
    from `TrainerState.log_history` and reports it with the latest checkpoint
    to Ray Train.

    Checkpoints will be saved in the following structure::

        checkpoint_00000*/   Ray Train Checkpoint
        └─ checkpoint/       Hugging Face Transformers Checkpoint

    For customized reporting and checkpointing logic, implement your own
    `transformers.TrainerCallback` following this user
    guide: :ref:`Saving and Loading Checkpoints <train-dl-saving-checkpoints>`.

    Note that users should ensure that the logging, evaluation, and saving frequencies
    are properly configured so that the monitoring metric is always up-to-date
    when `transformers.Trainer` saves a checkpoint.

    Suppose the monitoring metric is reported from evaluation stage:

    Some valid configurations:
        - evaluation_strategy == save_strategy == "epoch"
        - evaluation_strategy == save_strategy == "steps", save_steps % eval_steps == 0

    Some invalid configurations:
        - evaluation_strategy != save_strategy
        - evaluation_strategy == save_strategy == "steps", save_steps % eval_steps != 0

    """

    CHECKPOINT_NAME = "checkpoint"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        record_extra_usage_tag(TagKey.TRAIN_TRANSFORMERS_RAYTRAINREPORTCALLBACK, "1")


    def on_save(self, args, state, control, **kwargs):
        """Event called after a checkpoint save."""
        with TemporaryDirectory() as tmpdir:
            # Aggregate all the logged metrics
            metrics = {}
            for log in state.log_history:
                metrics.update(log)

            # Copy ckpt files and construct a Ray Train Checkpoint
            source_ckpt_path = transformers.trainer.get_last_checkpoint(args.output_dir)
            if source_ckpt_path is not None:
                target_ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
                shutil.copytree(source_ckpt_path, target_ckpt_path)
                checkpoint = Checkpoint.from_directory(tmpdir)
            else:
                checkpoint = None

            # Report latest metrics and checkpoint to Ray Train
            ray.train.report(metrics=metrics, checkpoint=checkpoint)





class RayTorchIterableDataset(IterableDataset):
    """
    Wraps a Ray dataset iterator as a PyTorch-compatible IterableDataset.

    Used to convert Ray's streaming datasets into PyTorch DataLoader-compatible format.
    """

    def __init__(self, data_iterable) -> None:
        super().__init__()
        self.data_iterable = data_iterable

    def __iter__(self) -> Iterator:
        return iter(self.data_iterable)



@PublicAPI(stability="beta")
def prepare_trainer_custom2(trainer: "Trainer") -> "Trainer":
    """
    Prepares a Hugging Face Trainer for use with Ray streaming datasets.

    If `train_dataset` or `eval_dataset` are Ray `IterableDataset`s, this function
    overrides the `get_train_dataloader()` and `get_eval_dataloader()` methods
    to return Ray-compatible PyTorch DataLoaders.

    Returns:
        Trainer: Modified Trainer instance with Ray-compatible data handling.
    """

    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise TRANSFORMERS_IMPORT_ERROR

    base_trainer_class: Type[transformers.trainer.Trainer] = trainer.__class__

    class RayTransformersTrainer(base_trainer_class):
        """
        A subclass of Hugging Face's Trainer that supports Ray Datasets.

        It overrides dataloader methods to support streaming input from Ray datasets.
        """

        def get_train_dataloader(self) -> DataLoader:
            if isinstance(self.train_dataset, _IterableFromIterator):
                dataset = RayTorchIterableDataset(self.train_dataset)
                return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
            else:
                return super().get_train_dataloader()


        def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
            # 1) Where to get candidate files from?
            #    - Prefer explicit list on the trainer: self.eval_parquet_paths = ["/path/a.parquet", ...]
            #    - Or a directory: self.eval_parquet_dir = "/path/to/val_parquet"
            #    - Otherwise fall back to the existing dataset-based subsampling.

            seed = getattr(self.args, "seed", None)
            if seed is not None:
                random.seed(seed + int(getattr(self.state, "global_step", 0) or 0))  # vary across evals but reproducible

            file_candidates = []
#            self.eval_parquet_dir = r"/scratch/usr/bemchrvt/data/eg_dataset_complete_v3_sharded"
            if hasattr(self, "eval_parquet_paths") and self.eval_parquet_paths:
                file_candidates = list(self.eval_parquet_paths)
            elif hasattr(self, "eval_parquet_dir") and self.eval_parquet_dir and os.path.isdir(self.eval_parquet_dir):
                file_candidates = sorted(glob.glob(os.path.join(self.eval_parquet_dir, "*.parquet")))

            if file_candidates:
                chosen_path = random.choice(file_candidates)
                print(f"[Eval] Using single Parquet file: {chosen_path}")
                # disable PG capture just around the read

                import ray
                import time, psutil
                # A tiny helper that runs OUTSIDE the trial PG
#                @ray.remote(num_cpus=4, scheduling_strategy="DEFAULT")
                from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

                @ray.remote(num_cpus=1,  # keep this small
                scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=None))
                def _build_eval_ds(path: str, parallelism: int = 2):
                    # Fan out across free CPUs on the node
                    print(f"[Eval] Read Parquet at {time.ctime()} ")
                    
                    proc = psutil.Process(os.getpid())
                    print(f"[Eval] PID {os.getpid()} started with CPU quota={os.environ.get('CPU_REQUESTED', 'N/A')} at {time.ctime()}")


                    ds = ray.data.read_parquet(
                        path,
                        parallelism=parallelism,
                        ray_remote_args={"num_cpus": 0.25} #, "scheduling_strategy": "DEFAULT"},
                    )
                    return ds.materialize()  # Execute now so blocks are built outside the PG
                
                # Build eval dataset OUTSIDE the trial's placement group
                ds_ref = _build_eval_ds.remote(chosen_path, parallelism=4)
                subsampled = ray.get(ds_ref)

            else:
                # Fallback: your old behavior (fractional subsample of the whole dataset)
                if eval_dataset is None:
                    eval_dataset = self.eval_dataset
                print(f"[Eval] Using fallback subsampling {self.eval_sample_fraction}")
                subsampled = eval_dataset.random_sample(self.eval_sample_fraction, seed=seed)

            # 2) Build the iterator and wrap in a tiny DataLoader that yields the Ray batches directly.
            print(f"[Eval]: Set up Eval Iterator")
            eval_dataset_iter = subsampled.iter_torch_batches(
                prefetch_batches=self.prefetch_batches,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.eval_collator,
            )
            print(f"[Eval]: That took long!")
            if isinstance(eval_dataset_iter, _IterableFromIterator):
                dataset = RayTorchIterableDataset(eval_dataset_iter)
                return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
            else:
                # Shouldn’t happen with Ray iterators, but keep the original fallback just in case.
                return super().get_eval_dataloader(eval_dataset)

#        def get_eval_dataloader(
#            self, eval_dataset: Optional[Dataset] = None
#        ) -> DataLoader:
#            if eval_dataset is None:
#                eval_dataset = self.eval_dataset
#
#            print(f"subsampling {self.eval_sample_fraction} %")
#            print(f"self.prefetch_batches {self.prefetch_batches}")
#            print(f"per_device_eval_batch_size, {self.args.per_device_eval_batch_size}")
#            subsampled = self.eval_dataset.random_sample(self.eval_sample_fraction)
#            
#            eval_dataset_iter = subsampled.iter_torch_batches(
#                prefetch_batches=self.prefetch_batches,
#                batch_size=self.args.per_device_eval_batch_size, collate_fn=self.eval_collator)
#                
#            print(f"Eval Dataset subsampled and ready to be evaluated on!")
#            
#            if isinstance(eval_dataset_iter, _IterableFromIterator):
#                dataset = RayTorchIterableDataset(eval_dataset_iter)
#                return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
#            else:
#                return super().get_eval_dataloader(eval_dataset)

    trainer.__class__ = RayTransformersTrainer

    record_extra_usage_tag(TagKey.TRAIN_TRANSFORMERS_PREPARE_TRAINER, "1")
    return trainer



@PublicAPI(stability="beta")
def prepare_trainer_custom(trainer: "Trainer") -> "Trainer":
    """
    Hugging Face Trainer shim for Ray Datasets when the *evaluation* dataset
    is already loaded and materialized outside the TorchTrainer.

    Usage expectations:
      - trainer.train_dataset  may be a Ray iterator OR plain HF dataset.
      - trainer.eval_dataset   MUST be a ray.data.Dataset (materialized).
      - trainer.eval_collator  is your collator for eval batches.
      - Optional knobs you can set on the trainer *before* training:
          * trainer.eval_sample_fraction: float in (0,1] (default=1.0)
          * trainer.eval_num_examples:   int, overrides fraction if set
          * trainer.prefetch_batches:    int, dataloader prefetch (default=1)
    """
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise TRANSFORMERS_IMPORT_ERROR

    base_cls: Type[transformers.trainer.Trainer] = trainer.__class__

    class RayTransformersTrainer(base_cls):
        # ---- helpers ---------------------------------------------------------
        @property
        def _eval_fraction(self) -> float:
            frac = getattr(self, "eval_sample_fraction", 1.0)
            # clamp to sane range
            try:
                frac = float(frac)
                if frac <= 0 or frac > 1:
                    frac = 1.0
            except Exception:
                frac = 1.0
            return frac

        def _choose_eval_subset(self, ds):
            """
            Return a (possibly) sub-sampled Ray Dataset, determined at each
            evaluate() call. Uses seed that changes with global_step for
            reproducibility across calls.
            """
            # vary the seed across evals but deterministic for given step
            base_seed = getattr(self.args, "seed", 0) or 0
            step = int(getattr(self.state, "global_step", 0) or 0)
            seed = base_seed + step

            # exact-N takes precedence if provided
            n_examples = getattr(self, "eval_num_examples", None)
            if isinstance(n_examples, int) and n_examples > 0:
                # Prefer randomized block order without a full shuffle: sample a fraction,
                # then trim with .limit(). We avoid .random_shuffle() because it can be slow.
                # Heuristic fraction: oversample a bit to counter randomness on small N.
                # We cache total rows once if available.
                total = getattr(self, "_eval_total_rows", None)
                if total is None:
                    try:
                        # This count is cheap if the dataset is materialized; we cache it.
                        total = ds.count()
                        setattr(self, "_eval_total_rows", total)
                    except Exception:
                        total = None

                if total and total > 0:
                    frac = min(1.0, max(0.0, 1.2 * float(n_examples) / float(total)))
                    sub = ds.random_sample(frac, seed=seed).limit(n_examples)
                else:
                    # Fallback: just limit() (may take first N in block order)
                    # Adding a tiny random_sample jitters block selection a bit.
                    sub = ds.random_sample(0.5, seed=seed).limit(n_examples)
                return sub

            # otherwise: fraction mode
            frac = self._eval_fraction
            if frac < 1.0:
                return ds.random_sample(frac, seed=seed)
            return ds

        # ---- dataloaders -----------------------------------------------------
        def get_train_dataloader(self) -> DataLoader:
            # If Ray is already giving us an iterator that yields *batched* tensors,
            # we wrap it so HF DataLoader doesn't re-batch it.
            if isinstance(self.train_dataset, _IterableFromIterator):
                dataset = RayTorchIterableDataset(self.train_dataset)
                # batch_size=1 + identity collate: each item is already a full batch
                return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
            return super().get_train_dataloader()

        def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
            # We expect a Ray Dataset here (materialized). If caller passes something,
            # prefer that; else use self.eval_dataset.
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset

            # If the user passed an HF dataset/iterator by accident, fall back.
            try:
                import ray.data
                is_ray_ds = isinstance(ds, ray.data.Dataset)
            except Exception:
                is_ray_ds = False

            if not is_ray_ds:
                # fall back to stock behavior (e.g., unit tests)
                return super().get_eval_dataloader(eval_dataset)

            # Subsample freshly on every evaluate() call
            sub = self._choose_eval_subset(ds)
            print("[Eval]: Length eval dataset ", ds.count())
            
            prefetch = getattr(self, "prefetch_batches", 1)
            collator = getattr(self, "eval_collator", None)

            # Build a Ray → Torch iterator (already yields full batches)
            iter_batches = sub.iter_torch_batches(
                prefetch_batches=prefetch,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=collator,
            )
            wrapped = RayTorchIterableDataset(iter_batches)
            return DataLoader(wrapped, batch_size=1, collate_fn=lambda x: x[0])

    trainer.__class__ = RayTransformersTrainer
    record_extra_usage_tag(TagKey.TRAIN_TRANSFORMERS_PREPARE_TRAINER, "1")
    return trainer

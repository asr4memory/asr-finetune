"""Regenerate trainers/data/validation_summary.csv with the SAME generation kwargs the
trial-time evaluation uses.

Why: the existing validation_summary.csv was created with unknown decoding settings, so
``eval_wer_diff`` is partially driven by decoding-method drift rather than fine-tuning.
This script re-runs the chosen baseline (default: openai/whisper-large-v3) over every
validation shard using greedy decoding, bf16, max_length=225, and the same shard
partitioning as the new hail-mary HPO setup (val_anchor + val_1..val_N).

Run on a single A100 from the login node via the matching SBATCH wrapper in
finetune_recompute_baseline.sh. Expected runtime ~30-60 min for 20K samples.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure the project root is on PYTHONPATH regardless of where the script is launched
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ray
import torch
from transformers import set_seed

from finetuning.data_and_collator.datasets_and_collators import (
    get_datasets_and_collators,
    make_dataset_kwargs,
)
from finetuning.models.whisper_models import get_whisper_models
from finetuning.trainers.metrics import get_metric_to_optimize
from finetuning.trainers.utils import normalize as normalize_fn
from finetuning.projects_paths import DATA_PATH


def partition_with_anchor(ds, fraction: float, anchor_size: int):
    """Take the first ``anchor_size`` rows as the fixed val_anchor shard, then split
    the remainder into ``ceil(1/fraction)`` random shards named val_1..val_N.

    The split is deterministic for a given input dataset because ``split_at_indices``
    operates on row order, which is stable for a parquet-backed Ray dataset.
    """
    total = ds.count()
    anchor_size = min(anchor_size, total)
    rest_size = total - anchor_size

    if fraction <= 0 or rest_size <= 0:
        n_rest_splits = 1
        target_rest = rest_size
    else:
        n_rest_splits = max(1, math.ceil(1 / fraction))
        target_rest = max(1, math.ceil(rest_size / n_rest_splits))

    split_indices = [anchor_size]
    for i in range(n_rest_splits - 1):
        idx = min(anchor_size + (i + 1) * target_rest, total)
        if idx > split_indices[-1]:
            split_indices.append(idx)

    splits = ds.split_at_indices(split_indices)

    out = {"val_anchor": splits[0].materialize()}
    for i, sub in enumerate(splits[1:]):
        out[f"val_{i + 1}"] = sub.materialize()
    return out


def evaluate_shard(
    model,
    ds,
    collator,
    processor,
    tokenizer,
    *,
    batch_size: int,
    prefetch_batches: int,
    max_length: int,
    language: str,
    task: str,
    device: torch.device,
    max_batches: int | None = None,
    shard_name: str = "?",
    log_every: int = 25,
):
    """Compute (eval_loss, eval_wer, n_samples) on a single shard.

    Generation kwargs are pinned to what the trial-time evaluator uses:
    greedy (num_beams=1), explicit language/task, max_length=225 by default.
    Prints progress every ``log_every`` batches with ``flush=True`` so the
    SLURM out file stays current.
    """
    import time

    model.eval()
    metric = get_metric_to_optimize("evaluate_wer", tokenizer=tokenizer)
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0

    total_loss = 0.0
    total_batches = 0
    preds: list[str] = []
    refs: list[str] = []

    eval_iter = ds.iter_torch_batches(
        prefetch_batches=prefetch_batches,
        batch_size=batch_size,
        collate_fn=collator,
    )

    t0 = time.time()
    last_log = t0
    for bi, batch in enumerate(eval_iter):
        if max_batches is not None and bi >= max_batches:
            break
        if not isinstance(batch, dict):
            continue
        if "input_features" not in batch or "labels" not in batch:
            continue

        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        # The model was cast to bf16 on CUDA; the parquet's input_features are fp32.
        # Cast floating-point inputs to match the model dtype to avoid
        # "Input type (float) and bias type (BFloat16) should be the same" in conv1.
        model_dtype = next(model.parameters()).dtype
        if "input_features" in batch and torch.is_tensor(batch["input_features"]) \
                and torch.is_floating_point(batch["input_features"]):
            batch["input_features"] = batch["input_features"].to(model_dtype)

        with torch.no_grad():
            outputs = model(**batch)
            total_loss += float(outputs.loss.detach().float().item())
            total_batches += 1

            pred_ids = model.generate(
                input_features=batch["input_features"],
                max_length=max_length,
                num_beams=1,
                language=language,
                task=task,
            )

        labels = batch["labels"].detach().clone()
        labels[labels == -100] = pad_id

        pred_text = processor.batch_decode(pred_ids.detach().cpu(), skip_special_tokens=True)
        ref_text = processor.batch_decode(labels.detach().cpu(), skip_special_tokens=True)
        pred_text = [normalize_fn(str(p)) for p in pred_text]
        ref_text = [normalize_fn(str(r)) for r in ref_text]

        n = min(len(pred_text), len(ref_text))
        if n > 0:
            preds.extend(pred_text[:n])
            refs.extend(ref_text[:n])

        if (bi + 1) % log_every == 0:
            now = time.time()
            avg_per_batch = (now - t0) / max(bi + 1, 1)
            since_last = now - last_log
            print(
                f"[baseline][{shard_name}] batch {bi + 1} | "
                f"{n * (bi + 1)} samples | "
                f"avg {avg_per_batch:.2f}s/batch | "
                f"+{since_last:.1f}s since last log",
                flush=True,
            )
            last_log = now

    avg_loss = total_loss / max(total_batches, 1)
    if preds and refs:
        wer_0_1 = metric.compute(predictions=preds, references=refs)
        wer_pct = 100.0 * float(wer_0_1) if wer_0_1 is not None else float("nan")
    else:
        wer_pct = float("nan")
    return avg_loss, wer_pct, len(preds)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_model", default="openai/whisper-large-v3",
                   help="HF hub id or local dir for the WER baseline model.")
    p.add_argument("--local_model", action="store_true",
                   help="Treat --baseline_model as a local path (uses get_whisper_models local mode).")
    p.add_argument("--target_language", default="german")
    p.add_argument("--gen_language", default="de",
                   help="Whisper generate(language=...) tag. Usually the 2-letter code.")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--data_mode", default="parquet",
                   choices=["h5", "parquet", "parquet_h5", "train_parquet", "val_parquet", "val_h5"])
    p.add_argument("--dataset_name", default="crisper_whisper_preprocessings")
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--eval_sample_fraction", type=float, default=0.05,
                   help="Random-shard size as a fraction of (val_total - anchor_size).")
    p.add_argument("--anchor_size", type=int, default=2000)
    p.add_argument("--output_csv", default=str(PROJECT_ROOT / "finetuning" / "trainers" / "data" / "validation_summary.csv"))
    p.add_argument("--cpus_per_trial", type=int, default=2)
    p.add_argument("--random_seed", type=int, default=1337)
    p.add_argument("--max_length", type=int, default=225)
    p.add_argument("--prefetch_batches", type=int, default=0)
    p.add_argument("--run_on_local_machine", action="store_true",
                   help="Pass through to get_whisper_models for the data-pipeline's tokenizer load.")
    p.add_argument("--peft", action="store_true",
                   help="Pass-through for data-pipeline model load (load_in_8bit). Should be False for the baseline.")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max_batches_per_shard", type=int, default=None,
                   help="Optional cap (debug). Leave unset for full eval.")
    p.add_argument("--shards_to_run", default=None,
                   help="Comma-separated subset of shard names to compute (e.g. 'val_anchor,val_1,val_2'). "
                        "Useful for resuming a partial run.")
    p.add_argument("--log_every", type=int, default=25,
                   help="Print progress every N batches.")
    p.add_argument("--force", action="store_true",
                   help="Recompute shards that are already present in the output CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.random_seed)

    print("[baseline] STEP 0: ray init", flush=True)
    if not ray.is_initialized():
        # Constrain Ray's footprint: this script needs Ray only for
        # ray.data.read_parquet. Spawning N workers on a 40-core HPC node has
        # been observed to hang during worker registration; explicit limits
        # avoid that. include_dashboard=False skips the aiohttp dashboard
        # bootstrap which is also a frequent stall point on HPC.
        ray_tmp = os.environ.get("TMPDIR", "/tmp")
        ray.init(
            num_cpus=int(os.environ.get("RAY_NUM_CPUS", "2")),
            include_dashboard=False,
            log_to_driver=True,
            _temp_dir=ray_tmp,
            ignore_reinit_error=True,
        )
    print("[baseline] STEP 0: ray init done", flush=True)

    pipeline_model_type = os.environ.get("PIPELINE_MODEL_TYPE", "CrisperWhisper")
    print(f"[baseline] STEP 1: build dataset kwargs (pipeline_model_type={pipeline_model_type}, "
          f"data_mode={args.data_mode}, dataset_name={args.dataset_name})", flush=True)
    # Build the dataset_kwargs the same way training does. We pretend to be the trial:
    # we load the tokenizer/processor of the FINE-TUNED model family (CrisperWhisper)
    # so that the labels in the parquet (token ids from that tokenizer) decode correctly.
    # The actual generation, however, uses the baseline model loaded separately below.
    data_args = SimpleNamespace(
        cpus_per_trial=args.cpus_per_trial,
        random_seed=args.random_seed,
        model_type=pipeline_model_type,
        target_language=args.target_language,
        return_timestamps=False,
        run_on_local_machine=args.run_on_local_machine,
        path_to_data=DATA_PATH,
        dataset_name=args.dataset_name,
        peft=args.peft,
        debug=args.debug,
        data_mode=args.data_mode,
    )
    dataset_kwargs = make_dataset_kwargs(data_args)
    print(f"[baseline] STEP 1: dataset_kwargs = {dataset_kwargs}", flush=True)

    print("[baseline] STEP 2: get_datasets_and_collators (will load pipeline tokenizer + open parquet)", flush=True)
    ray_datasets, data_collators = get_datasets_and_collators(dataset_kwargs)
    print(f"[baseline] STEP 2: done. keys={list(ray_datasets.keys())}", flush=True)

    val_ds = ray_datasets["val"]
    print("[baseline] STEP 3: materializing val_ds count ...", flush=True)
    n_val = val_ds.count()
    print(f"[baseline] STEP 3: full val_ds count = {n_val}", flush=True)

    print("[baseline] STEP 4: partition_with_anchor ...", flush=True)
    shards = partition_with_anchor(val_ds, args.eval_sample_fraction, args.anchor_size)
    print(f"[baseline] STEP 4: partitioned into {len(shards)} shards "
          f"(val_anchor={shards['val_anchor'].count()}, "
          f"random shards={len(shards) - 1})", flush=True)

    print(f"[baseline] STEP 5: loading BASELINE model "
          f"(baseline_model={args.baseline_model}, local={args.local_model}) ...", flush=True)
    # Now load the BASELINE model with its own tokenizer/processor for decoding generated ids.
    model, _, baseline_tokenizer, baseline_processor = get_whisper_models(
        args.baseline_model,
        args.target_language,
        return_timestamps=False,
        load_in_8bit=False,
        local=args.local_model,
    )
    print("[baseline] STEP 5: baseline model loaded", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] STEP 6: moving model to {device} ...", flush=True)
    model.to(device)
    if device.type == "cuda":
        print("[baseline] STEP 6: casting to bf16 ...", flush=True)
        model = model.to(torch.bfloat16)
    model.eval()
    print("[baseline] STEP 6: model ready for eval", flush=True)

    # NOTE: the parquet labels were tokenized with the CrisperWhisper tokenizer at
    # preprocessing time. Whisper tokenizers across model variants share the same
    # vocabulary and special-token ids for the same language, so decoding labels with
    # the baseline processor is OK. If your CrisperWhisper preprocessing used a
    # nonstandard vocabulary, override via the env var BASELINE_DECODE_WITH_CRISPER=1
    # and we'll fall back to using the data-pipeline's tokenizer instead.
    if os.environ.get("BASELINE_DECODE_WITH_CRISPER", "") == "1":
        # The data-pipeline's get_whisper_models call already created a processor;
        # reload it here so we can decode labels with it.
        _, _, decode_tokenizer, decode_processor = get_whisper_models(
            "CrisperWhisper",
            args.target_language,
            return_timestamps=False,
            load_in_8bit=False,
            local=True,
        )
    else:
        decode_tokenizer = baseline_tokenizer
        decode_processor = baseline_processor

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Support partial re-runs
    shard_names = list(shards.keys())
    if args.shards_to_run:
        wanted = {s.strip() for s in args.shards_to_run.split(",") if s.strip()}
        shard_names = [n for n in shard_names if n in wanted]
        print(f"[baseline] running subset of shards (--shards_to_run): {shard_names}", flush=True)

    # If output exists, read existing rows so we don't clobber a partial run.
    existing: dict[str, dict] = {}
    if out_path.exists():
        with out_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "shard" in row:
                    existing[row["shard"]] = row
        print(f"[baseline] found {len(existing)} existing rows in {out_path}", flush=True)

    # Auto-skip already-completed shards unless the user asked for them explicitly
    if existing and not args.shards_to_run and not args.force:
        skipped = [n for n in shard_names if n in existing]
        if skipped:
            print(f"[baseline] auto-skipping {len(skipped)} already-completed shards "
                  f"(use --force to recompute): {skipped[:5]}{'...' if len(skipped) > 5 else ''}",
                  flush=True)
        shard_names = [n for n in shard_names if n not in existing]

    if not shard_names:
        print("[baseline] nothing to do; CSV is already complete.", flush=True)
        return

    print(f"[baseline] will compute {len(shard_names)} shard(s): {shard_names}", flush=True)

    new_rows: dict[str, dict] = dict(existing)
    for idx, name in enumerate(shard_names, start=1):
        shard = shards[name]
        n_shard = shard.count()
        print(f"[baseline] ({idx}/{len(shard_names)}) evaluating {name} (count={n_shard}) ...",
              flush=True)
        loss, wer, n_used = evaluate_shard(
            model,
            shard,
            data_collators["val"],
            decode_processor,
            decode_tokenizer,
            batch_size=args.per_device_eval_batch_size,
            prefetch_batches=args.prefetch_batches,
            max_length=args.max_length,
            language=args.gen_language,
            task=args.task,
            device=device,
            max_batches=args.max_batches_per_shard,
            shard_name=name,
            log_every=args.log_every,
        )
        print(f"[baseline] {name}: loss={loss:.4f} wer={wer:.4f} n={n_used}", flush=True)
        new_rows[name] = {
            "shard": name,
            "eval_loss": f"{loss:.6f}",
            "eval_wer": f"{wer:.6f}",
            "n_samples": str(n_used),
        }

        # Persist after every shard so a crash / wall-clock kill doesn't lose progress.
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["shard", "eval_loss", "eval_wer", "n_samples"])
            writer.writeheader()
            for k in sorted(new_rows.keys(), key=_shard_sort_key):
                writer.writerow(new_rows[k])
        print(f"[baseline] checkpointed CSV after {name} -> {out_path}", flush=True)

    print(f"[baseline] DONE. wrote {out_path}", flush=True)


def _shard_sort_key(name: str):
    # val_anchor first, then val_1, val_2, ... in numeric order
    if name == "val_anchor":
        return (0, 0)
    if name.startswith("val_"):
        try:
            return (1, int(name.split("_")[-1]))
        except ValueError:
            return (2, name)
    return (3, name)


if __name__ == "__main__":
    main()

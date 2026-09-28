#!/usr/bin/env python3
"""
Compute pretrained-model baseline WER on val shards that exactly match the
shard boundaries train_hyper.py will use at a given --eval_sample_fraction.

Why this script exists (not compute_baseline_wer.py): compute_baseline_wer.py
assumes one shard == one physical val parquet file, which only lines up with
train_hyper.py's partition_dataset() when eval_sample_fraction happens to
produce exactly len(parquet_files) equal shards (true for tiny/small/medium
at 0.1 -> 10 shards == 10 parquet files). At other fractions (e.g. 0.05 -> 20
shards) that assumption breaks, since partition_dataset splits the whole
concatenated dataset by row count, not by file. This script instead loads the
val dataset the same way train_hyper.py does and calls the same
partition_dataset() function, so shard_i here is byte-identical to shard_i at
train time. Decoding (fp16, greedy, jiwer WER, same normalize()) matches
compute_baseline_wer.py so results stay comparable to the existing baselines.

Usage:
    python scripts/compute_baseline_wer_shard_matched.py \
        --model_type whisper-tiny --eval_sample_fraction 0.05

Output:
    trainers/data/validation_summary_<tag>_frac<pct>.csv
    columns: shard, eval_loss, eval_wer, n_samples
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from jiwer import wer as jiwer_wer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ray

from finetuning.data_and_collator.datasets_and_collators import get_datasets_and_collators, make_dataset_kwargs
from finetuning.models.whisper_models import get_whisper_models
from finetuning.train_hyper import partition_dataset
from finetuning.projects_paths import DATA_PATH

_TAG_MAP = {
    "whisper-tiny": "tiny",
    "whisper-base": "base",
    "whisper-small": "ws",
    "whisper-medium": "wm",
    "whisper-large": "wlv2",
    "whisper-large-v2": "wlv2",
    "whisper-large-v3": "wlv3",
}


def normalize(text):
    def _one(t):
        return re.sub(r"[!\?\.,;]", "", t.strip().lower())
    if isinstance(text, list):
        return [_one(t) for t in text]
    return _one(text)


def to_feat_tensor(f, n_mels):
    a = np.array(f, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(n_mels, -1)
    elif a.ndim == 2 and a.shape[0] != n_mels:
        a = a.T
    return a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", type=str, default="whisper-tiny",
                    help="subdirectory under MODEL_PATH containing the model")
    p.add_argument("--dataset_name", type=str, default="eq_complete_openai_whisper-small")
    p.add_argument("--eval_sample_fraction", type=float, required=True,
                    help="Must match the eval_sample_fraction in the HPO config, "
                         "e.g. 0.05 -> partition_dataset produces val_1..val_20.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--language", type=str, default="german")
    p.add_argument("--task", type=str, default="transcribe")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--cpus_per_trial", type=int, default=2)
    p.add_argument("--random_seed", type=int, default=1337)
    p.add_argument("--run_on_local_machine", action="store_true", default=True)
    p.add_argument("--peft", action="store_true", default=True)
    p.add_argument("--output_csv", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    IS_LARGE = any(x in args.model_type for x in ("large", "Large"))
    N_MELS = 128 if IS_LARGE else 80

    tag = _TAG_MAP.get(args.model_type, args.model_type.replace("/", "_").replace("-", "_"))
    frac_tag = f"frac{args.eval_sample_fraction:.3f}".rstrip("0").rstrip(".")
    output_csv = args.output_csv or str(
        PROJECT_ROOT / "finetuning" / "trainers" / "data" / f"validation_summary_{tag}_{frac_tag}.csv"
    )

    print(f"model_type           : {args.model_type}")
    print(f"dataset_name          : {args.dataset_name}")
    print(f"eval_sample_fraction  : {args.eval_sample_fraction}")
    print(f"n_mels                : {N_MELS}")
    print(f"output_csv            : {output_csv}\n")

    print("[baseline] ray init ...")
    if not ray.is_initialized():
        ray.init(
            num_cpus=int(os.environ.get("RAY_NUM_CPUS", "2")),
            include_dashboard=False,
            log_to_driver=True,
            _temp_dir=os.environ.get("TMPDIR", "/tmp"),
            ignore_reinit_error=True,
        )

    data_args = SimpleNamespace(
        cpus_per_trial=args.cpus_per_trial,
        random_seed=args.random_seed,
        model_type=args.model_type,
        target_language=args.language,
        return_timestamps=False,
        run_on_local_machine=args.run_on_local_machine,
        path_to_data=DATA_PATH,
        dataset_name=args.dataset_name,
        peft=args.peft,
        debug=False,
        data_mode="parquet",
    )
    dataset_kwargs = make_dataset_kwargs(data_args)
    print(f"[baseline] loading val dataset the same way train_hyper.py does "
          f"(dataset_kwargs={dataset_kwargs}) ...")
    ray_datasets, _ = get_datasets_and_collators(dataset_kwargs)
    val_ds = ray_datasets["val"]

    print("[baseline] partition_dataset() -- same function train_hyper.py calls at train time ...")
    shards = partition_dataset(val_ds, fraction=args.eval_sample_fraction)
    print(f"[baseline] produced {len(shards)} shards: {sorted(shards.keys(), key=lambda n: int(n.split('_')[1]))}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[baseline] loading pretrained {args.model_type} on {device} ...")
    model, _, tokenizer, processor = get_whisper_models(
        args.model_type, args.language, return_timestamps=False, load_in_8bit=False, local=True,
    )
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    print("[baseline] model ready.\n")

    results = []
    for shard_name in sorted(shards.keys(), key=lambda n: int(n.split("_")[1])):
        ds = shards[shard_name]
        df = ds.to_pandas()
        n = len(df)
        print(f"=== {shard_name} (n={n}) ===")

        all_refs, all_hyps = [], []
        n_batches = (n + args.batch_size - 1) // args.batch_size

        for bi in range(n_batches):
            rows = df.iloc[bi * args.batch_size: (bi + 1) * args.batch_size]

            feats_np = np.stack([to_feat_tensor(f, N_MELS) for f in rows["input_features"].tolist()])
            input_features = torch.from_numpy(feats_np).to(device, dtype=torch.float16)

            pad_id = processor.tokenizer.pad_token_id
            decoded_refs = []
            for lab in rows["labels"].tolist():
                lab_clean = [t if t != -100 else pad_id for t in lab]
                decoded_refs.append(processor.tokenizer.decode(lab_clean, skip_special_tokens=True))
            all_refs.extend(normalize(decoded_refs))

            with torch.no_grad():
                pred_ids = model.generate(
                    input_features,
                    language=args.language,
                    task=args.task,
                    num_beams=args.num_beams,
                    return_timestamps=False,
                )
            all_hyps.extend(normalize(
                processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            ))

            if (bi + 1) % 10 == 0 or bi == n_batches - 1:
                print(f"  batch {bi + 1}/{n_batches}")

        wer_val = 100.0 * jiwer_wer(all_refs, all_hyps)
        print(f"  WER = {wer_val:.4f}%\n")
        results.append({"shard": shard_name, "eval_loss": "",
                         "eval_wer": round(wer_val, 6), "n_samples": n})

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shard", "eval_loss", "eval_wer", "n_samples"])
        writer.writeheader()
        writer.writerows(results)

    print(f"saved {len(results)} rows -> {out_path}")
    for r in results:
        print(f"  {r['shard']}: WER={r['eval_wer']:.4f}%  n={r['n_samples']}")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "peft>=0.13",
#   "transformers>=4.45",
#   "torch>=2.4",
#   "ctranslate2>=4.4",
#   "accelerate",
#   "safetensors",
#   "sentencepiece",
# ]
# ///
"""Convert LoRA/AdaLoRA Whisper adapters to Faster-Whisper (CT2) format.

Per job: load base model -> merge adapter -> save merged HF model to temp dir
-> ct2-transformers-converter -> Faster-Whisper output directory.

Usage:

  # Single job via CLI
  uv run --script scripts/convert_lora_to_faster_whisper.py \\
      --adapter /path/to/checkpoint \\
      --base openai/whisper-large-v3 \\
      --output /path/to/output_fw

  # Multiple jobs from TOML (see convert_jobs.example.toml)
  uv run --script scripts/convert_lora_to_faster_whisper.py \\
      --config scripts/convert_jobs.toml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

DEFAULT_QUANTIZATION = "float16"

# Files ct2-transformers-converter should copy into the output - only those
# that actually exist in merged_dir after save_pretrained().
CANDIDATE_COPY_FILES = [
    "tokenizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "generation_config.json",
]


@dataclass(frozen=True)
class Job:
    adapter_dir: Path
    base_model: str
    output_dir: Path
    quantization: str = DEFAULT_QUANTIZATION
    # Optional: HF repo id or local path whose tokenizer.json + vocabulary.json
    # + preprocessor_config.json override the tokenizer extracted from the base
    # model. Required e.g. for CrisperWhisper: its HF base repo lacks
    # tokenizer.json, and the dynamically built fast tokenizer places
    # added_tokens at the wrong indices (startoftranscript ends up on the
    # wrong id).
    tokenizer_override: str | None = None


def merge_adapter(job: Job, merged_dir: Path) -> None:
    print(f"  [merge] loading base: {job.base_model}")
    base = WhisperForConditionalGeneration.from_pretrained(
        job.base_model, torch_dtype=torch.float32
    )
    print(f"  [merge] loading adapter: {job.adapter_dir}")
    peft_model = PeftModel.from_pretrained(base, str(job.adapter_dir))
    print("  [merge] merge_and_unload …")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)

    try:
        processor = WhisperProcessor.from_pretrained(str(job.adapter_dir))
        print("  [merge] processor from adapter dir.")
    except Exception as exc:
        print(f"  [merge] adapter processor failed ({exc}); using base processor.")
        processor = WhisperProcessor.from_pretrained(job.base_model)
    processor.save_pretrained(str(merged_dir))

    # WhisperProcessor.save_pretrained() does not reliably write
    # preprocessor_config.json (the feature extractor part may be missing
    # depending on how the processor was loaded). Save it explicitly from
    # the base model as a defensive fallback.
    if not (merged_dir / "preprocessor_config.json").exists():
        WhisperFeatureExtractor.from_pretrained(job.base_model).save_pretrained(str(merged_dir))
        print("  [merge] preprocessor_config.json added as fallback.")


def convert_to_ct2(merged_dir: Path, job: Job) -> None:
    if job.output_dir.exists():
        shutil.rmtree(job.output_dir)
    copy_files = [f for f in CANDIDATE_COPY_FILES if (merged_dir / f).exists()]
    cmd = [
        "ct2-transformers-converter",
        "--model", str(merged_dir),
        "--output_dir", str(job.output_dir),
        "--quantization", job.quantization,
        "--copy_files", *copy_files,
    ]
    print(f"  [ct2] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def apply_tokenizer_override(job: Job) -> None:
    """Overwrite tokenizer files in the output with those from job.tokenizer_override."""
    src = Path(job.tokenizer_override)
    if not src.exists():
        # Treat as HF repo id - fetch from cache or download.
        src = Path(snapshot_download(
            repo_id=job.tokenizer_override,
            allow_patterns=["tokenizer.json", "vocabulary.json", "preprocessor_config.json"],
        ))
    overridden = []
    for f in ("tokenizer.json", "vocabulary.json", "preprocessor_config.json"):
        if (src / f).exists():
            shutil.copy(src / f, job.output_dir / f)
            overridden.append(f)
    print(f"  [override] from {job.tokenizer_override}: {', '.join(overridden) or '(nothing)'}")


def run_job(job: Job, force: bool) -> None:
    if not job.adapter_dir.exists():
        print(f"!!  adapter path missing: {job.adapter_dir}", file=sys.stderr)
        sys.exit(1)
    if job.output_dir.exists() and not force:
        print(f"==> {job.output_dir} already exists - skipping (use --force to overwrite).")
        return
    job.output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> {job.adapter_dir.name}  ->  {job.output_dir}")
    with tempfile.TemporaryDirectory(prefix="merged_") as td:
        merged_dir = Path(td)
        merge_adapter(job, merged_dir)
        convert_to_ct2(merged_dir, job)
    if job.tokenizer_override:
        apply_tokenizer_override(job)
    print(f"==> done: {job.output_dir}\n")


def load_jobs_from_toml(path: Path) -> list[Job]:
    """Parse a TOML file with [[jobs]] tables.

    [[jobs]]
    adapter = "/path/to/checkpoint"
    base    = "openai/whisper-large-v3"
    output  = "/path/to/output_fw"
    quantization = "float16"        # optional
    tokenizer_override = "repo_id"  # optional
    """
    data = tomllib.loads(path.read_text())
    raw = data.get("jobs", [])
    if not raw:
        sys.exit(f"No [[jobs]] found in {path}.")
    jobs = []
    for i, j in enumerate(raw):
        try:
            jobs.append(
                Job(
                    adapter_dir=Path(j["adapter"]).expanduser(),
                    base_model=j["base"],
                    output_dir=Path(j["output"]).expanduser(),
                    quantization=j.get("quantization", DEFAULT_QUANTIZATION),
                    tokenizer_override=j.get("tokenizer_override"),
                )
            )
        except KeyError as e:
            sys.exit(f"Job #{i} in {path} is missing required field: {e}")
    return jobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--adapter", type=Path, help="Path to the PEFT checkpoint")
    p.add_argument("--base", help="HF repo id or local path to the base Whisper model")
    p.add_argument("--output", type=Path, help="Target directory for the Faster-Whisper model")
    p.add_argument(
        "--quantization", default=DEFAULT_QUANTIZATION,
        help=f"CT2 quantization (default: {DEFAULT_QUANTIZATION})",
    )
    p.add_argument("--config", type=Path, help="TOML file with multiple jobs")
    p.add_argument(
        "--tokenizer-override",
        help="HF repo or path whose tokenizer files override the output "
             "(e.g. nyrahealth/faster_CrisperWhisper)",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        jobs = load_jobs_from_toml(args.config)
    elif args.adapter and args.base and args.output:
        jobs = [Job(
            adapter_dir=args.adapter.expanduser(),
            base_model=args.base,
            output_dir=args.output.expanduser(),
            quantization=args.quantization,
            tokenizer_override=args.tokenizer_override,
        )]
    else:
        sys.exit(
            "Either pass --config FILE or --adapter + --base + --output."
        )

    for j in jobs:
        run_job(j, force=args.force)


if __name__ == "__main__":
    main()

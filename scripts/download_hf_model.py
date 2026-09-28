#!/usr/bin/env python3
"""Download a Whisper (or CrisperWhisper) model from the HuggingFace Hub into the
local directory layout this repo expects:

    <output_dir>/
        model/
        processor/
        feature_extractor/
        tokenizer/

Point ``MODEL_PATH`` at the parent directory and pass
``--output_dir "$MODEL_PATH/<model_type>"``; ``models/whisper_models.py`` then
loads the four components from those sub-directories.

Examples:
    python src/scripts/download_hf_model.py \\
        --model_id openai/whisper-large-v3 \\
        --output_dir "$MODEL_PATH/whisper-large-v3"

    python src/scripts/download_hf_model.py \\
        --model_id nyrahealth/CrisperWhisper \\
        --output_dir "$MODEL_PATH/CrisperWhisper"
"""
import argparse
import os

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model_id",
        default="openai/whisper-large-v3",
        help="HuggingFace model id (e.g. openai/whisper-large-v3, nyrahealth/CrisperWhisper).",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory; sub-dirs model/ processor/ feature_extractor/ "
             "tokenizer/ are written here.",
    )
    ap.add_argument(
        "--cache_dir",
        default=None,
        help="Optional HuggingFace download cache directory (defaults to --output_dir).",
    )
    args = ap.parse_args()

    out = args.output_dir
    cache_dir = args.cache_dir or out
    os.makedirs(out, exist_ok=True)

    AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, cache_dir=cache_dir).save_pretrained(
        os.path.join(out, "model")
    )
    AutoProcessor.from_pretrained(args.model_id, cache_dir=cache_dir).save_pretrained(
        os.path.join(out, "processor")
    )
    AutoFeatureExtractor.from_pretrained(args.model_id, cache_dir=cache_dir).save_pretrained(
        os.path.join(out, "feature_extractor")
    )
    AutoTokenizer.from_pretrained(args.model_id, cache_dir=cache_dir).save_pretrained(
        os.path.join(out, "tokenizer")
    )

    print(f"All components saved to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

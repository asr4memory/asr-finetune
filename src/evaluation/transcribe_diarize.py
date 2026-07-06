import os
import sys
import json
import glob
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path

# Make src/ importable whether this file is run via ``python -m`` or directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import configargparse
import soundfile as sf
import torch
from transformers import set_seed, pipeline
from pyannote.audio import Pipeline as DiarizationPipeline

from models.whisper_models import get_whisper_models_from_dir, get_whisper_models_from_hub

logger = logging.getLogger(__name__)


def check_binary(name, configured_path):
    path = shutil.which(configured_path) if not os.path.isabs(configured_path) else configured_path
    if path is None or not os.path.exists(path):
        raise RuntimeError(
            f"Required executable not found: {configured_path}. "
            f"Set the correct path in the config file."
        )
    return path


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", is_config_file=True, type=str)

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--output_tag", type=str, default="whisper_diarized")

    parser.add_argument("--model_type", type=str, default="whisper-large-v3")
    parser.add_argument("--target_language", type=str, default="german")
    parser.add_argument("--task", type=str, choices=["transcribe", "translate"], default="transcribe")

    parser.add_argument("--diarization_model", type=str, required=True,
                        help="Local directory of pyannote/speaker-diarization-3.1 (or HF id if online).")
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN",
                        help="Env var holding HuggingFace token. Only needed if diarization_model is an HF id.")

    parser.add_argument("--run_on_local_machine", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument("--chunk_length_s", type=int, default=30)
    parser.add_argument("--stride_length_s", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--ffmpeg_bin", type=str, default="ffmpeg")
    parser.add_argument("--ffprobe_bin", type=str, default="ffprobe")

    parser.add_argument("--audio_glob", type=str, default="*.mp3")
    parser.add_argument("--random_seed", type=int, default=1337)

    return parser.parse_args()


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def atomic_write_json(data, out_path):
    tmp_path = str(out_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)


def atomic_write_text(text, out_path):
    tmp_path = str(out_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, out_path)


def get_audio_duration_seconds(audio_path, ffprobe_bin):
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def convert_to_wav_16k_mono(audio_path, out_wav_path, ffmpeg_bin):
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(audio_path),
        "-ar", "16000",
        "-ac", "1",
        str(out_wav_path),
    ]
    subprocess.run(cmd, check=True)


def make_output_paths(audio_path, input_dir, output_root):
    rel_path = os.path.relpath(audio_path, input_dir)
    rel_stem = os.path.splitext(rel_path)[0]

    json_path = os.path.join(output_root, rel_stem + ".json")
    txt_path = os.path.join(output_root, rel_stem + ".txt")

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    return json_path, txt_path


def build_whisper_pipeline(args, device):
    if args.run_on_local_machine:
        model, feature_extractor, tokenizer, processor = get_whisper_models_from_dir(
            args.model_type,
            args.target_language,
            task=args.task,
            return_timestamps=False,
            load_in_8bit=False,
        )
    else:
        model, feature_extractor, tokenizer, processor = get_whisper_models_from_hub(
            args.model_type,
            args.target_language,
            task=args.task,
            return_timestamps=False,
            load_in_8bit=False,
        )

    use_fp16 = device.type == "cuda" and args.fp16
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    model = model.to(device)
    if use_fp16:
        model = model.half()
    model.eval()

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        chunk_length_s=args.chunk_length_s,
        stride_length_s=args.stride_length_s,
        return_timestamps="word",
        device=device,
        torch_dtype=torch_dtype,
    )
    return asr


def build_diarization_pipeline(args, device):
    kwargs = {}
    if args.hf_token_env:
        token = os.environ.get(args.hf_token_env)
        if token:
            kwargs["token"] = token

    pipe = DiarizationPipeline.from_pretrained(args.diarization_model, **kwargs)
    if pipe is None:
        raise RuntimeError(
            f"Failed to load diarization pipeline from {args.diarization_model}. "
            f"Check the path / HF terms acceptance / token."
        )
    if device.type == "cuda":
        pipe.to(device)
    return pipe


def diarize(diar_pipe, wav_path, num_speakers):
    # Bypass pyannote's built-in audio loader (which uses torchcodec and
    # often fails on HPC due to FFmpeg shared-lib version mismatch).
    # We already converted to 16 kHz mono WAV with the ffmpeg CLI, so just
    # read it with soundfile and hand pyannote a preloaded waveform dict.
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    diarization = diar_pipe(
        {"waveform": waveform, "sample_rate": sr},
        num_speakers=num_speakers,
    )
    # Newer pyannote-audio returns a DiarizeOutput wrapper; older versions
    # return an Annotation directly. Unwrap if needed.
    if not hasattr(diarization, "itertracks"):
        diarization = getattr(diarization, "speaker_diarization", None) \
            or getattr(diarization, "diarization", None) \
            or diarization[0]
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    turns.sort(key=lambda t: t["start"])
    return turns


def transcribe_with_word_timestamps(asr, wav_path, args):
    generate_kwargs = {
        "language": args.target_language,
        "task": args.task,
        "max_new_tokens": args.max_new_tokens,
    }
    result = asr(
        str(wav_path),
        batch_size=args.batch_size,
        generate_kwargs=generate_kwargs,
        return_timestamps="word",
    )

    words = []
    last_end = 0.0
    for chunk in result.get("chunks", []):
        ts = chunk.get("timestamp") or (None, None)
        start, end = ts
        if start is None:
            start = last_end
        if end is None:
            end = start
        text = chunk.get("text", "").strip()
        if not text:
            last_end = end
            continue
        words.append({
            "start": float(start),
            "end": float(end),
            "text": text,
        })
        last_end = end
    return words, result.get("text", "").strip()


def find_speaker_for_time(turns, t):
    if not turns:
        return None
    best_speaker = None
    best_dist = float("inf")
    for turn in turns:
        if turn["start"] <= t <= turn["end"]:
            return turn["speaker"]
        d = turn["start"] - t if t < turn["start"] else t - turn["end"]
        if d < best_dist:
            best_dist = d
            best_speaker = turn["speaker"]
    return best_speaker


def assign_speakers_and_group(words, turns):
    segments = []
    current = None
    for w in words:
        midpoint = (w["start"] + w["end"]) / 2.0
        speaker = find_speaker_for_time(turns, midpoint) or "UNKNOWN"

        if current is None or current["speaker"] != speaker:
            if current is not None:
                segments.append(current)
            current = {
                "speaker": speaker,
                "start": w["start"],
                "end": w["end"],
                "text": w["text"],
            }
        else:
            current["end"] = w["end"]
            current["text"] = (current["text"] + " " + w["text"]).strip()

    if current is not None:
        segments.append(current)
    return segments


def format_diarized_text(segments):
    lines = []
    for seg in segments:
        lines.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}: {seg['text']}")
    return "\n\n".join(lines)


def process_one_file(audio_path, args, asr, diar_pipe, device):
    output_root = os.path.join(args.output_dir, args.output_tag)
    json_path, txt_path = make_output_paths(audio_path, args.input_dir, output_root)

    if os.path.exists(json_path) and not args.overwrite:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("status") == "done":
                logger.info("Skipping finished file: %s", audio_path)
                return "skipped"
        except Exception:
            pass

    duration_s = get_audio_duration_seconds(audio_path, args.ffprobe_bin)

    progress = {
        "status": "running",
        "input_file": str(audio_path),
        "file_name": os.path.basename(audio_path),
        "duration_s": duration_s,
        "task": args.task,
        "target_language": args.target_language,
        "model_type": args.model_type,
        "num_speakers": args.num_speakers,
        "diarization_turns": [],
        "segments": [],
        "full_text": "",
    }
    atomic_write_json(progress, json_path)

    with tempfile.TemporaryDirectory(prefix="diarize_") as tmp_dir:
        wav_path = os.path.join(tmp_dir, "audio_16k_mono.wav")
        logger.info("Converting %s -> 16kHz mono WAV", audio_path)
        convert_to_wav_16k_mono(audio_path, wav_path, args.ffmpeg_bin)

        logger.info("Running diarization on %s", audio_path)
        turns = diarize(diar_pipe, wav_path, args.num_speakers)
        progress["diarization_turns"] = turns
        progress["status"] = "diarized"
        atomic_write_json(progress, json_path)
        logger.info("Found %d diarization turns", len(turns))

        logger.info("Running ASR on %s", audio_path)
        words, full_text = transcribe_with_word_timestamps(asr, wav_path, args)
        logger.info("Got %d words", len(words))

        segments = assign_speakers_and_group(words, turns)

    progress["segments"] = segments
    progress["full_text"] = full_text
    progress["status"] = "done"

    atomic_write_json(progress, json_path)
    atomic_write_text(format_diarized_text(segments), txt_path)

    logger.info("Finished file: %s", audio_path)
    return "done"


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.random_seed)

    args.ffmpeg_bin = check_binary("ffmpeg", args.ffmpeg_bin)
    args.ffprobe_bin = check_binary("ffprobe", args.ffprobe_bin)

    logger.info("Using ffmpeg: %s", args.ffmpeg_bin)
    logger.info("Using ffprobe: %s", args.ffprobe_bin)

    device = select_device()
    logger.info("Using device: %s", device)

    logger.info("Building Whisper pipeline...")
    asr = build_whisper_pipeline(args, device)

    logger.info("Building diarization pipeline from %s", args.diarization_model)
    diar_pipe = build_diarization_pipeline(args, device)

    all_audio_files = sorted(glob.glob(os.path.join(args.input_dir, "**", args.audio_glob), recursive=True))
    logger.info("Found %d matching files", len(all_audio_files))

    stats = {"done": 0, "skipped": 0, "failed": 0}

    for audio_path in all_audio_files:
        try:
            status = process_one_file(
                audio_path=audio_path,
                args=args,
                asr=asr,
                diar_pipe=diar_pipe,
                device=device,
            )
            stats[status] += 1
        except Exception as e:
            logger.exception("Failed file: %s | error=%s", audio_path, repr(e))
            stats["failed"] += 1

    logger.info("All done: %s", stats)


if __name__ == "__main__":
    main()

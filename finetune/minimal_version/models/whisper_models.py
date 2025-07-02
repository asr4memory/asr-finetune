"""
Model Loader Utilities for Whisper ASR fine-tuning with Hugging Face Transformers.

This module provides convenience functions to load Whisper models and associated components
(e.g., tokenizer, processor, feature extractor) from either local disk or Hugging Face Hub.
The loading is designed to be Ray-compatible and flexible for both PEFT and full fine-tuning workflows.
"""
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from projects_paths import MODEL_PATH

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration


def get_whisper_models_from_dir(model_type,target_language,return_timestamps=False, load_in_8bit=False):
    """
    Load Whisper model and components from a custom directory (e.g., pre-downloaded local models).

    This function assumes that each component (model, tokenizer, processor, etc.) is saved in a subfolder
    under a root `MODEL_PATH` defined in the project.

    Args:
        model_type (str): Name or identifier for the Whisper model variant (e.g., 'openai/whisper-tiny').
        target_language (str): The language code (e.g., 'german') for tokenization and generation.
        return_timestamps (bool): Whether the model should generate timestamp outputs.
        load_in_8bit (bool): Whether to load the model in 8-bit precision (for memory efficiency).

    Returns:
        Tuple: (model, feature_extractor, tokenizer, processor)
    """
    model_dir = MODEL_PATH
    
    # Load components from local file system (assumes correct folder structure)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"{model_dir}/feature_extractor", local_files_only=True,load_in_8bit=load_in_8bit)
    tokenizer = WhisperTokenizer.from_pretrained(f"{model_dir}/tokenizer", local_files_only=True, language=target_language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(f"{model_dir}/processor", local_files_only=True, language=target_language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(f"{model_dir}/model", local_files_only=True)

    # Set decoding configuration for transcription task
    model.generation_config.language = target_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.return_timestamps = return_timestamps

    return model, feature_extractor, tokenizer, processor


def get_whisper_models_local(model_type,target_language,return_timestamps=False, load_in_8bit=False):
    """
    Load Whisper model and components directly from Hugging Face model hub.

    Args:
        model_type (str): Hugging Face model name (e.g., 'openai/whisper-tiny').
        target_language (str): The spoken language for training and decoding.
        return_timestamps (bool): Whether to return timestamps in generation.
        load_in_8bit (bool): Whether to load model weights in 8-bit precision.

    Returns:
        Tuple: (model, feature_extractor, tokenizer, processor)
    """
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type,load_in_8bit=load_in_8bit)
    tokenizer = WhisperTokenizer.from_pretrained(model_type, language=target_language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_type, language=target_language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_type, load_in_8bit=load_in_8bit)
    
    # Set generation configuration
    model.generation_config.language = target_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.return_timestamps = return_timestamps

    return model, feature_extractor, tokenizer, processor

def get_whisper_models(model_type,
                       target_language,
                       return_timestamps=False,
                       load_in_8bit=False,
                       local=False
                       ):
    """
    Load Whisper model and components either from HF Hub or local project directory.

    This is a wrapper function that dispatches to either the Hugging Face loading function
    (`get_whisper_models_local`) or the custom directory loader (`get_whisper_models_from_dir`).

    Args:
        model_type (str): Model identifier (local path or HF model name).
        target_language (str): The spoken language for ASR.
        return_timestamps (bool): Enable timestamp prediction in model outputs.
        load_in_8bit (bool): Load the model in 8-bit precision (optional).
        local (bool): If True, load from Hugging Face; otherwise load from local disk.

    Returns:
        Tuple: (model, feature_extractor, tokenizer, processor)
    """
    if local:
        model, feature_extractor, tokenizer, processor = get_whisper_models_local(model_type,
                                                                                  target_language,
                                                                                  return_timestamps=return_timestamps,
                                                                                  load_in_8bit=load_in_8bit,
                                                                                  )
    else:
        model, feature_extractor, tokenizer, processor = get_whisper_models_from_dir(model_type,
                                                                                  target_language,
                                                                                  return_timestamps=return_timestamps,
                                                                                  load_in_8bit=load_in_8bit,
                                                                                  )
    return model, feature_extractor, tokenizer, processor

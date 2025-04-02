"""Collections of different model loaders"""

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
import os

def get_whisper_models(model_type,target_language,return_timestamps=False, load_in_8bit=False):
    """Loads Features Extractor, Tokenizer, Processecor and Model.

    Ray requires to load the model and datasets within the train_model function. Therefore, such a function for loading
    the models is required.

    Args:
       model_type (str): The model type to fine-tune.
       target_language (str): The spoken language of the training set.
    Returns:
       model (WhisperForConditionalGeneration): loaded Whisper Object
       features_extractor (WhisperFeatureExtractor): loaded Whisper Feature Extractor Object
       tokenizer (WhisperTokenizer): loaded Whiseper Tokenizer Object
       processor (WhisperProcessor): Loaded Whisper Processor Object
    """
    model_dir = os.path.join("/scratch/usr/",os.getenv('USER')+"/whisper-large-v3")
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"{model_dir}/feature_extractor", local_files_only=True,load_in_8bit=load_in_8bit)
    tokenizer = WhisperTokenizer.from_pretrained(f"{model_dir}/tokenizer", local_files_only=True, language=target_language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(f"{model_dir}/processor", local_files_only=True, language=target_language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(f"{model_dir}/model", local_files_only=True)
#    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type,load_in_8bit=load_in_8bit)
#    tokenizer = WhisperTokenizer.from_pretrained(model_type, language=target_language, task="transcribe")
#    processor = WhisperProcessor.from_pretrained(model_type, language=target_language, task="transcribe")
#    model = WhisperForConditionalGeneration.from_pretrained(model_type)
#    model.half()
    model.generation_config.language = target_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.return_timestamps = return_timestamps

    return model, feature_extractor, tokenizer, processor

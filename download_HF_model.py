# Run this on a machine with internet access
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq
import os

# Create a specific cache directory
cache_dir = "./whisper-large-v3"
os.makedirs(cache_dir, exist_ok=True)

# Download all components explicitly
model_id = "openai/whisper-large-v3"

# Download model
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=cache_dir)
model.save_pretrained(f"{cache_dir}/model")

# Download processor (contains feature extractor)
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
processor.save_pretrained(f"{cache_dir}/processor")

# Download feature extractor explicitly
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
feature_extractor.save_pretrained(f"{cache_dir}/feature_extractor")

# Download tokenizer explicitly
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
tokenizer.save_pretrained(f"{cache_dir}/tokenizer")

print(f"All components saved to: {os.path.abspath(cache_dir)}")

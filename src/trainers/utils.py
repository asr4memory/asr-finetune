import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback
import torch
from transformers import TrainingArguments, TrainerState, TrainerControl


# ------------------------------------------------------------------------------
# Callback to Reload PEFT Adapter Weights From the `adapter_model/` Subdirectory
# ------------------------------------------------------------------------------
class LoadAdapterFromSubdirCallback(TrainerCallback):
    """
    Reload PEFT adapter weights from `<checkpoint>/adapter_model/` after HF
    Trainer's own resume-load has run.

    Why: `SavePeftModelCallback` writes the real adapter into the
    `adapter_model/` subdirectory, while HF Trainer's PEFT auto-resume looks
    at `adapter_model.safetensors` at the checkpoint root - which under
    DeepSpeed ZeRO-3 saves is a ~40-byte empty placeholder. Without this
    callback, resume silently overwrites the adapter with empty weights and
    training restarts from base Whisper.
    """

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None or not self.checkpoint_path:
            return
        adapter_dir = os.path.join(self.checkpoint_path, "adapter_model")
        if not os.path.isdir(adapter_dir):
            print(f"[LoadAdapterFromSubdirCallback] No adapter_model/ at {adapter_dir}; skipping.")
            return

        safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        bin_path = os.path.join(adapter_dir, "adapter_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            src = safetensors_path
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
            src = bin_path
        else:
            print(f"[LoadAdapterFromSubdirCallback] No adapter weights found in {adapter_dir}; skipping.")
            return

        from peft.utils.save_and_load import set_peft_model_state_dict
        try:
            result = set_peft_model_state_dict(model, state_dict)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(
                    f"[LoadAdapterFromSubdirCallback] Rank mismatch between checkpoint "
                    f"and current model — skipping adapter load. {e}"
                )
                return
            raise
        missing = getattr(result, "missing_keys", None) or []
        unexpected = getattr(result, "unexpected_keys", None) or []
        print(
            f"[LoadAdapterFromSubdirCallback] Loaded {len(state_dict)} tensors from {src}. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[LoadAdapterFromSubdirCallback] First missing: {missing[:5]}")
        if unexpected:
            print(f"[LoadAdapterFromSubdirCallback] First unexpected: {unexpected[:5]}")

# ------------------------------------------------------------------------------
# Callback to Save Only Adapter Weights (e.g. for PEFT/LoRA)
# ------------------------------------------------------------------------------
class SavePeftModelCallback(TrainerCallback):
    """
    HuggingFace Trainer callback to save only the adapter model (e.g. LoRA weights)
    and remove the base model weights from checkpoints to save disk space.
    """
    
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

# ------------------------------------------------------------------------------
# Callback to Synchronize Trainer State from Previous Checkpoint
# ------------------------------------------------------------------------------

class StepSyncCallback(TrainerCallback):
    """
    Callback to synchronize the training step counter (`state.global_step`)
    with a previously saved checkpoint, for seamless resumption.
    """
    
    def __init__(self, starting_step):
        self.starting_step = starting_step
        self.has_synced = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.starting_step > 0 and not self.has_synced:
            print(f"Synchronizing step counter to {self.starting_step}")
            # Update the trainer's step counter
            state.global_step = self.starting_step
            self.has_synced = True

# ------------------------------------------------------------------------------
# Checkpoint Loader Utility
# ------------------------------------------------------------------------------

def load_checkpoints(checkpoint_dir):
    """
    Loads a HuggingFace trainer_state.json file to resume training from a checkpoint.

    Args:
        checkpoint_dir: Ray Train Checkpoint object from ray.train.get_checkpoint().

    Returns:
        trainer_state (dict): The parsed trainer_state.json
        starting_step (int): The global step at which to resume training
        resume_from_checkpoint (str): Path to checkpoint folder, or None
    """
    import json
    resume_from_checkpoint = None
    starting_step = 0
    try:
        # Ray 2.7+: Checkpoint.path was removed; use as_directory() instead.
        # For local checkpoints (SLURM scratch), as_directory() returns the
        # actual directory path without copying, so the path remains valid
        # after the context exits.
        with checkpoint_dir.as_directory() as ckpt_path:
            trainer_state_path = os.path.join(ckpt_path, "checkpoint", "trainer_state.json")
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                starting_step = trainer_state["global_step"]
                print(f"Will resume from step {starting_step}")
                resume_from_checkpoint = os.path.join(ckpt_path, "checkpoint")
                return trainer_state, starting_step, resume_from_checkpoint
            else:
                print(f"Path does not exist: {trainer_state_path}")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    return {}, starting_step, resume_from_checkpoint



# ------------------------------------------------------------------------------
# Identity Data Collator for Ray Integration (No Double Collation)
# ------------------------------------------------------------------------------

def data_collator_id(batch):
    """
    Identity data collator that simply moves tensors to the correct device
    without further collation. Used when Ray already collated the batch.

    Args:
        batch (dict): A batch of already-prepared inputs from Ray DataLoader.

    Returns:
        dict: Batch with tensors moved to the appropriate CUDA device.
    """
    local_rank = int(os.getenv("LOCAL_RANK", 0))  # Fallback to 0 if not set
    return {
        k: v.to(f"cuda:{local_rank}") if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

import re
def normalize(text):
    """
    Removes certain characters from text and lowers cases.

    Args:
        text (str or list of str): Single string or list of strings to be normalized.

    Returns:
        str or list of str: Normalized string or list of normalized strings.
    """
    def process_single_text(single_text):
        result = single_text.strip().lower()
        result = re.sub(r"[!\?\.,;]", "", result)
        return result

    if isinstance(text, list):
        return [process_single_text(t) for t in text]
    elif isinstance(text, str):
        return process_single_text(text)
    else:
        raise TypeError("Input must be a string or a list of strings.")

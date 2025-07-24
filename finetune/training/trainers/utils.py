import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback
import torch
from transformers import TrainingArguments, TrainerState, TrainerControl
import json


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
        checkpoint_dir (Path): Pathlib Path object pointing to the experiment directory.

    Returns:
        trainer_state (dict): The parsed trainer_state.json
        starting_step (int): The global step at which to resume training
        resume_from_checkpoint (str): Path to checkpoint folder, or None
    """
    resume_from_checkpoint = None
    starting_step = 0
    try:
        # Load trainer state to get current step
        trainer_state_path = os.path.join(checkpoint_dir.path, "checkpoint/trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
                starting_step = trainer_state["global_step"]
                print(f"Will resume from step {starting_step}")
                # Set this to tell the trainer where to load from
                resume_from_checkpoint = os.path.join(checkpoint_dir.path, "checkpoint")

                return trainer_state, starting_step, resume_from_checkpoint
        else:
            print(f"Path does not exists: {trainer_state_path}")

    except Exception as e:
        print(f"Error synchronizing iterator state: {e}")


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

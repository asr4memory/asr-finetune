import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback
import torch
from transformers import TrainingArguments, TrainerState, TrainerControl

# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
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

class StepSyncCallback(TrainerCallback):
    def __init__(self, starting_step):
        self.starting_step = starting_step
        self.has_synced = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.starting_step > 0 and not self.has_synced:
            print(f"Synchronizing step counter to {self.starting_step}")
            # Update the trainer's step counter
            state.global_step = self.starting_step
            self.has_synced = True


def load_checkpoints(checkpoint_dir):
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



# this is a hack - as train_ds from Ray requires the data_collotor, so does Seq2SeqTrainer from HF
# but collating twice does not make sense, therefore we introduce the indentity collator
def data_collator_id(batch):
    return {k: v.to(f"cuda:{local_rank}") if torch.is_tensor(v) else v for k, v in batch.items()}

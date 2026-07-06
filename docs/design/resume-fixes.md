# Resume Fix — Ray Tune trial errors on every SLURM resume

> All file paths below are relative to the source root `src/` (e.g.
> `trainers/utils.py` = `src/trainers/utils.py`). These fixes are already applied
> in the code; this document records *why* they exist so future changes don't
> regress them.

## Symptom

Every trial errors immediately when a Ray Tune job is resumed (re-sbatch'd).
The SLURM log shows:

```
TypeError: cannot unpack non-iterable NoneType object
```

All trials go to ERROR state. No training happens.

## Root cause

`trainers/utils.py` — `load_checkpoints()` (lines ~112–147) used
`checkpoint_dir.path` to get the checkpoint directory path.
That attribute was **removed in Ray 2.7+**. With Ray ≥ 2.7 (e.g. 2.46.0):

```python
checkpoint_dir.path          # → AttributeError
```

The AttributeError was silently caught by the bare `except Exception` block,
which caused the function to **return `None` implicitly** (Python returns None
when a function has no explicit return in the except path).

The caller then tries to unpack the result as a 3-tuple:

```python
trainer_state, starting_step, resume_from_checkpoint = load_checkpoints(checkpoint_dir)
#                                                       ^^^^ returns None → TypeError
```

→ TypeError → trial error → every resume fails.

## Fix

**File:** `trainers/utils.py`, function `load_checkpoints` (~line 112)

Replace the old body with:

```python
def load_checkpoints(checkpoint_dir):
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
    return {}, starting_step, resume_from_checkpoint   # ← safe fallback (was missing)
```

Two changes:
1. `checkpoint_dir.path` → `checkpoint_dir.as_directory()` context manager
2. Added explicit `return {}, starting_step, resume_from_checkpoint` at the end
   so the caller's tuple unpack never sees `None`

## Verification

After applying the fix, re-submit the job. In the SLURM log you should see:

```
Resuming Ray Tune experiment from <storage_path>/<output_tag>
...
(RayTrainWorker ...) Will resume from step 20
```

Trials will pick up from their last checkpoint instead of restarting from step 0.

## Checkpoint structure (for reference)

Ray Tune saves checkpoints via `RayTrainReportCallback` (in
`data_and_collator/hf_to_ray_custom_utils.py`). The structure is:

```
<ray_results>/<output_tag>/<trial_name>/
  checkpoint_000000/          ← Ray Checkpoint directory
    checkpoint/               ← HF trainer checkpoint (CHECKPOINT_NAME = "checkpoint")
      trainer_state.json      ← contains global_step
      adapter_model/
      optimizer.pt
      ...
```

`load_checkpoints` looks for `<ckpt_path>/checkpoint/trainer_state.json`
and reads `global_step` from it.

---

## Optuna rank mismatch on resume

### Symptom

One or more trials error on resume with:

```
RuntimeError: Error(s) in loading state_dict for PeftModel:
    size mismatch for ...: copying a param with shape torch.Size([...]) from checkpoint,
    the shape in current model is torch.Size([...]).
```

The error is thrown from `LoadAdapterFromSubdirCallback.on_train_begin`
(`trainers/utils.py`, line ~49).

### Root cause

When Optuna resumes a trial, it **re-samples the hyperparameters from the search
space** rather than reusing the exact values from the interrupted run.  If
`target_r` (LoRA rank) differs between the original run and the re-sample, the
model is built with the new rank before the checkpoint is inspected. When
`LoadAdapterFromSubdirCallback` then tries to load the saved adapter weights
(which were written with the original rank), the weight shapes don't match →
`RuntimeError`.

This is distinct from the Ray 2.7+ `checkpoint_dir.path` bug above. A trial can
survive the `load_checkpoints()` fix and still crash here if Optuna proposes a
different rank on the resumed run.

### Fix (two-layer defence)

**Layer 1 — root cause fix** (`trainers/trainers.py`):

Read `r` and `lora_alpha` from the checkpoint's `adapter_config.json` **before**
`LoraConfig` is constructed, and override `config["target_r"]` / `config["alpha"]`
if they differ. Insert this block after the `alpha_coupled` expansion (where
`alpha_coupled` is already `del`'d from config) but before `lora_dropout = ...`
and `LoraConfig(...)`:

```python
# If resuming, reconcile target_r with the checkpoint's adapter_config.json.
# Optuna may re-sample a different rank on resume, causing a size mismatch
# when LoadAdapterFromSubdirCallback tries to load the saved adapter weights.
_early_ckpt = _get_tune_checkpoint()
if _early_ckpt:
    import json as _json
    try:
        with _early_ckpt.as_directory() as _ckpt_path:
            _adapter_cfg_path = os.path.join(
                _ckpt_path, "checkpoint", "adapter_model", "adapter_config.json"
            )
            if os.path.exists(_adapter_cfg_path):
                _adapter_cfg = _json.load(open(_adapter_cfg_path))
                _ckpt_r = _adapter_cfg.get("r")
                _ckpt_alpha = _adapter_cfg.get("lora_alpha")
                if _ckpt_r is not None and int(_ckpt_r) != int(config["target_r"]):
                    print(
                        f"[resume] target_r mismatch: Optuna proposed r={config['target_r']} "
                        f"alpha={config['alpha']}, checkpoint has r={_ckpt_r} "
                        f"lora_alpha={_ckpt_alpha}. Using checkpoint values.",
                        flush=True,
                    )
                    config["target_r"] = int(_ckpt_r)
                    if _ckpt_alpha is not None:
                        config["alpha"] = int(_ckpt_alpha)
    except Exception as _e:
        print(f"[resume] Could not read adapter_config.json for rank reconciliation: {_e}", flush=True)
```

Note: `lora_alpha` must be read from `adapter_config.json` directly — it cannot
be recomputed from `alpha_coupled * target_r` because `alpha_coupled` has already
been deleted from config by the time this block runs.

**Layer 2 — safety net** (`trainers/utils.py`, `LoadAdapterFromSubdirCallback.on_train_begin`):

Wrap `set_peft_model_state_dict` in a try/except so that any residual rank
mismatch (e.g. from a checkpoint that has no `adapter_config.json`) skips the
load instead of crashing the trial:

```python
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
```

### Why only 2 of 48 trials were affected

Optuna's re-sampling is probabilistic. For most trials the re-sampled `target_r`
happens to match the original value (or the trial never needed to resume).
Only trials where Optuna drew a *different* rank on the resumed run are affected,
which was 2 out of 48 in the observed case.

### Verification

After applying the fix, re-submit the job. For any trial that resumes, the log
should show:

```
[resume] target_r mismatch: Optuna proposed r=16 alpha=32, checkpoint has r=12
lora_alpha=12. Using checkpoint values.
```

(if a mismatch is detected and corrected), or nothing if ranks already match.
No `size mismatch` RuntimeErrors should appear.

---

## Secondary issue — `accelerate` version mismatch

If you see this error instead of (or before) the TypeError:

```
TypeError: Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'
```

Your `accelerate` package is too old for the installed `transformers`.
`transformers ≥ 4.44` calls `unwrap_model(..., keep_torch_compile=False)` which
requires `accelerate ≥ 1.3`.

Fix:
```bash
pip install "accelerate>=1.14.0"
```

Verify:
```python
from accelerate import Accelerator
import inspect
print(inspect.signature(Accelerator.unwrap_model))
# should include 'keep_torch_compile' in the parameter list
```

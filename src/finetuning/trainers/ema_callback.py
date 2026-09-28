"""Adapter-only exponential moving average for PEFT/LoRA Whisper training.

The callback maintains a CPU-resident shadow of every trainable parameter (i.e. LoRA /
AdaLoRA adapter weights, since the base Whisper is frozen). After ``start_step`` it
updates the shadow each optimizer step:

    shadow <- decay * shadow + (1 - decay) * live

The custom Seq2SeqTrainer wraps each ``evaluate()`` call in ``apply_to`` / ``restore``,
so the metric reported by Ray / Optuna reflects the EMA weights, which is usually a
smoother, slightly better model than the noisy live weights late in training.

A snapshot of the EMA shadow is also saved alongside each Trainer checkpoint at
``<checkpoint>/adapter_ema.pt`` so it can be re-loaded on a multi-slot resume.

Notes
-----
* Memory: for AdaLoRA r=4 on Whisper-large the shadow is on the order of a few MB,
  so keeping it on CPU is essentially free.
* DeepSpeed ZeRO-3: live parameters can be sharded across ranks. We only EMA params
  whose ``data`` is materialised locally; with the recommended "no DeepSpeed in the
  HPO path" change this is always all trainable params. If you re-enable ZeRO-3
  later, gate this callback or use ``deepspeed.zero.GatheredParameters``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
from transformers import TrainerCallback


class AdapterEMACallback(TrainerCallback):
    def __init__(
        self,
        decay: float = 0.99,
        start_step: int = 0,
        save_filename: str = "adapter_ema.pt",
    ):
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.save_filename = save_filename
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._initialised = False

    # -- lifecycle hooks ----------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        self._initialise_shadow(model)
        self._maybe_load_snapshot(args, state)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or not self._initialised:
            return
        if state.global_step < self.start_step:
            return

        d = self.decay
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in self.shadow:
                    # New params (e.g. AdaLoRA reallocations) — initialise lazily.
                    self.shadow[name] = p.detach().float().cpu().clone()
                    continue
                live = p.detach().float().cpu()
                self.shadow[name].mul_(d).add_(live, alpha=1.0 - d)

    def on_save(self, args, state, control, **kwargs):
        if not self._initialised:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(self.shadow, os.path.join(ckpt_dir, self.save_filename))
        except Exception as e:
            print(f"[AdapterEMACallback] WARN: failed to save EMA snapshot: {e}")

    # -- swap helpers, called by the custom evaluator ------------------------

    def apply_to(self, model) -> None:
        """Swap live trainable weights with the EMA shadow. Records a backup so
        ``restore`` can undo the swap after evaluate().
        """
        if not self._initialised or model is None:
            return
        self._backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad or name not in self.shadow:
                    continue
                self._backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].to(p.device).to(p.dtype))

    def restore(self, model) -> None:
        if not self._backup or model is None:
            return
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self._backup:
                    p.data.copy_(self._backup[name])
        self._backup = {}

    # -- internals ----------------------------------------------------------

    def _initialise_shadow(self, model) -> None:
        if self._initialised:
            return
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().float().cpu().clone()
        self._initialised = True
        print(f"[AdapterEMACallback] initialised shadow over {len(self.shadow)} tensors "
              f"(decay={self.decay}, start_step={self.start_step}).")

    def _maybe_load_snapshot(self, args, state) -> None:
        """If the trainer is resuming, look for an EMA snapshot in the resume dir
        and load it so multi-slot training keeps a coherent EMA trajectory.
        """
        resume = getattr(args, "resume_from_checkpoint", None) or os.environ.get("RESUME_CKPT_DIR")
        if not resume:
            return
        snap = os.path.join(str(resume), self.save_filename)
        if not os.path.exists(snap):
            return
        try:
            self.shadow = torch.load(snap, map_location="cpu")
            self._initialised = True
            print(f"[AdapterEMACallback] resumed EMA shadow from {snap} "
                  f"({len(self.shadow)} tensors).")
        except Exception as e:
            print(f"[AdapterEMACallback] WARN: failed to load EMA snapshot {snap}: {e}")

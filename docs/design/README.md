# Design notes

Background on why the code is built the way it is — useful when extending or
debugging it, and as supporting detail for the paper.

- [`peft-method.md`](peft-method.md) — the LoRA+DoRA choice (and why AdaLoRA was
  removed / PiSSA disabled), the `alpha_coupled` scaling, the baseline-corrected
  `eval_wer_diff` objective, and the search/stopping strategy.
- [`resume-fixes.md`](resume-fixes.md) — the Ray 2.7+ checkpoint API change and
  the Optuna rank-resampling mismatch that broke SLURM resume, and the fixes now
  in the code.

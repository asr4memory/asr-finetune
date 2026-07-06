# PEFT method & the baseline-corrected objective

Rationale behind the fine-tuning design in `src/trainers/` and
`src/searchers_and_schedulers/`, so the code's non-obvious choices are documented.

## LoRA + DoRA (not AdaLoRA)

The active PEFT configuration (`trainers/trainers.py`,
`train_whisper_peft_model`) is:

```python
LoraConfig(
    r=target_r,
    lora_alpha=alpha,          # alpha = alpha_coupled * target_r  (see below)
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    use_dora=True,
)
```

- **Attention q/k/v *and* the output projection** are adapted — roughly 2× the
  parameters of adapting q/v alone, which empirically gives more leverage than
  an equivalent rank increase (matches the paper and HF's Whisper PEFT recipes).
- **DoRA** (`use_dora=True`) is a consistent, low-risk win over plain LoRA
  (Liu et al. 2024).
- **AdaLoRA was removed.** An earlier generation used `AdaLoraConfig`, but its
  `RankAllocator.update_and_allocate()` was never called from the HF `Trainer`
  loop, so no rank pruning ever happened — the model simply trained at a fixed
  `init_r`. "AdaLoRA" was a label, not a behaviour. Switching to plain
  fixed-rank LoRA gives honest `r = target_r` semantics and removes a whole class
  of resume hazards (no `rank_pattern` state to restore).
- **PiSSA is present but disabled by default.** `init_lora_weights="pissa"` was
  tried but its SVD ran on CPU and stalled initialization for minutes per trial
  on Whisper-large. It is kept as a documented back-pocket option (re-enable by
  moving the base model to GPU before `get_peft_model`, or using
  `pissa_niter_4`). The base is loaded non-quantized (`load_in_8bit=False`) so
  PiSSA's SVD would be possible if re-enabled.

### `alpha_coupled` (the α multiplier)

Rather than searching `lora_alpha` directly, the search space exposes
`alpha_coupled ∈ {1, 2}` and sets `lora_alpha = alpha_coupled * target_r`. This
keeps the effective LoRA scaling (`alpha / r`) in a stable band regardless of the
sampled rank, so rank and scaling don't fight each other during the search.

## The baseline-corrected WER objective

The HPO objective is **not** raw validation WER but
`eval_wer_diff = WER(fine-tuned, shard i) − WER(pretrained, shard i)`, evaluated
on a validation shard `i` sampled per evaluation:

1. The validation set is split into shards of ~2000 samples.
2. `compute_baseline_wer` evaluates the **pretrained** model on every shard once,
   up front, writing `src/trainers/data/validation_summary_<tag>.csv`.
3. During HPO each trial's WER on the sampled shard is scored *relative to* that
   shard's baseline. Negative `eval_wer_diff` means the adapter beats pretrained.

Why: absolute WER varies a lot between shards, which would inject noise into the
objective. Anchoring each measurement to the pretrained WER on the *same* shard
cancels most of that difficulty variation. Because the pretrained and early
fine-tuned WERs are strongly correlated, the difference has lower variance than
the raw fine-tuned WER — a variance-reduction (control-variate) effect. The
estimator stays unbiased: in expectation over shards it equals the difference on
the full validation set.

## Search & stopping

- **Optuna TPE sampler + ASHA scheduler** on Ray Tune
  (`searchers_and_schedulers/ray_searchers_and_schedulers.py`). ASHA stops
  unpromising trials early and reallocates compute — important because getting a
  WER signal (autoregressive decoding) is expensive. Aggressive early pruning was
  found to be key to finding good hyper-parameters.
- **`DecorrelationStopper`** (`searchers_and_schedulers/stoppers.py`) kills a
  trial when its loss keeps falling while WER rises — the train/eval-metric
  mismatch that signals overfitting.
- **`AdapterEMACallback`** (`trainers/ema_callback.py`) keeps a CPU-resident EMA
  shadow of the adapter weights and swaps it in at evaluation for a smoother,
  less noisy metric.

## Resume robustness

Ray-version and Optuna-resampling resume hazards and their fixes are documented
separately in [`resume-fixes.md`](resume-fixes.md).

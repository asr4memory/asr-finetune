# asr-finetune — parameter-efficient fine-tuning of Whisper for oral-history ASR

Fine-tune OpenAI's **Whisper** for domain-specific, verbatim automatic speech
recognition (ASR) using **LoRA / DoRA** adapters, with hyper-parameter search
driven by **Bayesian optimization (Optuna TPE + ASHA) on Ray Tune** and a
**baseline-corrected WER objective**. This is the code accompanying the
*asr4memory* case study on adapting Whisper for oral history (to appear in the
*Zeitschrift für digitale Geisteswissenschaften*, ZfdG — see [Citation](#citation)).

On the project's held-out oral-history test set, fine-tuning Whisper large-v3
improves word error rate from **18.4 % to 13.3 %** without catastrophic
forgetting on out-of-distribution audio.

> A qualitative baseline-vs-fine-tuned comparison is available
> [here](https://media.oral-history.digital/asr4memory/ev001_comparison_vanilla_may26.mp4).

---

## What this project does

- **PEFT fine-tuning** — LoRA with **DoRA** (`use_dora=True`) on the attention
  query/key/value and output projections of Whisper. (PiSSA initialization
  scaffolding is present but disabled by default.)
- **Hyper-parameter optimization** — Ray Tune with an Optuna **TPE** sampler and
  the **ASHA** early-stopping scheduler; runs many trials in parallel across GPUs
  and nodes, with fault-tolerant resume.
- **Baseline-corrected objective** — the HPO objective is
  `eval_wer_diff = WER(fine-tuned) − WER(pretrained)` measured **per validation
  shard**. The pretrained model is evaluated on each shard *once, up front*
  (`compute_baseline_wer`), and every trial is scored against that anchor. This
  isolates the adapter's effect from shard-to-shard difficulty and reduces the
  objective's variance.
- **Stability aids** — an EMA shadow of the adapter weights at evaluation
  (`AdapterEMACallback`) and a `DecorrelationStopper` that kills trials whose
  loss falls while WER rises (an overfitting signature).
- **Standalone test-set evaluation** — `evaluation/evaluate.py` streams an HDF5
  test set, loads a fine-tuned adapter (or the pretrained baseline), and reports
  WER, with resume and a `--max_eval_batches` smoke-test knob.

---

## Repository layout

```
asr-finetune/
├── README.md  LICENSE  CITATION.cff  requirements.txt  environment.yml
├── src/                         # all Python source (put on PYTHONPATH)
│   ├── projects_paths.py        # env-driven path resolution (MODEL_PATH, DATA_PATH, ...)
│   ├── utils.py                 # shared helpers (normalize, steps_per_epoch, ...)
│   ├── train_hyper.py           # ★ HPO training entry point
│   ├── train_single_peft.py     # ★ single fixed-config run (reproduce a trial)
│   ├── models/                  # Whisper loaders (local dir / HF hub)
│   ├── data_and_collator/       # HDF5 / Parquet loaders, Ray streaming collators
│   ├── prepare_data/            # materialize_dataset.py — HDF5 → Parquet features
│   ├── trainers/                # training loops, PEFT build, EMA, WER metric, baselines/
│   │   └── data/                #   validation_summary_*.csv (per-shard baselines)
│   ├── searchers_and_schedulers/# Optuna/ASHA searchers, DecorrelationStopper, spaces
│   └── evaluation/              # ★ evaluate.py, validate_model.py, compute_baseline_wer.py
├── configs/                     # .config files (configargparse) — train/ and eval/
├── slurm/                       # runnable SLURM job templates (+ examples/ per cluster)
├── scripts/  (under src/)       # helper utilities (download model, migrate Optuna DB, ...)
└── docs/                        # MONITORING.md, HPC.md, design notes
```

The code is run with `src/` on the Python path (`PYTHONPATH=src`); it is **not**
a pip-installable package. Every entry point is invoked as
`PYTHONPATH=src python -m <module>`.

---

## 1. Install

```bash
git clone https://github.com/asr4memory/asr-finetune.git
cd asr-finetune

# Recommended: a fresh Python 3.12 environment (conda or venv)
conda create -n asr-finetune python=3.12 && conda activate asr-finetune
conda install -c conda-forge libsndfile        # native lib for soundfile

# Install a CUDA-matched PyTorch FIRST (pick the index for your CUDA), then the rest:
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`environment.yml` provides a one-shot conda alternative (`conda env create -f
environment.yml`); still reinstall a CUDA-matched torch on a GPU cluster.

## 2. Set paths

All machine-specific locations are read from environment variables by
`src/projects_paths.py`. On a cluster, point them at fast scratch storage:

```bash
export MODEL_PATH=/path/to/models     # holds <model_type>/{model,processor,tokenizer,feature_extractor}
export DATA_PATH=/path/to/data        # holds the Parquet shards and *_test.h5 files
export PYTHONPATH="$PWD/src"          # makes the packages importable
```

Optional: `VALIDATION_SUMMARY_CSV` (per-shard baseline CSV used by the training
objective — see step 5) and `ASR_FINETUNE_ROOT` (override the source root). If
unset, `MODEL_PATH`/`DATA_PATH` default to `<repo>/models_local` and `<repo>/data`.
To extend to your own cluster, copy a `slurm/*.sh` template and set these there.

## 3. Download the base model

```bash
python src/scripts/download_hf_model.py \
    --model_id openai/whisper-large-v3 \
    --output_dir "$MODEL_PATH/whisper-large-v3"

# Cache the WER metric once so later runs work offline:
python -c "import evaluate; evaluate.load('wer')"
```

This writes the four components (`model/`, `processor/`, `feature_extractor/`,
`tokenizer/`) into `$MODEL_PATH/whisper-large-v3/`, the layout the loaders expect.
`--model_id nyrahealth/CrisperWhisper` fetches CrisperWhisper instead.

## 4. Prepare the dataset

Training reads **pre-computed Parquet feature shards**; the standalone evaluator
reads the **HDF5 test set** directly. Materialize each split from its HDF5 corpus
(audio + transcription) one at a time:

```bash
PYTHONPATH=src python -m prepare_data.materialize_dataset \
    --hdf5_path   "$DATA_PATH/eg_dataset_complete_v3_train.h5" \
    --output_path "$DATA_PATH/eg_dataset_complete_v3_sharded" \
    --split       train_parquet \
    --model_type  whisper-large-v3
# repeat with --split val_parquet (and the *_val.h5 corpus)
```

Expected data directory contract: `train_parquet/` and `val_parquet/` shards for
training, plus `<dataset_name>_test.h5` for evaluation, all under `$DATA_PATH`.

## 5. Compute the per-shard baseline WER

The HPO objective is measured *relative to the pretrained model*, so first record
the pretrained WER on every validation shard:

```bash
PYTHONPATH=src python -m evaluation.compute_baseline_wer --model_type whisper-small
# writes src/trainers/data/validation_summary_<tag>.csv
```

Point `VALIDATION_SUMMARY_CSV` at the CSV that matches your model and
`eval_sample_fraction` (e.g. `validation_summary_ws_frac0.05.csv`). A set of
baselines for tiny/small/medium is already committed under `src/trainers/data/`.

## 6. Train

**Hyper-parameter optimization (main entry point):**

```bash
PYTHONPATH=src python -m train_hyper \
    -c configs/train/small_hailmary_phase1.config \
    --storage_path   "$SCRATCH/ray_results" \
    --optuna_db_path "$SCRATCH/optuna/small_wer_diff.db"
```

Key config knobs: `search_schedule_mode=large_small_OPTUNA`, `num_samples`
(trials), `metric_to_optimize=eval_wer_diff`, `eval_sample_fraction`,
`hyperparameters=learning_rate_lora,warmup_ratio,alpha_coupled,target_r_wide,lora_dropout`,
`decorr_stopper`, `ema_decay`. Re-running the same command **resumes** the Ray
Tune experiment and the Optuna study (`resume_training=True`).

**Single fixed-config run** (reproduce the best trial deterministically):

```bash
PYTHONPATH=src python -m train_single_peft -c configs/train/small_hailmary_phase1.config
```

## 7. Evaluate on the test set

Set `model_ckpt_path` (the best-trial checkpoint/adapter directory) and
`path_to_data` in an eval config, then:

```bash
# smoke test on 5 batches first:
PYTHONPATH=src python -m evaluation.evaluate -c configs/eval/small_hailmary.config --max_eval_batches 5
# full run:
PYTHONPATH=src python -m evaluation.evaluate -c configs/eval/small_hailmary.config
```

The evaluator resolves nested adapter directories, guards against DoRA/peft
version mismatches, and writes per-utterance results to `eval_final.json` /
`eval_step_<N>.json` (corpus WER is the mean over utterances). It reports **WER**
only. Use `configs/eval/<size>_baseline.config` (with `peft=False`) for the
pretrained baseline.

## 8. Monitoring & HPC

- **Monitoring** (TensorBoard + the Ray dashboard) — see [`docs/MONITORING.md`](docs/MONITORING.md).
- **HPC / SLURM** — generic job templates live in `slurm/` (fill in the `<...>`
  placeholders); concrete worked examples for two clusters are in
  `slurm/examples/`. See [`docs/HPC.md`](docs/HPC.md).

---

## Reproducing the paper

The reported best model fine-tunes **whisper-large-v3** with LoRA+DoRA on the
attention `q/k/v` and output projections, bf16 mixed precision, batch size 8,
`random_seed = 1337`. Best hyper-parameters found by the search:

| Hyper-parameter | Best value |
|---|---|
| learning rate | 1.99e-4 |
| warm-up ratio | 0.01 |
| alpha multiplier (`alpha_coupled`) | 1 |
| target rank (`target_r`) | 8 |
| LoRA dropout | 0.05 |

The best model came from early training (~1000 steps ≈ 8 h of audio). Training
used 4×A100 GPUs (~48 h); evaluation used a single RTX 2080 Ti (~17 h). See the
per-model baseline CSVs in `src/trainers/data/` and the design notes in
`docs/design/` for the rationale behind DoRA, the α-coupling, and the resume fixes.

## Citation

If you use this code, please cite the accompanying article:

> Christian Horvat, Peter Kompiel, Tobias Kilgus.
> *Adapting Automatic Speech Recognition for Oral History: A Case Study for
> Fine-tuning the Whisper Model with Curated and Domain-specific Training Data.*
> Zeitschrift für digitale Geisteswissenschaften (ZfdG), forthcoming.

A machine-readable entry is in [`CITATION.cff`](CITATION.cff).

## License

See [`LICENSE`](LICENSE).

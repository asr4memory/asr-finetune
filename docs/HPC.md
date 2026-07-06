# Running on an HPC / SLURM cluster

The pipeline is cluster-agnostic: everything machine-specific is passed through
environment variables (read by `src/finetuning/projects_paths.py`) and a handful of SLURM
directives. To port it to your cluster you only edit those, not the Python code.

## The job templates

`slurm/` holds one runnable template per pipeline step:

| Script | Step | GPUs |
|---|---|---|
| `slurm/download_model.sh`   | download a base model            | none (login node) |
| `slurm/prepare_data.sh`     | HDF5 → Parquet feature shards    | none (CPU/Ray)    |
| `slurm/compute_baseline.sh` | per-shard pretrained baseline WER| 1                 |
| `slurm/train_hpo.sh`        | Ray Tune + Optuna HPO            | ≥1 (2 in template)|
| `slurm/train_single.sh`     | single fixed-config PEFT run     | 1                 |
| `slurm/evaluate.sh`         | test-set WER of an adapter       | 1                 |

Each template marks cluster-specific values with `<...>` placeholders:

- `<PARTITION>` — the SLURM partition/queue.
- `<CUDA_MODULE>` — the `module load` name for CUDA (e.g. `CUDA/12.6.0`).
- `<CONDA_BASE>` / `<CONDA_ENV>` — your conda install and environment name.
- `<MODEL_PATH>` / `<DATA_PATH>` / `<SCRATCH>` — fast-storage locations.

Every template sets `export PYTHONPATH="$PWD/src"` and invokes the entry point as
`python -m <module>`, so nothing needs to be `pip install`-ed.

## Worked examples

`slurm/examples/` contains fully filled-in versions for two real clusters, so you
can see concrete module stacks and partitions:

- `curta_train_hpo.sh` — FU Berlin *curta* (`module load CUDA/12.6.0`, `scavenger`
  partition, conda in `$HOME/miniconda3`).
- `nhr_train_hpo.sh` — an NHR@ZIB A100 cluster (`NHRZIBenv` / `sw.a100.el9` /
  `cuda/12.9` module stack, `gpu-a100:shared` partition).

Both read the username from `$USER`, so they carry no personal paths — copy one,
adjust the partition/account/modules, and submit.

## Single-node vs. multi-node

`train_hyper.py` initialises Ray automatically:

- **Single node** — it caps CPU/GPU/object-store resources from the SLURM
  allocation and starts a local Ray runtime. This is what the templates do.
- **Multi-node** — set the `ip_head` / `RAY_ADDRESS` environment variables to an
  already-running Ray head (start the head + workers across nodes with your
  cluster's `srun`/`ray start` pattern); `train_hyper.py` then attaches to that
  cluster instead of starting its own. The templates `unset ip_head RAY_ADDRESS`
  to force the single-node path — remove that line for a multi-node run.

## Resume

Both HPO and evaluation are resumable: re-submitting the same script restores the
Ray Tune experiment and Optuna study (`resume_training=True`) or the evaluation
progress (`resume_evaluation=True`). See `docs/design/resume-fixes.md` for the
Ray/Optuna version-compatibility fixes that make this robust.

from typing import Optional, List, Dict, Any, Callable
import torch
from transformers import Seq2SeqTrainer

try:
    import evaluate as hf_evaluate
except Exception:
    hf_evaluate = None

import random
from .metrics import get_metric_to_optimize
from .utils import normalize as normalize_fn  # noqa


import math
from contextlib import nullcontext
from typing import Optional, List, Dict, Any
import torch
from transformers import Seq2SeqTrainer

from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import csv
import math
import random

import torch
import torch.distributed as dist
from transformers import Seq2SeqTrainer

from .metrics import get_metric_to_optimize
from .utils import normalize as default_normalize_fn

from projects_paths import VALIDATION_SUMMARY_CSV as _DEFAULT_VAL_SUMMARY_CSV


def _shard_sort_key(name: str):
    """Sort shard keys numerically when possible so the deterministic schedule
    cycles val_1, val_2, ... in a stable order regardless of dict insertion."""
    try:
        return (0, int(str(name).split("_")[-1]))
    except (ValueError, IndexError):
        return (1, str(name))


class Seq2SeqTrainerEvalSamplingPeft(Seq2SeqTrainer):
    """
    PEFT-friendly evaluation trainer for Whisper on Ray shards.

    Computes in one manual pass over one randomly selected eval shard:
      - eval_loss
      - eval_wer
      - eval_loss_wer = (1 - wer_weight) * eval_loss + wer_weight * eval_wer
      - eval_wer_diff = eval_wer - baseline_eval_wer_for_that_shard

    Notes
    -----
    - Keeps HF Trainer's evaluate(...) interface.
    - Does NOT pass Ray iterators into super().evaluate().
    - Keeps random shard selection, but synchronizes it across distributed workers.
    - Gathers decoded predictions/references across workers before computing WER.
    """

    def __init__(
        self,
        *args,
        processor: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        eval_sample_fraction: float = 1.0,   # kept for API parity
        prefetch_batches: int = 0,
        eval_collator: Optional[Callable] = None,
        wer_weight: float = 1.0,
        normalize_fn: Optional[Callable[[str], str]] = None,
        language: Optional[str] = "de",
        task: str = "transcribe",
        forced_decoder_ids: Optional[List[List[int]]] = None,
        input_key: str = "input_features",
        max_eval_batches: Optional[int] = None,
        validation_summary_csv: str = _DEFAULT_VAL_SUMMARY_CSV,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.processor = processor
        self.tokenizer = tokenizer
        self.eval_sample_fraction = float(eval_sample_fraction)
        self.prefetch_batches = int(prefetch_batches)
        self.eval_collator = eval_collator
        self.wer_weight = float(wer_weight)
        self.input_key = input_key
        self.max_eval_batches = max_eval_batches
        self.language = language
        self.task = task
        self.text_normalize_fn = normalize_fn or default_normalize_fn

        self._val_wer_lookup = self._load_validation_wer_lookup(validation_summary_csv)

        if forced_decoder_ids is not None:
            self.forced_decoder_ids = forced_decoder_ids
        else:
            fdi = None
            try:
                if self.processor is not None and hasattr(self.processor, "get_decoder_prompt_ids"):
                    fdi = self.processor.get_decoder_prompt_ids(
                        language=self.language,
                        task=self.task,
                    )
            except Exception as e:
                if self.is_world_process_zero():
                    print(f"[WARN] Could not derive forced_decoder_ids: {e}")
            self.forced_decoder_ids = fdi

    @staticmethod
    def _load_validation_wer_lookup(csv_path: str) -> Dict[str, float]:
        path = Path(csv_path)
        lookup: Dict[str, float] = {}

        if not path.exists():
            print(f"[Eval]: WARNING: validation summary CSV not found: {path}")
            return lookup

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shard = row.get("shard")
                wer = row.get("eval_wer")
                if shard is None or wer is None:
                    continue
                try:
                    lookup[str(shard)] = float(wer)
                except ValueError:
                    continue

        return lookup

    @staticmethod
    def _dist_is_ready() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _broadcast_object(self, obj: Any) -> Any:
        if not self._dist_is_ready():
            return obj
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]

    def _all_gather_object(self, obj: Any) -> List[Any]:
        if not self._dist_is_ready():
            return [obj]
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, obj)
        return gathered

    def _all_reduce_sum(self, value: float, device: torch.device) -> float:
        if not self._dist_is_ready():
            return float(value)
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def _select_eval_key(self, eval_key: Optional[Any] = None) -> str:
        """Pick which shard to evaluate this call.

        With ``self.shard_schedule == "deterministic"`` (default) we use a
        step-keyed round-robin so every trial evaluates the same shard at the
        same training step — makes inter-trial ASHA comparisons noise-free in
        the shard dimension. With "random" we fall back to ``random.choice``.
        """
        if not hasattr(self, "eval_shards") or not self.eval_shards:
            raise RuntimeError("self.eval_shards is missing or empty.")

        keys = sorted(self.eval_shards.keys(), key=_shard_sort_key)

        if eval_key is not None:
            chosen = str(eval_key)
        else:
            chosen = None
            if (not self._dist_is_ready()) or dist.get_rank() == 0:
                schedule = getattr(self, "shard_schedule", "deterministic")
                if schedule == "deterministic":
                    eval_steps = max(int(getattr(self.args, "eval_steps", 1)), 1)
                    step = int(getattr(self.state, "global_step", 0))
                    idx = (step // eval_steps) % len(keys)
                    chosen = str(keys[idx])
                else:
                    chosen = str(random.choice(keys))
            chosen = self._broadcast_object(chosen)

        if chosen not in self.eval_shards:
            raise KeyError(f"Selected eval shard '{chosen}' not found. Available: {keys}")

        return chosen

    def _get_decoder(self):
        decoder = self.processor if self.processor is not None else self.tokenizer
        if decoder is None or not hasattr(decoder, "batch_decode"):
            raise RuntimeError("Need a processor/tokenizer with batch_decode for WER decoding.")
        return decoder

    def _get_pad_token_id(self) -> int:
        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token_id", None) is not None:
            return int(self.tokenizer.pad_token_id)

        if (
            self.processor is not None
            and hasattr(self.processor, "tokenizer")
            and getattr(self.processor.tokenizer, "pad_token_id", None) is not None
        ):
            return int(self.processor.tokenizer.pad_token_id)

        return 0

    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        eval_key: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Evaluate on one shard (step-keyed by default) and report both the
        instantaneous ``eval_wer_diff`` AND a per-trial running mean of it.

        The running mean is the HPO objective: it is an unbiased estimator of
        the model's average improvement over whisper-large-v3 across the full
        val set, with variance shrinking as 1/N_evals.
        """
        ema = getattr(self, "ema_callback", None)
        if ema is not None and hasattr(ema, "apply_to"):
            try:
                ema.apply_to(self.model)
            except Exception as e:
                if self.is_world_process_zero():
                    print(f"[Eval]: WARN: EMA apply_to failed: {e}")

        try:
            self._eval_call_count = int(getattr(self, "_eval_call_count", 0)) + 1

            shard_key = self._select_eval_key(eval_key=eval_key)
            if self.is_world_process_zero():
                print(f"[Eval]: shard: {shard_key} (call #{self._eval_call_count})")

            metrics = self._evaluate_single_shard(
                ray_ds=self.eval_shards[shard_key],
                shard_key=shard_key,
                metric_key_prefix=metric_key_prefix,
                max_length=max_length,
                num_beams=num_beams,
            )

            # Running-mean accumulator across this trial's evals. Drives the HPO
            # objective; the instantaneous diff stays in metrics for diagnostics.
            diff_key = f"{metric_key_prefix}_wer_diff"
            wer_key = f"{metric_key_prefix}_wer"
            loss_key = f"{metric_key_prefix}_loss"

            running_keys = (
                ("_diff_sum", "_diff_count", diff_key, f"{diff_key}_running"),
                ("_wer_sum",  "_wer_count",  wer_key,  f"{wer_key}_running"),
                ("_loss_sum", "_loss_count", loss_key, f"{loss_key}_running"),
            )
            for sum_attr, count_attr, src_key, out_key in running_keys:
                if src_key in metrics and metrics[src_key] is not None \
                        and not (isinstance(metrics[src_key], float) and math.isnan(metrics[src_key])):
                    s = float(getattr(self, sum_attr, 0.0)) + float(metrics[src_key])
                    n = int(getattr(self, count_attr, 0)) + 1
                    setattr(self, sum_attr, s)
                    setattr(self, count_attr, n)
                    metrics[out_key] = s / n

            if self.is_world_process_zero():
                short = {k: v for k, v in metrics.items() if "running" in k or k in (diff_key, wer_key, loss_key)}
                print(f"[Eval]: shard {shard_key} -> {short}")

            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
            self.log(metrics)
            return metrics
        finally:
            if ema is not None and hasattr(ema, "restore"):
                try:
                    ema.restore(self.model)
                except Exception as e:
                    if self.is_world_process_zero():
                        print(f"[Eval]: WARN: EMA restore failed: {e}")

    def _evaluate_single_shard(
        self,
        ray_ds: Any,
        shard_key: str,
        metric_key_prefix: str,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        model = self.model
        model.eval()

        decoder = self._get_decoder()
        pad_id = self._get_pad_token_id()
        device = next(model.parameters()).device

        loss_key = f"{metric_key_prefix}_loss"
        wer_key = f"{metric_key_prefix}_wer"
        fused_key = f"{metric_key_prefix}_loss_wer"
        wer_diff_key = f"{metric_key_prefix}_wer_diff"

        local_loss_sum = 0.0
        local_loss_count = 0.0
        local_preds: List[str] = []
        local_refs: List[str] = []

        metric = get_metric_to_optimize("evaluate_wer", tokenizer=self.tokenizer)

        eval_iter = ray_ds.iter_torch_batches(
            prefetch_batches=self.prefetch_batches,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.eval_collator,
        )

        gen_max_length = max_length if max_length is not None else self.args.generation_max_length
        gen_num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        for bi, batch in enumerate(eval_iter):
            if self.max_eval_batches is not None and bi >= self.max_eval_batches:
                break

            if not isinstance(batch, dict):
                continue
            if self.input_key not in batch or "labels" not in batch:
                continue

            batch = self._prepare_inputs(batch)

            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss.detach().float()

                gen_kwargs = {}
                if self.forced_decoder_ids is not None:
                    gen_kwargs["forced_decoder_ids"] = self.forced_decoder_ids
                if gen_max_length is not None:
                    gen_kwargs["max_length"] = gen_max_length
                if gen_num_beams is not None:
                    gen_kwargs["num_beams"] = gen_num_beams

                # Helpful when running with DeepSpeed / distributed generation.
                if getattr(self, "is_deepspeed_enabled", False) or self._dist_is_ready():
                    gen_kwargs["synced_gpus"] = True

                pred_ids = model.generate(
                    inputs=batch[self.input_key],
                    **gen_kwargs,
                )

            local_loss_sum += float(loss.item())
            local_loss_count += 1.0

            labels = batch["labels"].detach().clone()
            labels[labels == -100] = pad_id

            pred_text = decoder.batch_decode(
                pred_ids.detach().cpu(),
                skip_special_tokens=True,
            )
            ref_text = decoder.batch_decode(
                labels.detach().cpu(),
                skip_special_tokens=True,
            )

            pred_text = [self.text_normalize_fn(str(x)) for x in pred_text]
            ref_text = [self.text_normalize_fn(str(x)) for x in ref_text]

            n = min(len(pred_text), len(ref_text))
            if n > 0:
                local_preds.extend(pred_text[:n])
                local_refs.extend(ref_text[:n])

        # ---- reduce loss across workers ----
        total_loss_sum = self._all_reduce_sum(local_loss_sum, device=device)
        total_loss_count = self._all_reduce_sum(local_loss_count, device=device)

        metrics: Dict[str, float] = {}
        if total_loss_count > 0:
            metrics[loss_key] = total_loss_sum / total_loss_count
        else:
            metrics[loss_key] = float("inf")

        # ---- gather decoded text across workers for WER ----
        gathered_preds = self._all_gather_object(local_preds)
        gathered_refs = self._all_gather_object(local_refs)

        preds_text: List[str] = []
        refs_text: List[str] = []

        for part in gathered_preds:
            if part:
                preds_text.extend(part)
        for part in gathered_refs:
            if part:
                refs_text.extend(part)

        wer_val = None
        if preds_text and refs_text:
            wer_0_1 = metric.compute(predictions=preds_text, references=refs_text)
            if wer_0_1 is not None:
                wer_val = 100.0 * float(wer_0_1)
                if not (math.isnan(wer_val) or math.isinf(wer_val)):
                    metrics[wer_key] = wer_val
                else:
                    wer_val = None

        if wer_val is None:
            metrics[fused_key] = float(metrics[loss_key])
        else:
            alpha = 1.0 - float(self.wer_weight)
            beta = float(self.wer_weight)
            metrics[fused_key] = alpha * float(metrics[loss_key]) + beta * float(wer_val)

        # ---- baseline comparison like your other Seq2SeqTrainerEvalSampling ----
        if wer_val is not None:
            try:
                csv_shard_key = f"val_{int(shard_key)}"
            except ValueError:
                csv_shard_key = f"val_{shard_key}"

            csv_eval_wer = self._val_wer_lookup.get(csv_shard_key)
            if csv_eval_wer is not None:
                metrics[wer_diff_key] = float(wer_val) - float(csv_eval_wer)
            else:
                metrics[wer_diff_key] = float("nan")
                if self.is_world_process_zero():
                    print(f"[Eval]: WARNING: shard {csv_shard_key} not found in validation_summary.csv")

        if self.is_world_process_zero():
            print(f"[Eval]: metrics for shard {shard_key}: {metrics}")

        return metrics


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False

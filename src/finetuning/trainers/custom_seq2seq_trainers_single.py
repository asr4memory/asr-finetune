from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import csv
import math
import random

from transformers import Seq2SeqTrainer

from .metrics import get_metric_to_optimize
from .utils import normalize as default_normalize_fn

from finetuning.projects_paths import VALIDATION_SUMMARY_CSV as _DEFAULT_VAL_SUMMARY_CSV

import torch


def _shard_sort_key(name: str):
    try:
        return (0, int(str(name).split("_")[-1]))
    except (ValueError, IndexError):
        return (1, str(name))

class Seq2SeqTrainerEvalSamplingPeftSingle(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        processor: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        eval_sample_fraction: float = 1.0,
        prefetch_batches: int = 0,
        eval_collator: Optional[Callable] = None,
        wer_weight: float = 1.0,
        normalize_fn: Optional[Callable[[str], str]] = None,
        language: Optional[str] = "de",
        task: str = "transcribe",
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

    def _select_eval_key(self, eval_key: Optional[Any] = None) -> str:
        """Pick the eval shard. Deterministic step-keyed schedule by default."""
        if not hasattr(self, "eval_shards") or not self.eval_shards:
            raise RuntimeError("self.eval_shards is missing or empty.")

        keys = sorted(self.eval_shards.keys(), key=_shard_sort_key)
        if eval_key is not None:
            chosen = str(eval_key)
        else:
            schedule = getattr(self, "shard_schedule", "deterministic")
            if schedule == "deterministic":
                eval_steps = max(int(getattr(self.args, "eval_steps", 1)), 1)
                step = int(getattr(self.state, "global_step", 0))
                idx = (step // eval_steps) % len(keys)
                chosen = str(keys[idx])
            else:
                chosen = str(random.choice(keys))

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
        ema = getattr(self, "ema_callback", None)
        if ema is not None and hasattr(ema, "apply_to"):
            try:
                ema.apply_to(self.model)
            except Exception as e:
                print(f"[Eval]: WARN: EMA apply_to failed: {e}")

        try:
            self._eval_call_count = int(getattr(self, "_eval_call_count", 0)) + 1
            shard_key = self._select_eval_key(eval_key=eval_key)
            print(f"[Eval]: shard: {shard_key} (call #{self._eval_call_count})")

            metrics = self._evaluate_single_shard(
                ray_ds=self.eval_shards[shard_key],
                shard_key=shard_key,
                metric_key_prefix=metric_key_prefix,
                max_length=max_length,
                num_beams=num_beams,
            )

            diff_key = f"{metric_key_prefix}_wer_diff"
            wer_key = f"{metric_key_prefix}_wer"
            loss_key = f"{metric_key_prefix}_loss"
            for sum_attr, count_attr, src_key, out_key in (
                ("_diff_sum", "_diff_count", diff_key, f"{diff_key}_running"),
                ("_wer_sum",  "_wer_count",  wer_key,  f"{wer_key}_running"),
                ("_loss_sum", "_loss_count", loss_key, f"{loss_key}_running"),
            ):
                if src_key in metrics and metrics[src_key] is not None \
                        and not (isinstance(metrics[src_key], float) and math.isnan(metrics[src_key])):
                    s = float(getattr(self, sum_attr, 0.0)) + float(metrics[src_key])
                    n = int(getattr(self, count_attr, 0)) + 1
                    setattr(self, sum_attr, s)
                    setattr(self, count_attr, n)
                    metrics[out_key] = s / n

            print(f"[Eval]: shard {shard_key} -> {metrics}")
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

        loss_key = f"{metric_key_prefix}_loss"
        wer_key = f"{metric_key_prefix}_wer"
        fused_key = f"{metric_key_prefix}_loss_wer"
        wer_diff_key = f"{metric_key_prefix}_wer_diff"

        total_loss = 0.0
        total_batches = 0
        preds_text: List[str] = []
        refs_text: List[str] = []

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
                if gen_max_length is not None:
                    gen_kwargs["max_length"] = gen_max_length
                if gen_num_beams is not None:
                    gen_kwargs["num_beams"] = gen_num_beams
                if self.language is not None:
                    gen_kwargs["language"] = self.language
                if self.task is not None:
                    gen_kwargs["task"] = self.task

                pred_ids = model.generate(
                    input_features=batch[self.input_key],
                    **gen_kwargs,
                )

            total_loss += float(loss.item())
            total_batches += 1

            labels = batch["labels"].detach().clone()
            labels[labels == -100] = pad_id

            pred_text = decoder.batch_decode(pred_ids.detach().cpu(), skip_special_tokens=True)
            ref_text = decoder.batch_decode(labels.detach().cpu(), skip_special_tokens=True)

            pred_text = [self.text_normalize_fn(str(x)) for x in pred_text]
            ref_text = [self.text_normalize_fn(str(x)) for x in ref_text]

            n = min(len(pred_text), len(ref_text))
            if n > 0:
                preds_text.extend(pred_text[:n])
                refs_text.extend(ref_text[:n])

        metrics: Dict[str, float] = {}
        metrics[loss_key] = total_loss / max(total_batches, 1)

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

            try:
                csv_shard_key = f"val_{int(shard_key)}"
            except ValueError:
                csv_shard_key = f"val_{shard_key}"

            csv_eval_wer = self._val_wer_lookup.get(csv_shard_key)
            if csv_eval_wer is not None:
                metrics[wer_diff_key] = float(wer_val) - float(csv_eval_wer)
            else:
                metrics[wer_diff_key] = float("nan")
                print(f"[Eval]: WARNING: shard {csv_shard_key} not found in validation_summary.csv")

        print(f"[Eval]: metrics for shard {shard_key}: {metrics}")
        return metrics


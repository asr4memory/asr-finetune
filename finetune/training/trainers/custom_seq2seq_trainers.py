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

class Seq2SeqTrainerEvalSamplingPeft(Seq2SeqTrainer):
    """
    PEFT-friendly evaluation trainer that computes:
      - eval_loss   (via standard HF eval loop)
      - eval_wer    (via an internal generation pass)
      - eval_loss_wer = (1 - wer_weight) * eval_loss + wer_weight * eval_wer

    IMPORTANT: This class does NOT do any subsampling.
    Sampling is delegated entirely to your `prepare_trainer_custom`, which
    already applies `random_sample(self.eval_sample_fraction)` to the Ray Dataset.
    """

    def __init__(
        self,
        *args,
        # kept only for API parity; NOT used here (sampling is done by your wrapper):
        processor: Callable=None,
        tokenizer: Callable=None,
        eval_sample_fraction: float = 1.0,
        prefetch_batches: int = 1,
        eval_collator: Optional[Callable] = None,
        wer_weight: float = 1.0,
        normalize_fn: Optional[Callable[[List[str]], List[str]]] = None,
        language: str | None = "de",
        task: str = "transcribe",
        forced_decoder_ids: Optional[List[List[int]]] = None,
        input_key: str = "input_features",
        max_eval_batches: Optional[int] = None,  # optional hard cap, default None
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Keep attributes for compatibility / wrapper access, but don't resample here
        self.processor = processor
        self.tokenizer = tokenizer
        self.eval_sample_fraction = float(eval_sample_fraction)
        self.prefetch_batches = int(prefetch_batches)
        self.eval_collator = eval_collator

        self.wer_weight = float(wer_weight)
        self.forced_decoder_ids = forced_decoder_ids
        self.input_key = input_key
        self.max_eval_batches = max_eval_batches  # if set, we stop after N batches
        
        # keep for later
        self.language = language
        self.task = task
        # Prefer explicit value, otherwise derive from processor if possible
        if forced_decoder_ids is not None:
            self.forced_decoder_ids = forced_decoder_ids
        else:
            fdi = None
            try:
                if self.processor is not None and hasattr(self.processor, "get_decoder_prompt_ids"):
                    # language can be None; Whisper defaults are okay if user doesn’t set it,
                    # but better to pass it when you know it.
                    fdi = self.processor.get_decoder_prompt_ids(language=self.language, task=self.task)
            except Exception as e:
                print(f"[WARN] Could not derive forced_decoder_ids: {e}")
            self.forced_decoder_ids = fdi
        
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:

        print("I AM HERE!")
        random_key = random.choice(list(self.eval_shards.keys()))
        # Get the corresponding dataset
        random_ds = self.eval_shards[random_key]
        ds = random_ds.iter_torch_batches(
                        prefetch_batches = self.prefetch_batches,
                        batch_size=self.args.per_device_eval_batch_size,
                        collate_fn=self.eval_collator
                        )
        
#        ds2 = random_ds.iter_torch_batches(
#                prefetch_batches = self.prefetch_batches,
#                batch_size=self.args.per_device_eval_batch_size,
#                collate_fn=self.eval_collator
#                )
                        
        print(f"[Eval]: Selected shard: {random_key}")
        print("[Eval]: Eval dataset:", ds)
                        
        loss_key = f"{metric_key_prefix}_loss"
        wer_key = f"{metric_key_prefix}_wer"
        fused_key = f"{metric_key_prefix}_loss_wer"
        
        # Try to compute WER using the SAME eval dataloader (no re-sampling).
        wer_val = None
        try:
            wer_val = self._compute_wer_from_eval_loader(random_ds)
        except Exception as e:
            print(f"Could not calculate WER: {e}")
            # Don’t crash the run if WER fails; just log why it’s missing.
            self.log({"warning": f"Skipping {wer_key}: {type(e).__name__}: {e}"})

        
        # Run the standard HF evaluation on the ALREADY-PREPARED eval loader.
        metrics = super().evaluate(
            eval_dataset=ds,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            max_length=max_length,
            num_beams=num_beams,
        )

        if wer_val is not None:
            metrics[wer_key] = float(wer_val)
            
        # Always emit the fused metric so Tune’s strict check is satisfied.
        # If WER is missing, fall back to pure loss (so the key exists).
        base_loss = float(metrics.get(loss_key, float("inf")))
        if wer_val is None:
            metrics[fused_key] = base_loss
            self.log({"info": f"{wer_key} missing; setting {fused_key} = {loss_key} ({base_loss:.4f})"})
        else:
            alpha = 1.0 - float(self.wer_weight)  # weight loss
            beta = float(self.wer_weight)         # weight WER
            metrics[fused_key] = alpha * base_loss + beta * float(wer_val)

        # Make sure the new keys are propagated to logs/callbacks/Tune.
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self.log(metrics)
        return metrics

    def _compute_wer_from_eval_loader(self, eval_dataset):
        """
        Returns WER as a float (percentage), computed with `evaluate.load("wer")`.
        Raises on fatal issues so the caller's try/except can catch and log.
        """
        import math
        import torch
        try:
            import evaluate
        except Exception as e:
            raise RuntimeError(f"evaluate.load('wer') not available: {e}")
        
        device_type = "cuda" if torch.cuda.is_available() else "cpu"


        # ---- setup decoders / normalizer ----
        processor = getattr(self, "processor", None)
        tokenizer = getattr(self, "tokenizer", None)

        if processor is None and tokenizer is None:
            raise RuntimeError("Neither self.processing_class nor self.tokenizer is set; cannot decode.")

        # Prefer processor for Whisper; fall back to tokenizer
        decoder = processor if processor is not None else tokenizer
        
        print("[Eval WER]: Load wer metric")
#        metric = evaluate.load("wer")
        metric = get_metric_to_optimize("evaluate_wer", tokenizer=tokenizer)

        device = getattr(self.model, "device", torch.device("cpu"))
        preds_text, refs_text = [], []

        total_batches = 0
        used_batches = 0

        # helpful one-time debug print of shapes/keys
        def _debug_batch(batch, tag="first"):
            try:
                keys = list(batch.keys())
                shapes = {k: tuple(getattr(batch[k], "shape", (len(batch[k]),))) for k in keys}
                print(f"[WER DEBUG] {tag} batch keys: {keys}, shapes: {shapes}")
            except Exception as e:
                print(f"[WER DEBUG] {tag} batch debug failed: {e}")
        
        print("[Eval WER]: prepare dataloader")
        eval_ds = eval_dataset.iter_torch_batches(
                    prefetch_batches=self.prefetch_batches,
                    batch_size=self.args.per_device_eval_batch_size,
                    collate_fn=self.eval_collator,
                )
        print("[Eval WER]: Dataloader Prepared")
        for bi, batch in enumerate(eval_ds):
            print("[Eval WER]: Batch ",bi)
            
            total_batches += 1
            if bi == 0:
                try:
                    keys = list(batch.keys())
                    shapes = {k: tuple(getattr(batch[k], "shape", (len(batch[k]),))) for k in keys}
                    print(f"[WER DEBUG] first batch keys: {keys}, shapes: {shapes}")
                except Exception as e:
                    print(f"[WER DEBUG] first batch debug failed: {e}")
                    
            # ---- find inputs & labels robustly ----
            input_feats = None
            for k in ("input_features", "inputs", "input_values"):
                if isinstance(batch, dict) and (k in batch):
                    input_feats = batch[k]
                    break

            if input_feats is None:
                print("[WER DEBUG] skipping batch: no input_features/inputs/input_values key")
                continue

            if "labels" not in batch:
                print("[WER DEBUG] skipping batch: no 'labels' key")
                continue

            labels = batch["labels"]

            # Ensure tensors on correct device / dtype
            if isinstance(input_feats, torch.Tensor):
                input_feats = input_feats.to(device)
            else:
                # list/np → tensor
                input_feats = torch.as_tensor(input_feats, device=device)

            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, device=device)

            # ---- generate predictions ----
            with torch.no_grad():
                gen_kwargs = {}
                fdi = getattr(self, "forced_decoder_ids", None)
                if fdi is not None:
                    gen_kwargs["forced_decoder_ids"] = fdi

                # autocast only helps on CUDA; enabled=False elsewhere is a safe no-op
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
                    pred_ids = self.model.generate(input_feats, **gen_kwargs)

            # ---- prepare labels for decoding (unmask -100) ----
            if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
                pad_id = tokenizer.pad_token_id
            elif processor is not None and hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "pad_token_id"):
                pad_id = processor.tokenizer.pad_token_id
            else:
                pad_id = 0  # safe fallback

            # Make a tensor with same device/dtype for torch.where
            pad_tensor = torch.tensor(pad_id, device=labels.device, dtype=labels.dtype)
            labels = torch.where(labels == -100, pad_tensor, labels)

            # ---- decode → strings ----
            # Use processor.batch_decode if present (preferred for Whisper); otherwise tokenizer.batch_decode
            if hasattr(decoder, "batch_decode"):
                pred_str = decoder.batch_decode(pred_ids, skip_special_tokens=True)
                ref_str = decoder.batch_decode(labels,   skip_special_tokens=True)
            else:
                # Very unlikely, but keep a guard
                raise RuntimeError("Decoder has no 'batch_decode' method.")

            # Convert to plain Python strings & normalize
            pred_str = [normalize_fn(str(s)) for s in pred_str]
            ref_str  = [normalize_fn(str(s)) for s in ref_str]

            # ---- collect ----
            # Sanity: lengths should match
            if len(pred_str) != len(ref_str):
                print(f"[WER DEBUG] batch {bi}: length mismatch preds({len(pred_str)}) vs refs({len(ref_str)}) — will align to min().")
            n = min(len(pred_str), len(ref_str))
            if n == 0:
                print(f"[WER DEBUG] batch {bi}: empty decoded lists, skipping.")
                continue

            preds_text.extend(pred_str[:n])
            refs_text.extend(ref_str[:n])
            used_batches += 1

            # periodic debugging
            if bi == 0 or (bi + 1) % 50 == 0:
                print(f"[WER DEBUG] collected so far: {len(preds_text)} preds / {len(refs_text)} refs (used_batches={used_batches}, total_batches={total_batches})")

        # ---- final checks & compute ----
        if used_batches == 0 or len(preds_text) == 0 or len(refs_text) == 0:
            raise RuntimeError(
                f"No usable evaluation data collected "
                f"(used_batches={used_batches}, total_batches={total_batches}, "
                f"len(preds)={len(preds_text)}, len(refs)={len(refs_text)})."
            )
        
        
        # evaluate expects lists of strings; just in case
        preds_text = [str(x) for x in preds_text]
        refs_text  = [str(x) for x in refs_text]

        wer_0_1 = metric.compute(predictions=preds_text, references=refs_text)
        if wer_0_1 is None or (isinstance(wer_0_1, float) and (math.isnan(wer_0_1) or math.isinf(wer_0_1))):
            raise RuntimeError(f"evaluate('wer') returned invalid value: {wer_0_1}")

        wer_pct = 100.0 * float(wer_0_1)
        print(f"[WER DEBUG] FINAL WER: {wer_pct:.4f}% over {len(preds_text)} samples (used_batches={used_batches}/{total_batches})")
        return wer_pct



class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False

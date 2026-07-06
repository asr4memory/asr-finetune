"""Ray Tune stoppers used by the hail-mary HPO run."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ray.tune import Stopper


class DecorrelationStopper(Stopper):
    """Kill trials whose eval_loss decreases AND eval_wer increases for K
    consecutive evals after a grace period.

    This is the textbook signature of overfitting in next-token CE while
    generation quality silently degrades — the exact pathology observed in
    Whisper PEFT fine-tuning. Killing these trials early frees ASHA budget for
    candidates whose WER actually improves with the loss.

    Parameters
    ----------
    grace_period : int
        Minimum ``step`` (or ``training_iteration``) before any kill decision.
        Pick at least ``warmup_steps`` + a few evals so we don't kill trials
        based on noisy warmup-era metrics.
    consecutive : int
        Number of consecutive decorrelated evals required to trigger a kill.
        Default 2 — a single noisy step shouldn't stop a trial, but two in a
        row is a strong signal.
    loss_key, wer_key : str
        Metric keys to watch. Prefer ``eval_loss_anchor`` / ``eval_wer_anchor``
        for the deterministic-anchor objective.
    time_key : str
        Which result field to read for the current step. Falls back to
        ``training_iteration`` if ``step`` is missing.
    min_delta : float
        Required magnitude of (loss-decrease, wer-increase) to count as
        "really" decorrelated. 0 means any monotone direction counts.
    """

    def __init__(
        self,
        grace_period: int = 2000,
        consecutive: int = 2,
        loss_key: str = "eval_loss_anchor",
        wer_key: str = "eval_wer_anchor",
        time_key: str = "step",
        min_delta: float = 0.0,
    ):
        self.grace_period = int(grace_period)
        self.consecutive = max(1, int(consecutive))
        self.loss_key = loss_key
        self.wer_key = wer_key
        self.time_key = time_key
        self.min_delta = float(min_delta)
        self._history: Dict[str, List[Tuple[float, float]]] = {}

    def _read_step(self, result) -> int:
        step = result.get(self.time_key)
        if step is None:
            step = result.get("training_iteration", 0)
        try:
            return int(step or 0)
        except (TypeError, ValueError):
            return 0

    def __call__(self, trial_id: str, result) -> bool:
        step = self._read_step(result)
        if step < self.grace_period:
            return False

        loss = result.get(self.loss_key)
        wer = result.get(self.wer_key)
        if loss is None or wer is None:
            return False

        hist = self._history.setdefault(trial_id, [])
        hist.append((float(loss), float(wer)))

        # Need consecutive+1 observations to evaluate `consecutive` transitions.
        if len(hist) < self.consecutive + 1:
            return False

        bad = 0
        for i in range(-self.consecutive, 0):
            prev_loss, prev_wer = hist[i - 1]
            cur_loss, cur_wer = hist[i]
            loss_down = (prev_loss - cur_loss) > self.min_delta
            wer_up = (cur_wer - prev_wer) > self.min_delta
            if loss_down and wer_up:
                bad += 1
        if bad >= self.consecutive:
            print(
                f"[DecorrelationStopper] killing trial {trial_id} at step {step}: "
                f"last {self.consecutive} evals showed loss-down/WER-up "
                f"(history: {hist[-(self.consecutive + 1):]})"
            )
            return True
        return False

    def stop_all(self) -> bool:
        return False

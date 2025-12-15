# 26. Sep 2025

Long time ago! Managed to fix a bug for the restarting training (lets see) 

What about PEFT? Code seems to not be tested or not up-to-date, `Seq2SeqTrainerEvalSamplingPeft`
is missing.

Waiting for test run result on a cluster, but ultimate goal is to train a Peft Model but be
also able to monitor the WER (even though it seems better to ignore it...very contraintuitive)


Issue:

+ [ ] I get this weird load_in8bit error ... makes me really want to switch to `uv`
  + for running locally, this should work:
  `python -u train_hyper.py -c configs/largev3_peft_debug.config`
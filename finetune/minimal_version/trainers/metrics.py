import evaluate
from utils import normalize
# Define metric for evaluation

def get_metric_to_optimize(which_metric, tokenizer = None):

    if which_metric == "wer":
        try:
            print("trying to get wer")
            metric = evaluate.load("wer")
        except Exception as e:
            print(f"Evaluate.load failed: {e}"
                  f"Trying to load wer metric locally", flush=True)
            try:
                metric = evaluate.load("wer.py")
            except Exception as e:
                print(f"Save the 'wer.py'. Please ensure it exists in the trainers dir: {e}", flush=True)

        def compute_metrics(pred):
            """Performance Metric calculator, here: Word Error Rate (WER)

            Note: 'Normalizes' the strings before calculating the WER.

            Requires:
                Initialized Tokenizer for decoded the predicitions and labels into human language
                WER metric from the evaluate package
            Args:
                pred (dict): a dictionary with keys "predictions" and "label_ids"
            Returns:
                (dict): A dictionary with key "wer" and the corresponding value
            """
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            # replace -100 with the pad_token_id
            label_ids[label_ids == -100] = tokenizer.pad_token_id

            # we do not want to group tokens when computing the metrics
            pred_str = normalize(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
            label_str = normalize(tokenizer.batch_decode(label_ids, skip_special_tokens=True))
            wer = 100 * metric.compute(predictions=pred_str, references=label_str)

            return {"wer": wer}

    return compute_metrics
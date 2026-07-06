# Monitoring training

Ray Tune writes each trial's checkpoints, TensorBoard event files, and logs under
the `--storage_path` you pass to `train_hyper` (e.g. `$SCRATCH/ray_results/<output_tag>`).

## TensorBoard

On a cluster, forward the TensorBoard port to your machine over SSH, then launch
TensorBoard against the storage path:

```bash
# on your laptop — forward remote port 6007 to local 16006
ssh -L 16006:127.0.0.1:6007 <USER>@<LOGIN_NODE>

# on the cluster
tensorboard --logdir "$SCRATCH/ray_results/<output_tag>" --port 6007 --bind_all
```

Then open <http://localhost:16006>. Useful signals: `eval_wer`, the
baseline-corrected `eval_wer_diff` (negative = better than pretrained), and
`eval_loss` vs. WER divergence (the `DecorrelationStopper` acts on exactly this).

### Step / iteration arithmetic

```
total_gradient_steps = ceil(len(train_set) / per_device_train_batch_size) * num_epochs
checkpoints          = ceil(total_gradient_steps / save_steps)
```

## Ray dashboard

The Ray dashboard shows cluster utilisation, actor status, and per-trial resource
use. Install the dashboard extra (already included via
`ray[...,default]` in `requirements.txt`), forward port `8265`, and open it
locally:

```bash
ssh -L 8265:127.0.0.1:8265 <USER>@<LOGIN_NODE>
# dashboard is served automatically when Ray starts; open http://localhost:8265
```

You can forward both ports at once:

```bash
ssh -L 16006:127.0.0.1:6007 -L 8265:127.0.0.1:8265 <USER>@<LOGIN_NODE>
```

For cluster-wide metrics, Ray also integrates with
[Grafana](https://grafana.com/) and [Prometheus](https://prometheus.io/); see the
[Ray observability docs](https://docs.ray.io/en/latest/ray-observability/getting-started.html).

# Login on Zuse-HPC 
There are different login nodes. The GPUs are in ```blogin2.nhr.zib.de``` and ```blogin1.nhr.zib.de```.
So you need to ssh in ```USERNAME@blogin2.nhr.zib.de``` for example, [see here](https://nhr-zib.atlassian.net/wiki/spaces/PUB/pages/6717441/GPU+A100+partition)

# Installation on Zuse-HPC 

1. Create a folder of your choice in you $HOME HPC directory.
2. Pull the repo into the folder.
3. Install packages in the [requirements.txt](requirements.txt) in your conda environment of choice.
   For example, ```module load anaconda3/2023.09``` and then 
   - `conda create -n "finetune" python=3.12.7`  change "finetune" to your environment name of choice
     (if you change the conda environment name, you need to change it in the `.sh` scripts as well!)
   - Activate the environment `conda activate finetune`
   - navigate into the asr-finetune folder and `pip install -r requirements.txt`
   - Install soundfile from anaconda: `conda install -c conda-forge libsndfile`

# Storage on Zuse-HPC
There are 3 types of storage systems: 

1. HOME ... the usual home directory (i.e. /home/$USER/)
2. WORK ... the /scratch/ directory  
3. PERM ... the permanent /perm/$USER

There are different quotas for each system, have a look [here](https://nhr-zib.atlassian.net/wiki/spaces/PUB/pages/428627/System+Quota).
Important for us: WORK is designed for fast I/O operations so it makes sense to save data there.
Also: with the new code, we need to pre-download the model to finetune.
To avoid unnessary data storage, we can share both data and models in our project directory. 
Our data ist stored in ```/scratch/usr/bemchrvt/data``` until we have a project directory.


In PERM, we can store fine-tuned models. 

# Download the model
Different to the FU-Cluster, we need to pre-download the Whisper Model. For that:

1. Run `download_HF_model`
2. Move the created folder to `/scratch/usr/$USER/whisper-large-v3`. In there, there should be the following folders:
   - `feature_extractor`
   - `model`
   - `models--openai--whisper-large-v3`
   - `processor`
   - `tokenizer`
3. You need to download the `WER` metric also manually. For that, `conda activate evaluate` (or whatever name you chose)
   type `python` into the terminal, and then `import evaluate` and then `metric = evaluate.load("wer")`. This should
   start a download of the metric. After that, stop `python` and proceed to the first job submission!


# First job submission 

0. Activate your environment in your preferred way. E.g. in the `.bash_profile`, within the `.sh` script, or in terminal
   (default: in the `.sh` script)
1. Open [train_whisper_largev3.config](finetune/configs/train_whisper_largev3.config) and adjust the `path_to_data` to 
   the path to the data folder you defined before. Then, adjust `dataset_name` to the name of the dataset you want to 
   train on, default value is `eg_dataset_complete_v2.h5`. 
   Hint: Use a subset of `eg_dataset_complete_v2.h5` for debugging, e.g. `eg_dataset_subset_1000.h5` in
         [/Volumes/asr4mem/asr-daten-sets/finetuning/datasets/ready](/Volumes/asr4mem/asr-daten-sets/finetuning/datasets/ready)
2. Submit a job on a single node via `sbatch finetune_large_debug.sh`
3. Submit a job on multiple nodes via `sbatch finetune_large_debug_multi_node.sh`

Remark: We use the `gpu-a100:test` [partition for testing](https://nhr-zib.atlassian.net/wiki/spaces/PUB/pages/430579/Slurm+partition+GPU+A100)
        for debugging and testing.

*Some further notes*: 
- All relevant files are automatically saved in the scratch folder [/scratch/src/USERNAME/](/scratch/USERNAME/). Results of the 
submitted job with [data_modes.py](finetune%2Fminimal_version%2Fdata_and_collator%2Fdata_modes.py)efined `output_tag` are stored in [/scratch/USERNAME/ray_results/output_tag](/scratch/USERNAME/ray_results/output_tag) and the temporary
files are automaticall stored in [/scratch/USERNAME/tmp](/scratch/USERNAME/tmp) 
- For runs on you local machine for debugging, see the next section.


# Monitor jobs: Tensorboard and Ray Dashboard

1.[train_single_model.py](finetune%2Ftrain_single_model.py) To track the progress of your experiments, log into you HPC account forwarding port 6007 onto you local machine through
`ssh -L 16006:127.0.0.1:6007 USER@blogin2.nhr.zib.de`  (if you used `blogin2` as login node).

Run `tensorboard --logdir /scratch/usr/$USER/ray_results/output_tag/ --bind_all` where output_tag is again the one from
the config file (e.g. `whisper_large_jan`).

2.  To track more general cluster utility, check the ray dashboard. For that, you need to 
    - set up ray dashboard by installing `pip install -U "ray[default]"`
    - start the dashboard config in the [finetune_large_debug_dashboard.sh](finetune_large_debug_dashboard.sh) script.
      have a look [here](https://docs.ray.io/en/latest/ray-observability/getting-started.html) for more details
    - forward port `8265` onto your local machine, so e.g. 
      `ssh -L 16006:127.0.0.1:6007 -L 8265:127.0.0.1:8265 USER@blogin2.nhr.zib.de`.
      Ray dashbaord should be accessible through `localhost:8265`.

3. There are more advanced ways for monitoring using [grafana dashboards](https://grafana.com/) and 
   [Prometheus](https://prometheus.io/docs/introduction/overview/). For installation you can follow [this instruction](https://docs.ray.io/en/latest/cluster/configure-manage-dashboard.html)
   however, we can also do a workshop which might be easier (took me quite some time to make it running).

# Useful formulas

Here are some formulas to understand how many training steps are needed and how many iterations are needed (relevant 
for undertanding the tensorboard loggings)

`total_Gradient_steps = round_up(length_train_set / per_device_train_batch_size) * num_epochs`

`iterations = round_up(total_Gradient_steps / save_steps)`

# Parameter Efficient Finetuning (PEFT)

We follow the tutorial from [here](https://github.com/Vaibhavs10/fast-whisper-finetuning).
In short: PEFT allows to train large models on small resources as not all but only a fraction of the parameters are 
trained. 



# UPDATES 19.10

[//]: # (PATH are now defined in projects_paths and should be adjusted there)

[ ] Add datasets parquet preparation (materialize_ds.sh in curta zedat)
#!/bin/bash
#SBATCH --mail-type=fail,end
#SBATCH --job-name="materialize"
#SBATCH --time=24:00:00
###SBATCH --time=2-00:00:00
#SBATCH --mem=32G  #32

#SBATCH --partition=main
#SBATCH --qos=standard

###SBATCH --partition=gpu-a100:shared

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1  ### ensure that each Ray worker runtime will run on a separate node
#SBATCH --cpus-per-task=8  ### cpus and gpus per node



cd finetune_peft
python -u materialize_dataset.py
#python -u materialize_test.py

#!/bin/bash

#SBATCH --mail-type=fail,end
#SBATCH --job-name="train_tiny"
#SBATCH --time=02:00:00
#SBATCH --mem=64G  #32

#SBATCH --nodes=1
###SBATCH --exclusive
#SBATCH --tasks-per-node=1  ### ensure that each Ray worker runtime will run on a separate node
#SBATCH --cpus-per-task=32  ### cpus and gpus per node
#SBATCH --gres=gpu:4 ##change num_GPU below to same number
num_gpus=4

#SBATCH --partition=gpu
#SBATCH --qos=standard

###SBATCH --nodelist=g007

# automaticall set-up user mail
scontrol update job $SLURM_JOB_ID MailUser=$USER@zedat.fu-berlin.de
echo "num_gpus is $num_gpus"


###module load cuDNN/8.4.1.50-CUDA-11.7.0
module load CUDA/12.0.0
nvidia-smi
nvcc --version

echo "Temp dir $TMPDIR"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --temp-dir "${TMPDIR}" --block &


# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    sleep 5
done


echo "STARTING python command"

cd finetuning
python -u train.py -c configs/tiny_debug.config --storage_path /scratch/$USER/ray_results
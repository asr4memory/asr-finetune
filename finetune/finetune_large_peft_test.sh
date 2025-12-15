#!/bin/bash
#SBATCH --mail-type=fail,end
#SBATCH --job-name="peft_debug"
#SBATCH --time=00:10:00
#SBATCH --mem=512G  #32

#SBATCH --partition=gpu-a100:test
#SBATCH --cpus-per-task=10 ### cpus and gpus per node
#SBATCH --gres=gpu:A100:1 ##change num_GPU below to same number

num_gpus=1
#ray stop

ulimit -u 65536

# automaticall set-up user mail
##scontrol update job $SLURM_JOB_ID MailUser=$USER@zedat.fu-berlin.de

echo "num_gpus is $num_gpus"

# module avail hdf5

###module load cuDNN/8.4.1.50-CUDA-11.7.0
#module avail cuda
#./cuda.bin
#./cuda_cublas.bin
#module avail
#module reset
module purge
module load NHRZIBenv
module load sw.a100.el9
module load slurm
module use /sw/modules/clx.el9/TOOLS


module load cuda/12.9


#module load anaconda3/2020.11
module load anaconda3/2023.09
source activate finetune
#module load anaconda3/2023.09
#source activate finetune
##module load openmpi/gcc.11/4.1.4

nvidia-smi
nvcc --version

export TMPDIR=/scratch/usr/$USER/tmp
mkdir -p $TMPDIR
echo "Temp dir $TMPDIR created"

#export TMPDIR=/scratch/usr/$USER/tmp
#mkdir -p $TMPDIR
#RAY_TMPDIR="/scratch/usr/$USER/tmp/ray_${SLURM_JOB_ID}_$(hostname)"
#mkdir -p $RAY_TMPDIR
#echo "Temp dir $RAY_TMPDIR created"

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
    --num-cpus "${SLURM_CPUS_PER_TASK}" --temp-dir "${TMPDIR}" --num-gpus $num_gpus --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    tmp_node = "/scratch/usr/$USER/tmp/ray_${node_i}"
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --temp-dir "${tmp_node}" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    sleep 5
done

echo "STARTING python command"

export TQDM_DISABLE=1

cd minimal_version
python -u train_hyper.py -c configs/largev3_peft_debug.config --storage_path /scratch/usr/${USER}/ray_results

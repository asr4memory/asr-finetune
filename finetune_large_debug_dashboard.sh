#!/bin/bash
#SBATCH --mail-type=fail,end
#SBATCH --job-name="large_debug"
#SBATCH --time=01:00:00
#SBATCH --mem=64G  #32

#SBATCH --partition=gpu-a100:test
###SBATCH --partition=gpu-a100:shared


##SBATCH --qos=standard
###SBATCH --nodelist=g007

#SBATCH --nodes=1
###SBATCH --exclusive
#SBATCH --tasks-per-node=1  ### ensure that each Ray worker runtime will run on a separate node
#SBATCH --cpus-per-task=4  ### cpus and gpus per node
#SBATCH --gres=gpu:A100:1 ##change num_GPU below to same number

num_gpus=1
ray stop

#### sound file bug

# automaticall set-up user mail
##scontrol update job $SLURM_JOB_ID MailUser=$USER@zedat.fu-berlin.de

echo "num_gpus is $num_gpus"

# module avail hdf5

###module load cuDNN/8.4.1.50-CUDA-11.7.0
module load cuda/11.8

module load anaconda3/2023.09
source activate FU2
##module load openmpi/gcc.11/4.1.4

nvidia-smi
nvcc --version

export TMPDIR=/scratch/usr/$USER/tmp
mkdir -p $TMPDIR
echo "Temp dir $TMPDIR created"

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

# Find a free port for Grafana (let's try 3010 instead of 3000)
GRAFANA_PORT=3000
# More reliable way to get login node hostname
login_node_ip=$(hostname -I | awk '{print $1}')
echo "Login node IP: ${login_node_ip}"

#echo "Setting up reverse port forwarding for Grafana on port $GRAFANA_PORT..."
## This captures your login node hostname
#srun --nodes=1 --ntasks=1 -w "$head_node" \
#    ssh -f -N -T -R $GRAFANA_PORT:${login_node}:$GRAFANA_PORT ${login_node} &
#
## Verify the tunnel is working
#sleep 2
#srun --nodes=1 --ntasks=1 -w "$head_node" \
#    bash -c 'netstat -tulpn | grep 3000 || echo "Tunnel not established"'
#
## Wait for the port forwarding to establish
#sleep 5
#
## Test if the head node can reach Grafana through the tunnel
#srun --nodes=1 --ntasks=1 -w "$head_node" \
#    bash -c 'curl -s http://localhost:3010/api/health || echo "Cannot reach Grafana"'
#
## Wait for the port forwarding to establish
#sleep 5
#
## Set up reverse tunnel for Prometheus
#echo "Setting up reverse port forwarding for Prometheus on port 9090..."
#srun --nodes=1 --ntasks=1 -w "$head_node" \
#    ssh -f -N -T -R 9090:${login_node}:9090 ${login_node} &

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"


#RAY_GRAFANA_HOST=http://localhost:3000
#export RAY_GRAFANA_HOST
#echo "RAY_GRAFANA_HOST is $RAY_GRAFANA_HOST"
#
#RAY_PROMETHEUS_HOST=http://localhost:9090
#export RAY_PROMETHEUS_HOST
#echo "RAY_PROMETHEUS_HOST is $RAY_PROMETHEUS_HOST"




# 👉 START GRAFANA ON HEAD NODE (right before Ray head)
#echo "Starting Grafana on the head node..."
#srun --nodes=1 --ntasks=1 -w "$head_node" \
#    bash -c 'cd /home/bemchrvt/asr-finetune/grafana-v11.5.2; nohup ./bin/grafana server --config /scratch/usr/bemchrvt/tmp/session_latest/metrics/grafana/grafana.ini web > grafana.log 2>&1 &'

echo "Starting Grafana on the head node..."
# First, create a separate script to run on the head node
cat > /tmp/start_grafana.sh << 'EOF'
#!/bin/bash
# Change to Grafana directory
cd /home/bemchrvt/asr-finetune/grafana-v11.5.2

# Create data directory
mkdir -p /scratch/usr/bemchrvt/tmp/grafana

# Start Grafana with basic configuration
nohup ./bin/grafana server \
  --config="/scratch/usr/bemchrvt/tmp/session_latest/metrics/grafana/grafana.ini" \
  --homepath="/home/bemchrvt/asr-finetune/grafana-v11.5.2" \
  > /scratch/usr/bemchrvt/tmp/grafana.log 2>&1 &

# Give it a moment to start
sleep 5

# Check if it's running
echo "Checking if Grafana started:"
ps aux | grep grafana | grep -v grep
echo "Testing Grafana API access:"
curl -s http://localhost:3000/api/health || echo "Grafana not responding"
EOF

# Make the script executable
chmod +x /tmp/start_grafana.sh

# Run the script on the head node
echo "Starting Grafana on the head node..."
srun --nodes=1 --ntasks=1 -w "$head_node" /tmp/start_grafana.sh



sleep 5  # give Grafana a moment to start

# Set the iframe host for browser access
RAY_GRAFANA_IFRAME_HOST=http://localhost:${GRAFANA_PORT}
export RAY_GRAFANA_IFRAME_HOST
echo "RAY_GRAFANA_IFRAME_HOST is $RAY_GRAFANA_IFRAME_HOST"

# This tells Ray on the head node to look for Grafana on localhost
export RAY_GRAFANA_HOST=http://localhost:3000
export RAY_GRAFANA_HOST

# This tells Ray where to find Prometheus (it's on the head node)
RAY_PROMETHEUS_HOST=http://localhost:9090
export RAY_PROMETHEUS_HOST

echo "RAY_GRAFANA_HOST=$RAY_GRAFANA_HOST"
echo "RAY_PROMETHEUS_HOST=$RAY_PROMETHEUS_HOST"
echo "RAY_GRAFANA_IFRAME_HOST=$RAY_GRAFANA_IFRAME_HOST"


# This tells your browser where to find Grafana (after tunneling)
#RAY_GRAFANA_IFRAME_HOST=http://localhost:${GRAFANA_PORT}
#export RAY_GRAFANA_IFRAME_HOST
#
## Now set the environment variables to point to the forwarded port
#RAY_GRAFANA_HOST=http://${head_node_ip}:${GRAFANA_PORT}
#export RAY_GRAFANA_HOST
#echo "RAY_GRAFANA_HOST is $RAY_GRAFANA_HOST"
#
#RAY_PROMETHEUS_HOST=http://${head_node_ip}:9090
#export RAY_PROMETHEUS_HOST
#echo "RAY_PROMETHEUS_HOST is $RAY_PROMETHEUS_HOST"


# Update the SSH tunnel instructions
#echo "==================================================="
#echo "To access the Ray dashboard with Grafana visualizations, run:"
#echo "ssh -L 8265:${head_node_ip}:8265 -L ${GRAFANA_PORT}:${head_node_ip}:${GRAFANA_PORT} -L 9090:${head_node_ip}:9090 $USER@CLUSTER_LOGIN_NODE"
#echo "Then open http://localhost:8265 in your browser"
#echo "==================================================="

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    --export=ALL,RAY_GRAFANA_HOST,RAY_PROMETHEUS_HOST,RAY_GRAFANA_IFRAME_HOST \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --metrics-export-port=9090 \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --temp-dir "${TMPDIR}" --block &


##--temp-dir "${TMPDIR}"

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

##export TQDM_DISABLE=1

cd finetune
python -u train.py -c configs/train_whisper_largev3.config --storage_path /scratch/usr/$USER/ray_results

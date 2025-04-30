#!/bin/bash
#SBATCH --job-name=EasyR1-qwen2p5VL-7b-DocAgent
#SBATCH --nodes=2
#SBATCH --mem=450G
#SBATCH --mail-user=tianyu.yang@uni-goettingen.de
#SBATCH --mail-type=all
#SBATCH --cpus-per-task=64
#SBATCH -p kisski
#SBATCH --gpus-per-node=4
#SBATCH -t 48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#############module load cuda/12.2.1
############SBATCH --constraint=80gb
################SBATCH --mem=500G
set -x

WANDB_API_KEY=a3b3f7b7962a8b549c4635ee3a03944d554f1a10
MODEL_PATH=/mnt/vast-kisski/projects/kisski-sub-doc-understanding/EasyR1/models/qwen_2_5_vl_32b  # replace it with your local file path

if [ "$WANDB_API_KEY" != "None" ]; then
    wandb login --relogin $WANDB_API_KEY
fi

# make output directory
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

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


echo "StartingHEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" /bin/bash -c \
       "source /user/yang28/u14705/.bashrc && source /mnt/vast-kisski/projects/kisski-sub-doc-understanding/miniconda3/bin/activate EasyR1 \
        && ray start --head --node-ip-address="$head_node_ip" --port=$port \
         --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --include-dashboard true --dashboard-host 0.0.0.0 --dashboard-port 8265 --block" &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
#export worker_num = 1

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" /bin/bash -c \
      "source /user/yang28/u14705/.bashrc && source /mnt/vast-kisski/projects/kisski-sub-doc-understanding/miniconda3/bin/activate EasyR1  \
      && ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block" &
    sleep 5
done

srun --overlap --nodes=1 --ntasks=1 -w "$head_node"  /bin/bash -c \
"source /user/yang28/u14705/.bashrc && source /mnt/vast-kisski/projects/kisski-sub-doc-understanding/miniconda3/bin/activate EasyR1  \
  && python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=8 \
    trainer.experiment_name=qwen2_5_vl_32b_geo_grpo \
    trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
    trainer.nnodes=${SLURM_NNODES} "

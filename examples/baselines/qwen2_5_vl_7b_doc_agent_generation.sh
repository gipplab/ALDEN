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
#export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=checkpoints/EasyR1/qwen2_5_vl_7b_doc_agent_turn-level-ppo_bi-level-gae-mask-without-temp_new_reward/global_step_105/actor/huggingface  # replace it with your local file path
WANDB_API_KEY=a3b3f7b7962a8b549c4635ee3a03944d554f1a10
ROLLOUT_NAME=vllm_agent
SEARCH_TOP_N=1
SEARCH_URL=http://10.241.148.48:42354
LIMIT_IMAGES=15
MAX_RESPONSE_LENGTH=19000
MAX_PROMPT_LENGTH=1024
ROLLOUT_MAX_NUM_BATCHED_TOKENS=22000
TENSOR_PARALLEL_SIZE=2
PROMPT_KEY=question
ROLLOUT_BATCH_SIZE=128
ROLLOUT_N=1
VAL_BATCH_SIZE=-1
TEMPERATURE=0.2
MAX_PIXELS=2508800
MIN_PIXELS=261070
MAX_TURN_NUM=5
TEST_DATA_PATH=/mnt/vast-kisski/projects/kisski-sub-doc-understanding/EasyR1/dataset/test/feta.parquet

CONFIG_PATH=/mnt/vast-kisski/projects/kisski-sub-doc-understanding/EasyR1/examples/generation_config.yaml
SAVE_PATH=/mnt/vast-kisski/projects/kisski-sub-doc-understanding/EasyR1/generation_results/qwen2_5_vl_7b_doc_agent_turn-level-ppo_bi-level-gae-mask-without-temp_new_reward_colqwen

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
  && python -m verl.trainer.main_generation \
    config=${CONFIG_PATH} \
    data.test_files=${TEST_DATA_PATH} \
    data.max_pixels=${MAX_PIXELS} \
    data.min_pixels=${MIN_PIXELS} \
    data.prompt_key=${PROMPT_KEY} \
    data.format_prompt=./examples/format_prompt/doc_agent.py \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.rollout.name=${ROLLOUT_NAME} \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.max_turn_num=${MAX_TURN_NUM} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
    worker.rollout.top_n=${SEARCH_TOP_N} \
    worker.rollout.search_url=${SEARCH_URL} \
    worker.rollout.limit_images=${LIMIT_IMAGES} \
    worker.reward.score_function=./examples/score_function/doc_agent.py:compute_score \
    trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.save_checkpoint_path=${SAVE_PATH}"
#    trainer.load_checkpoint_path=/mnt/vast-kisski/projects/kisski-sub-doc-understanding/EasyR1/checkpoints/qwen2_5_vl_7b_doc_agent/global_step_160"
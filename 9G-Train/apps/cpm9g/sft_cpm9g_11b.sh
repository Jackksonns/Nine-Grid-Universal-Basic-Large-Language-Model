#! /bin/bash
#SBATCH --partition=gpu3-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

export MASTER_ADDR=g3002
export MASTER_PORT=12345

CPM_PATH="/data/groups/QY_LLM_Core/projects/202311-release/Models/11B-Chat/9G-Train"
CONFIG_NAME="${CPM_PATH}/apps/cpm9g/config/11b"
EXP_PATH=./exp
mkdir -p $EXP_PATH
MODEL_NAME="cpm9g-11b-sft"

OPTS=""
OPTS+=" --model-config ${CONFIG_NAME}/config.json"
OPTS+=" --vocab ${CONFIG_NAME}/vocab.txt"

OPTS+=" --train-iters 10000"
OPTS+=" --inspect-iters 200"
OPTS+=" --warmup-iters 500"

OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --max-loss-scale 33554432"
OPTS+=" --min-loss-scale 1"
OPTS+=" --loss-scale-steps 32"

OPTS+=" --offload"
OPTS+=" --batch-size 4"
OPTS+=" --max-length 4096"
OPTS+=" --lr 2e-5"
OPTS+=" --start-step 0"
OPTS+=" --epoch 8"
OPTS+=" --load /data/groups/QY_LLM_Core/models/20231010/11b-base/11b.pt"
OPTS+=" --dataset /data/groups/QY_LLM_Core/datasets/sft/20231025/merge_qy_sft_bin"
# TODO 这些 /data 在启元机器上需要改成 /home 下的路径
OPTS+=" --save ${EXP_PATH}/checkpoints"
OPTS+=" --save-name ${MODEL_NAME}"
# OPTS+=" --tensorboard /data/logs/tensorboard/${MODEL_NAME}/${CUR_DATE}/"
# OPTS+=" --flash triton"
# OPTS+=" --flash cuda"
# OPTS+=" --load-grad"

OPTS+=" $@"


CMD="torchrun --nnodes=4 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/cpm9g/sft_cpm9g.py ${OPTS}"

echo "${CMD}"
$CMD

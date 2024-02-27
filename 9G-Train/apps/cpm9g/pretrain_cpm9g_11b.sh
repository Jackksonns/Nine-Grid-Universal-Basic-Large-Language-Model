#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512GB

# use 8 GPU for example, pretrain may need 32 GPU
export MASTER_ADDR=`hostname`
export MASTER_PORT=12345

mkdir -p /home/${USERNAME}/logs/debug
mkdir -p /home/${USERNAME}/logs/tensorboard/cpm9g/

cd apps/cpm9g
CONFIG_NAME="config/11b"
# --------------- 运行参数 ---------------
OPTS=""
OPTS+=" --model-config ${CONFIG_NAME}/config.json"
OPTS+=" --vocab ${CONFIG_NAME}/vocab.txt"
OPTS+=" --batch-size 4"
OPTS+=" --train-iters 400000"
OPTS+=" --save-iters 250"
OPTS+=" --save-name cpm9g_checkpoint"
OPTS+=" --max-length 4096"
OPTS+=" --lr 1.5e-5"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --loss-scale-steps 32"
OPTS+=" --offload"
OPTS+=" --flash cuda"
# OPTS+=" --load-grad"

# --------------- 写文件路径 ---------------
## checkpoint
OPTS+=" --save /home/${USERNAME}/checkpoints/cpm9g/"
OPTS+=" --save-model /home/${USERNAME}/models/cpm9g/"

## logs，/local/logs 等价于 /data/logs（软链）
OPTS+=" --log-dir /home/${USERNAME}/logs/train/"
OPTS+=" --tensorboard /home/${USERNAME}/tensorboard/cpm9g/"`date +"%Y%m%d%H%M%S"`

# --------------- 读文件路径 ---------------
OPTS+=" --dataset config/datasets.json"
OPTS+=" --load ${CHECKPOINT}"
OPTS+=" --start-step 1"

# --------------- 透传参数 ---------------
OPTS+=" $@"

# --------------- 最终指令 ---------------

CMD="torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain_cpm9g.py ${OPTS}"
echo "${CMD}"

$CMD

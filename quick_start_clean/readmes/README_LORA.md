# Lora 训练

## lora 训练脚本

``` shell 
#! /bin/bash

#!/bin/bash
#SBATCH --partition=gpu3
#SBATCH --nodes=1
#SBATCH --nodelist=g3005
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=512GB

export MASTER_ADDR="localhost"
export MASTER_PORT=12347

CPM_PATH="/home/wangxvjia/CPM-onlyllama"
EXP_PATH=/home/wangxvjia/9g_models/cpm_fin_new_1e4
MODEL_NAME="9g-finance-sft"

OPTS=""
OPTS+=" --vocab /home/wangxvjia/9g_models/vocab.txt"
OPTS+=" --model-config /home/wangxvjia/9g_models/config.json"

OPTS+=" --train-iters 695"
OPTS+=" --inspect-iters 2000"
OPTS+=" --warmup-iters 20"

OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --max-loss-scale 33554432"
OPTS+=" --min-loss-scale 1"
OPTS+=" --loss-scale-steps 32"

OPTS+=" --offload"
OPTS+=" --batch-size 2"
OPTS+=" --max-length 4096"
OPTS+=" --lr 3e-4"
OPTS+=" --start-step 0"
OPTS+=" --epoch 4"
OPTS+=" --load /data/groups/QY_LLM_Other/anrongqiao/UltraEval/caterpillar_8b_checkpoint-22000-float16.pt"
OPTS+=" --dataset /home/wangxvjia/molora/data_process/fin_9g/train_data_30000"
# TODO 这些 /data 在启元机器上需要改成 /home 下的路径
OPTS+=" --save ${EXP_PATH}/checkpoints"
OPTS+=" --save-name ${MODEL_NAME}"

OPTS+=" --delta-tuning"
OPTS+=" --delta-type lora"
OPTS+=" --lora-r 64" # 常用的lora 参数
OPTS+=" --lora-dropout 0.05"
OPTS+=" --lora-alpha 64" # 常用的lora alpha 参数
OPTS+=" --lora-layer project_q project_v project_k w_0 w_1 w_out"
OPTS+=" --save-origin-model"

OPTS+=" $@"


CMD="torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/cpm9g/sft_cpm9g_delta.py ${OPTS}"

echo "${CMD}"
$CMD
```

## 合并模型
- 将lora delta model参数和original model merge在一起 作为新的模型，但是模型的参数数量并没有增多
```python
python merge_lora_delta.py  --base_path cpm9g-8b-sft.pt --delta_path cpm9g-lora.pt --merge_path  cpm9g-8b-sft_with_lora.pt
```

# lora 推理

合并后的lora模型可以直接采用基础模型推理代码
见[quick start](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md)



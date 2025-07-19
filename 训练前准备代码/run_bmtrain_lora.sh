#!/bin/bash

#export OMP_NUM_THREADS=16

# export CUDA_VISIBLE_DEVICES=3,4

declare -A args  # Declare an associative array to store arguments and values

args["model_unique"]="4b_lora"
args["pretrained_model"]="./9g_models/4b.pt"
args["tokenizer_path"]="./ckpt"
args["dataset_config"]="data_test"
args["config"]="4b_lora"
args["n_gpus"]=2

args["lora_layer"]= "[\"project_q\",\"project_k\",\"project_v\",\"w_0\",\"w_1\",\"w_out\"]"
args["lora_r"]=16
args["lora_alpha"]=32
args["lora_dropout"]=0.0

args["resume_ckpt"]=""
args["flash"]="cuda"
args["batch_size"]="1"
args["max_length"]="4096"
args["save_iters"]="500"
args["train_iters"]="10"
args["local"]="False"
args["dataloader"]="indexed"
args["save"]="True"
args["dataloader_num_threads"]=1
args["dataloader_prefetch"]=1
args["dataloader_prefetch_factor"]=1
args["dataloader_num_workers"]=1
args["lr"]="1e-5"
args["warmup_iters"]="20"
args["drop_iters"]="0.1"
args["load_grad"]="False"
args["grad_ckpt_num"]="160"
args["exp_group"]=""
args["ignore_cuda_oom"]="1"
args["tensorboard_all_tasks"]="0"
args["stop_when_end"]="1"
args["only_run_dataloader"]="0"
args["eps"]="1e-6"
args["inspect_iters"]="100"
args["strict_state_dict"]="1"
args["only_load_model"]="1"
args["lr_scheduler"]="cosine"
args["resume_no_optimze"]="0"
args["tp_size"]="1"
args["parallel_load_datastate"]="8"
args["async_save"]="False"
args["load_dataloader_ckpt"]="0"
args["drop_begin"]="-1"
args["drop_rate"]="0.5"
args["use_checkpoint"]="0"


# Loop through the arguments
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    # Check if the argument starts with "--"
    if [[ "$arg" == --* ]]; then
        arg_name="${arg:2}"  # Remove leading "--"
        valueid=$((i+1))
        # Get the value of the argument if it exists
        if ((i+1 <= $#)); then
            args["$arg_name"]="${!valueid}"
            i=$((i+1))  # Skip the next argument (its value)
        else
            args["$arg_name"]=""  # Set empty value if no value provided
        fi
    fi
done

# 使用 Python 读取 JSON 文件并更新 Bash 字典
while read -r key value; do
  args["$key"]="$value"
done < <(python -c 'import json, sys; obj = json.load(open("train_configs/'${args['config']}'.json"))["pretrain"]; print("\n".join(["{} {}".format(k, v) for k, v in obj.items()]))')

while read -r key value; do
  # echo $key $value "finetune"
  args["$key"]="$value"
done < <(python -c 'import json, sys; obj = json.load(open("train_configs/'${args['config']}'.json"))["finetune"]; print("\n".join(["{} {}".format(k, v) for k, v in obj.items()]))')


# 用cmd arg 再更新一次
# Loop through the arguments
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    # Check if the argument starts with "--"
    if [[ "$arg" == --* ]]; then
        arg_name="${arg:2}"  # Remove leading "--"
        valueid=$((i+1))

        # Get the value of the argument if it exists
        if ((i+1 <= $#)); then
            args["$arg_name"]="${!valueid}"
            i=$((i+1))  # Skip the next argument (its value)
        else
            args["$arg_name"]=""  # Set empty value if no value provided
        fi
    fi
done

# Print the values of the arguments
echo "----------- CMD args ----------"
for key in "${!args[@]}"; do
    echo "$key: ${args[$key]}"
done
echo "--------- END CMD args --------"


if [[ ${args["flash"]} == "triton" ]]; then
    sudo cp /usr/local/cuda-11.6/compat/libcuda.so.510.108.03 /usr/lib/x86_64-linux-gnu/libcuda.so.510.108.03
    sudo ln /usr/lib/x86_64-linux-gnu/libcuda.so.510.108.03 /usr/lib/x86_64-linux-gnu/libcuda.so
    echo "triton flash"
fi

GPUS_PER_NODE=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
# GPUS_PER_NODE=1
echo "Using ${GPUS_PER_NODE} GPU each machine"


if [[ ${args["model_unique"]} == "" ]]; then
    MODEL_UNIQUE=${JEEVES_JOB_ID}  # 写入的位置，没传的话自动构造
    # JOBID+CreateTime, 本次run的唯一标识符。在白箱里可以通过/projects/${PROJECTID}-${PROJECTNAME}/checkpoints/${MODEL_UNIQUE} 拿到 checkpoint
#                                               通过/projects/${PROJECTID}-${PROJECTNAME}/tensorboard/${MODEL_UNIQUE} 拿到 tensorboard
else
    MODEL_UNIQUE=${args["model_unique"]} # 给了写入的位置
fi
echo "model_unique: "$MODEL_UNIQUE

# --------------- 运行参数 ---------------

OPTS+=" --model-config model_configs/"${args['config']}".json" # [CHANGE]
OPTS+=" --batch-size ${args["batch_size"]}"
OPTS+=" --train-iters ${args["train_iters"]}"
OPTS+=" --save-iters ${args["save_iters"]}"
OPTS+=" --save-name fm9g_live_checkpoint"
OPTS+=" --max-length ${args["max_length"]}"
OPTS+=" --lr ${args["lr"]}"
OPTS+=" --inspect-iters ${args["inspect_iters"]}"
OPTS+=" --warmup-iters ${args["warmup_iters"]}"
OPTS+=" --drop-iters ${args["drop_iters"]}"
OPTS+=" --lr_scheduler ${args["lr_scheduler"]}"
OPTS+=" --offload"
#OPTS+=" --vocab ./tokenizer/vocab.txt"
OPTS+=" --flash ${args["flash"]}"
OPTS+=" --tensorboard_all_tasks ${args["tensorboard_all_tasks"]}"
OPTS+=" --ignore_cuda_oom ${args["ignore_cuda_oom"]}"
OPTS+=" --stop_when_end ${args["stop_when_end"]}"
OPTS+=" --only_run_dataloader ${args["only_run_dataloader"]}"
OPTS+=" --eps ${args["eps"]}"
OPTS+=" --strict_state_dict ${args["strict_state_dict"]}"
OPTS+=" --only_load_model ${args["only_load_model"]}"
OPTS+=" --resume_no_optimze ${args["resume_no_optimze"]}"
OPTS+=" --tokenizer_path ${args["tokenizer_path"]}" 
OPTS+=" --weight-decay 0.1"
OPTS+=" --tp-size ${args["tp_size"]}"
OPTS+=" --parallel_load_datastate ${args["parallel_load_datastate"]}"
OPTS+=" --load_dataloader_ckpt ${args["load_dataloader_ckpt"]}"
OPTS+=" --drop_begin ${args["drop_begin"]}"
OPTS+=" --drop_rate ${args["drop_rate"]}"
OPTS+=" --use_checkpoint ${args["use_checkpoint"]}"
OPTS+=" --delta-tuning"

OPTS+=" --lora-layer ${args["lora_layer"]}"
OPTS+=" --lora-r ${args["lora_r"]}"
OPTS+=" --lora-alpha ${args["lora_alpha"]}"
OPTS+=" --lora-dropout ${args["lora_dropout"]}"


if [[ ${args["load_grad"]} == "True" ]]; then
    OPTS+=" --load-grad"
    OPTS+=" --grad-ckpt-num ${args["grad_ckpt_num"]}"
fi


if [[ ${args["async_save"]} == "True" ]]; then
    OPTS+=" --async_save"
fi


if [[ ${args["dataloader"]} == "indexed" ]]; then
    OPTS+=" --dataloader_num_threads ${args["dataloader_num_threads"]}"
    OPTS+=" --dataloader_prefetch ${args["dataloader_prefetch"]}"
    OPTS+=" --dataloader_num_workers ${args["dataloader_num_workers"]}"
    OPTS+=" --dataloader_prefetch_factor ${args["dataloader_prefetch_factor"]}"
fi


# --------------- 写文件路径 ---------------
## checkpoint
if [[ ${args["save"]} == "True" ]]; then
  
    OPTS+=" --save ./data/checkpoints/${MODEL_UNIQUE}/"
    OPTS+=" --save-model ./not_exist/${MODEL_UNIQUE}/"
else
    echo "won't save model"
fi


## logs，/local/logs 等价于 ./datalogs（软链）
mkdir -p ./data/checkpoints/logs/${MODEL_UNIQUE}
OPTS+=" --log-dir ./data/checkpoints/logs/${MODEL_UNIQUE}"
OPTS+=" --tensorboard ./data/tensorboard/${args["exp_group"]}${MODEL_UNIQUE}/"



if [[ ${args["local"]} == "True" ]]; then
    current_dir=$(pwd)
    OPTS+=" --dataset ${current_dir}/dataset_configs/${args["dataset_config"]}.json"
else
    current_dir=$(pwd)
    OPTS+=" --dataset ${current_dir}/dataset_configs/${args["dataset_config"]}.json"
    echo "Platform config:"${PLATFORM_CONFIG_PATH}
fi

if [ "${args["resume_ckpt"]}" != "" ]; then
  OPTS+=" --load ./data/checkpoints/${MODEL_UNIQUE}/${args["resume_ckpt"]}"
elif [ "${args["pretrained_model"]}" != "" ]; then
  OPTS+=" --load ${args["pretrained_model"]}"
else
  echo "No checkpoint to load"
fi


filename="finetune_dragonfly"

if [[ ${args["local"]} == "True" ]]; then
    PRETRAIN_ENTRY="$filename.py"
else
    PRETRAIN_ENTRY="$filename.py"
fi


GPUS_PER_NODE=${args["n_gpus"]}
NNODES=1
RANK=0
MASTER_ENDPOINT=localhost
MASTER_PORT=12345

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK}  --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ENDPOINT}:${MASTER_PORT} ${PRETRAIN_ENTRY} ${OPTS}"

echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

$CMD

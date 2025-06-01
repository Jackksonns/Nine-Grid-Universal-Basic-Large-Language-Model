#! /usr/bin/env bash

export TOKENIZERS_PARALLELISM=true

train_data_config_path="./dataset_configs/data_test.json"
eval_data_path="../../data_test/indexed_data/indexed_test_data"
batch_size=1
model_max_length=4096
gradient_acc_steps=1
lr=1e-4
identity=testqlora1
model_name_or_path="/public/kqa/9G_4_7_70/70b"

use_lora=true
qlora=true
lora_modules="[\"q_proj\", \"v_proj\", \"k_proj\"]"
lora_r=64
lora_alpha=32
lora_dropout=0.1



exp_dir="./data/checkpoints/huggingface_70b"

set -ue

torchrun \
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6004 \
    -m finetune_hgface \
    --model_name_or_path $model_name_or_path \
    --train_data_config_path $train_data_config_path \
    --eval_data_path $eval_data_path \
    --use_lora $use_lora \
    --qlora $qlora \
    --lora_modules "$lora_modules" \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $lora_dropout \
    --bf16 true \
    --output_dir "$exp_dir/$identity" \
    --num_train_epochs 2 \
    --per_device_train_batch_size $batch_size \
    --model_max_length $model_max_length \
    --gradient_accumulation_steps $gradient_acc_steps \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 5 \
    --eval_strategy "epoch" \
    --learning_rate $lr \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --logging_dir "$exp_dir/$identity/tblogs" \
    --dataloader_num_workers 4 
#! /usr/bin/env bash 

set -ue
type="lora" # available choices: full/lora

huggingface_ref_dir=./ckpt
# bmtrain_ckpt_path="/FM9G4770B/apps/fm9g_7b/data/checkpoints/7b/5/fm9g_live_checkpoint-5.pt"
bmtrain_ckpt_path="/FM9G4770B/apps/fm9g_7b/9g_models/7b.pt"
# huggingface_save_dir="/FM9G4770B/apps/fm9g_7b/9g_models/hf-5"
huggingface_save_dir="/FM9G4770B/apps/fm9g_7b/9g_models/hf-lora-10"
# huggingface_save_dir="/FM9G4770B/apps/fm9g_4b/9g_models/hf-lora-5"

lora_ckpt_path="/FM9G4770B/apps/fm9g_7b/data/checkpoints/7b_lora/10/fm9g_live_checkpoint-10.pt"
lora_config_path="/FM9G4770B/apps/fm9g_7b/train_configs/7b_lora.json"


if [[ $type == "full" ]]; then

    python bmtrain_2_hgface.py \
        --huggingface-ref-dir $huggingface_ref_dir \
        --bmtrain-ckpt-path $bmtrain_ckpt_path \
        --huggingface-save-dir $huggingface_save_dir

elif [[ $type == "lora" ]]; then
    python bmtrain_2_hgface.py \
        --huggingface-ref-dir $huggingface_ref_dir \
        --bmtrain-ckpt-path $bmtrain_ckpt_path \
        --huggingface-save-dir $huggingface_save_dir \
        --lora-ckpt-path $lora_ckpt_path \
        --lora-config-path $lora_config_path

fi
#! /usr/bin/env bash 

set -ue
type="lora" # available choices: full/lora

huggingface_ref_dir=./ckpt
# bmtrain_ckpt_path="/FM9G4770B/apps/fm9g_4b/data/checkpoints/4b/10/fm9g_live_checkpoint-10.pt"
bmtrain_ckpt_path="/FM9G4770B/apps/fm9g_4b/9g_models/4b.pt"
# huggingface_save_dir="/FM9G4770B/apps/fm9g_4b/9g_models/hf-10"
huggingface_save_dir="/FM9G4770B/apps/fm9g_4b/9g_models/hf-lora-5"

lora_ckpt_path="/FM9G4770B/apps/fm9g_4b/data/checkpoints/4b_lora/5/fm9g_live_checkpoint-5.pt"
lora_config_path="/FM9G4770B/apps/fm9g_4b/train_configs/4b_lora.json"


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
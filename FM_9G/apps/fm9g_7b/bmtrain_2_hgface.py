# 替换ckpt name
import os
import json
import shutil
import argparse

import torch
from huggingface_hub import save_torch_state_dict

parser = argparse.ArgumentParser()
parser.add_argument("--huggingface-ref-dir", "-r", type=str, required=True, help="path to copy configs and tokenizers")
parser.add_argument("--bmtrain-ckpt-path", "-i", type=str, required=True)
parser.add_argument("--huggingface-save-dir", "-o", type=str, required=True)
parser.add_argument("--lora-ckpt-path", type=str, default=None)
parser.add_argument("--lora-config-path", type=str, default=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name_replace_dict = {
    "input_embedding.": "model.embed_tokens.",
    "output_layernorm.": "norm.",
    "encoder.": "model.",
    "self_att.self_attention.project_q.": "self_attn.q_proj.",
    "self_att.self_attention.project_k.": "self_attn.k_proj.",
    "self_att.self_attention.project_v.": "self_attn.v_proj.",
    "self_att.self_attention.attention_out.": "self_attn.o_proj.",
    "self_att.layernorm_before_attention.": "input_layernorm.",
    "ffn.layernorm_before_ffn.": "post_attention_layernorm.",
    "ffn.ffn.w_in.w_0.": "mlp.gate_proj.",
    "ffn.ffn.w_in.w_1.": "mlp.up_proj.",
    "ffn.ffn.w_out.": "mlp.down_proj."
}

if __name__ == "__main__":
    args = parser.parse_args()
    output_dir = args.huggingface_save_dir
    bmt_ckpt_path = args.bmtrain_ckpt_path
    config_copy_path = args.huggingface_ref_dir
    lora_path = args.lora_ckpt_path
    lora_config_path = args.lora_config_path

    os.makedirs(output_dir, exist_ok=True)

    print(f"loading {bmt_ckpt_path}...")
    bmt_ckpt = torch.load(bmt_ckpt_path, map_location="cpu")
    print("done loading")

    lora_mat_dict = {}
    if lora_path is not None:
        assert lora_config_path is not None
        with open(lora_config_path) as f:
            lora_config = json.load(f)['finetune']
            lora_r = lora_config["lora_r"]
            lora_alpha = lora_config["lora_alpha"]
        lora_ckpt = torch.load(lora_path)
        for k, v in lora_ckpt.items():
            if "lora_A" in k:
                weight_name = k.replace(".lora.lora_A", "")
                lora_A = v
                lora_B = lora_ckpt[k.replace("lora_A", "lora_B")]
                lora_mat = torch.mm(lora_B.to(device), lora_A.to(device)).to(dtype=torch.float32).to(device="cpu")
                lora_mat_dict[weight_name] = lora_mat

    new_ckpt = {}
    for k, v in bmt_ckpt.items():
        new_weight_name = k
        for replace_keyword, new_keyword in name_replace_dict.items():
            if replace_keyword in new_weight_name:
                new_weight_name = new_weight_name.replace(replace_keyword, new_keyword)
        if ".weight" in k:
            lora_mat = lora_mat_dict.get(k.replace(".weight", ""), None)
            if lora_mat is not None:
                new_ckpt[new_weight_name] = (v.to(device) + lora_alpha / lora_r * lora_mat.to(device=device, dtype=v.dtype)).to("cpu")
            else:
                new_ckpt[new_weight_name] = v
        else:
            new_ckpt[new_weight_name] = v

    print(f"writing model ckpt into {output_dir}...")
    save_torch_state_dict(new_ckpt, output_dir)
    print("done writing")

    for file in os.listdir(config_copy_path):
        if ".json" in file or "token" in file or ".py" in file:
            shutil.copy(
                os.path.join(config_copy_path, file),
                os.path.join(output_dir, file)
            )

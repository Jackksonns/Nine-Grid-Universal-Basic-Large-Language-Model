# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import torch
import argparse

parser = argparse.ArgumentParser(description='Load and save model weights with specified paths.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory.')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the new weights.')
parser.add_argument('--layer_num', type=int, required=True, help='The layers of model')


args = parser.parse_args()

src_path = args.model_path
dst_path = args.output_path
layer_num = args.layer_num

def convert_hf_to_fm9g():
    ckpt = torch.load(src_path)
    new_ckpt = OrderedDict()

    new_ckpt['input_embedding.weight'] = ckpt['model.embed_tokens.weight']
    new_ckpt["encoder.output_layernorm.weight"] = ckpt['model.norm.weight']
    for i in range(layer_num):
        new_ckpt[f"encoder.layers.{i}.self_att.self_attention.project_q.weight"] = ckpt[f"model.layers.{i}.self_attn.q_proj.weight"]
        new_ckpt[f"encoder.layers.{i}.self_att.self_attention.project_k.weight"] = ckpt[f"model.layers.{i}.self_attn.k_proj.weight"]
        new_ckpt[f"encoder.layers.{i}.self_att.self_attention.project_v.weight"] = ckpt[f"model.layers.{i}.self_attn.v_proj.weight"]
        new_ckpt[f"encoder.layers.{i}.self_att.self_attention.attention_out.weight"] = ckpt[f"model.layers.{i}.self_attn.o_proj.weight"]
        new_ckpt[f"encoder.layers.{i}.self_att.layernorm_before_attention.weight"] = ckpt[f"model.layers.{i}.input_layernorm.weight"]
        new_ckpt[f"encoder.layers.{i}.ffn.layernorm_before_ffn.weight"] = ckpt[f"model.layers.{i}.post_attention_layernorm.weight"]
        
        new_ckpt[f"encoder.layers.{i}.ffn.ffn.w_in.w_0.weight"] = ckpt[f'model.layers.{i}.mlp.gate_proj.weight']
        new_ckpt[f"encoder.layers.{i}.ffn.ffn.w_in.w_1.weight"] = ckpt[f'model.layers.{i}.mlp.up_proj.weight']
        new_ckpt[f"encoder.layers.{i}.ffn.ffn.w_out.weight"] = ckpt[f'model.layers.{i}.mlp.down_proj.weight']

    torch.save(new_ckpt, dst_path)

def convert_fm9g_to_hf():
    state = torch.load(src_path)
    new_state = {}
    new_state["model.embed_tokens.weight"] = state["input_embedding.weight"]
    new_state["model.norm.weight"] = state["encoder.output_layernorm.weight"]
    for lid in range(40):
        print(lid)
        new_state[f"model.layers.{lid}.self_attn.q_proj.weight"] = state[f"encoder.layers.{lid}.self_att.self_attention.project_q.weight"]
        new_state[f"model.layers.{lid}.self_attn.k_proj.weight"] = state[f"encoder.layers.{lid}.self_att.self_attention.project_k.weight"]
        new_state[f"model.layers.{lid}.self_attn.v_proj.weight"] = state[f"encoder.layers.{lid}.self_att.self_attention.project_v.weight"]

        new_state[f"model.layers.{lid}.self_attn.o_proj.weight"] = state[f"encoder.layers.{lid}.self_att.self_attention.attention_out.weight"]
        new_state[f"model.layers.{lid}.mlp.gate_proj.weight"] = state[f"encoder.layers.{lid}.ffn.ffn.w_in.w_0.weight"]
        new_state[f"model.layers.{lid}.mlp.up_proj.weight"] = state[f"encoder.layers.{lid}.ffn.ffn.w_in.w_1.weight"]
        new_state[f"model.layers.{lid}.mlp.down_proj.weight"] = state[f"encoder.layers.{lid}.ffn.ffn.w_out.weight"]

        new_state[f"model.layers.{lid}.input_layernorm.weight"] = state[f"encoder.layers.{lid}.self_att.layernorm_before_attention.weight"]
        new_state[f"model.layers.{lid}.post_attention_layernorm.weight"] = state[f"encoder.layers.{lid}.ffn.layernorm_before_ffn.weight"]
    del state
    state = None
    torch.save(new_state, f"{dst_path}pytorch_model.bin")
    

if __name__ == "__main__":
    convert_hf_to_fm9g()
 

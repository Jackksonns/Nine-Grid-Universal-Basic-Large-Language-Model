import sys
import shutil
import json
sys.path.insert(0, "/home/wangshuo1/projects/CPM-9G/gejiu_train")
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from cpm.cpm9g.models import CPM9GTorch, CPM9GConfig

from transformers import AutoTokenizer, AutoConfig
from cpm.cpm9g import CPM9GTokenizer as BMTCPM9GTokenizer
from cpm.cpm9g import CPM9GTokenizer
from cpm.cpm9g.generation.cpm9g import CPM9GBeamSearch

source_path = "/data/public/zwl_data/11b-base/"
target_path = "/home/wangshuo1/projects/CPM-9G/convert_to_hf/11b-base-hf/"
file_name = "11b.pt"

def convert_pkl():
    shutil.copyfile(f"{source_path}vocabs.txt", f"{target_path}vocabs.txt")
    with open(f"{source_path}config.json") as f:
        bmt_config = json.load(f)
    config = {
        "architectures": [
          "LlamaForCausalLM"
        ],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": bmt_config['dim_model'],
        "initializer_range": 0.02,
        "intermediate_size": bmt_config['dim_ff'],
        "max_length": 4096,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": bmt_config['num_heads'],
        "num_hidden_layers": bmt_config['num_layers'],
        "num_key_value_heads": bmt_config['num_kv_heads'],
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.31.0",
        "use_cache": True,
        "vocab_size": bmt_config['vocab_size'],
    }
    with open(f"{target_path}config.json", "w") as f:
        json.dump(config, f)

    state = torch.load(f"{source_path}{file_name}")
    new_state = {}
    new_state["model.embed_tokens.weight"] = state["input_embedding.weight"]
    new_state["lm_head.weight"] = state["lm_head.weight"]
    new_state["model.norm.weight"] = state["encoder.output_layernorm.weight"]
    layer_num = bmt_config['num_layers']
    for lid in range(layer_num):
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
    torch.save(new_state, f"{target_path}pytorch_model.bin")

def test():
    config = LlamaConfig.from_pretrained(f"{target_path}")
    tokenizer = CPM9GTokenizer(f"{target_path}vocabs.txt")
    model = LlamaForCausalLM.from_pretrained(f"{target_path}").cuda()

    text = "请介绍一下清华大学："
    inputs = torch.tensor([[tokenizer.bos_id] + tokenizer.encode(text)]).cuda()
    output = model.generate(inputs, max_length=200)[0].tolist()
    print(tokenizer.decode(output))

if __name__ == "__main__":
    convert_pkl()
    test()

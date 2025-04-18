import torch
from transformers import AutoModelForCausalLM

# === 加载模型 ===
model = AutoModelForCausalLM.from_pretrained(
    "/public/kqa/9G_4_7_70/70b", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
state_dict = model.state_dict()

# === 替换规则 ===
name_replace_dict = {
    "model.embed_tokens.": "input_embedding.",
    "model.norm.weight": "encoder.output_layernorm.weight",
    "model.": "encoder.",
    "self_attn.q_proj.": "self_att.self_attention.project_q.",
    "self_attn.k_proj.": "self_att.self_attention.project_k.",
    "self_attn.v_proj.": "self_att.self_attention.project_v.",
    "self_attn.o_proj.": "self_att.self_attention.attention_out.",
    "input_layernorm.": "self_att.layernorm_before_attention.",
    "post_attention_layernorm.": "ffn.layernorm_before_ffn.",
    "mlp.gate_proj.": "ffn.ffn.w_in.w_0.",
    "mlp.up_proj.": "ffn.ffn.w_in.w_1.",
    "mlp.down_proj.": "ffn.ffn.w_out.",
}

# === 替换 key ===
new_state_dict = {}
for k, v in model.named_parameters():
    new_k = k
    for old, new in name_replace_dict.items():
        if old in new_k:
            new_k = new_k.replace(old, new)
    new_state_dict[new_k] = v

# === 保存为 .pt 文件 ===
torch.save(new_state_dict, "./9g_models/70b.pt")
print("✅ 已保存为 ./9g_models/70b.pt")
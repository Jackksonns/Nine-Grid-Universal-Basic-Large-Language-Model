import torch
import json
import os
import bitsandbytes as bnb
import copy
from bitsandbytes.functional import dequantize_4bit
from peft import PeftModel    
from peft.utils import _get_submodules
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def dequantize_model(model, tokenizer, to='./dequantized_model', dtype=torch.bfloat16, device="cpu"):
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """

    # # Delete the model object if it exists
    # if os.path.exists(to):
    #     shutil.rmtree(to)
    # os.makedirs(to, exist_ok=True)

    cls = bnb.nn.Linear4bit

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                # print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)

                quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(
                    module.in_features, 
                    module.out_features, 
                    bias=(module.bias is not None),
                    dtype=dtype
                )
                new_module.weight = torch.nn.Parameter(weights)

                if module.bias is not None:
                    new_module.bias = torch.nn.Parameter(module.bias.data.to(dtype))

                # new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                # new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

        model.is_loaded_in_4bit = False

        # save_model(model, tokenizer, to)
        
        return model

model_path = "/data/public/kqa/9G_4_7_70/70b"
adapters_path = "/FM9G4770B/apps/fm9g_70b_hf/data/checkpoints/huggingface_70b/testqlora/checkpoint-100"
output_path ="./data/checkpoints/huggingface_70b/q_merged"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,device_map="auto", trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 是否进行4bit量化
    load_in_8bit=False,  # 是否进行8bit量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
    bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
    bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正态度分布的int4
    bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
    llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
    llm_int8_has_fp16_weight=False,  # 是否启用混合精度
    # llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
    llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=quantization_config,
)

model = dequantize_model(model, tokenizer, to='')

model = PeftModel.from_pretrained(model, adapters_path)
model = model.merge_and_unload()

model.save_pretrained(output_path)
config_path = os.path.join(output_path, "config.json")

# 读取 config.json
with open(config_path, "r") as f:
    config = json.load(f)

# 删除指定的键（例如 "architectures"）
if "quantization_config" in config:
    del config["quantization_config"]

# 保存修改后的 config.json
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

tokenizer.save_pretrained(output_path)
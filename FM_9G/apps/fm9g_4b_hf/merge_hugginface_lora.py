import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/public/kqa/9G_4_7_70/4b"
adapters_path = "/FM9G4770B/apps/fm9g_4b_hf/data/checkpoints/huggingface_4b/testqlora/checkpoint-50"
output_path ="./data/checkpoints/huggingface_4b/merged"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,device_map="auto", trust_remote_code=True)

model = PeftModel.from_pretrained(model, adapters_path)
model = model.merge_and_unload()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
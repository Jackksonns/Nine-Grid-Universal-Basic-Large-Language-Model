# import os
# import json
# import random
# from datasets import load_dataset
# from transformers import AutoTokenizer

# random.seed(3014)

# data_path = "/FM9G4770B/data_test/raw_data/medical_o1_sft_Chinese.json"
# tokenizer_path = "./ckpt"
# output_path = "/FM9G4770B/data_test/converted_data/medical"

# os.makedirs(output_path, exist_ok=True)

# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
# dataset = load_dataset("json", data_files=data_path, split="train")
# dataset = dataset.train_test_split(test_size=0.05, seed=3014)

# def is_chinese(strs):
#     chines_count = 0
#     for _char in strs:
#         if '\u4e00' <= _char <= '\u9fa5':
#             chines_count += 1
#     return chines_count/len(strs) > 0.5

# with open(os.path.join(output_path, 'train.jsonl'), 'w') as f:
#     for sample in dataset['train']:
#         if is_chinese(sample['Question']):
#             conversation = [{
#                 'role': 'User',
#                 'content': sample['Question']
#             }]
#             input_text = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True, tokenize=False)
#             response_text = sample['Response'] + tokenizer.eos_token
#             f.write(json.dumps({
#                 'input': input_text,
#                 'output': response_text
#             }, ensure_ascii=False) + '\n')

# with open(os.path.join(output_path, 'test.jsonl'), 'w') as f:
#     for sample in dataset['test']:
#         if is_chinese(sample['Question']):
#             conversation = [{
#                 'role': 'User',
#                 'content': sample['Question']
#             }]
#             input_text = tokenizer.apply_chat_template(conversation=conversation, add_generation_prompt=True, tokenize=False)
#             response_text = sample['Response'] + tokenizer.eos_token
#             f.write(json.dumps({
#                 'input': input_text,
#                 'output': response_text
#             }, ensure_ascii=False) + '\n')


import os
import json
import tqdm

import random
from datasets import load_dataset

random.seed(3014)

data_path = "/FM9G4770B/data_test/raw_data/medical_o1_sft_Chinese.json"
tokenizer_path = "./ckpt"
output_path = "/FM9G4770B/data_test/converted_data/medical"


os.makedirs(output_path, exist_ok=True)
print("loading dataset...")
dataset = load_dataset("json", data_files=data_path, split="train")
print("done")
dataset = dataset.train_test_split(test_size=0.05, seed=3014)

def is_chinese(strs):
    chines_count = 0
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            chines_count += 1
    return chines_count/len(strs) > 0.5

with open(os.path.join(output_path, 'train.jsonl'), 'w') as f:
    for sample in tqdm.tqdm(dataset['train'], desc="writing train json"):
        if is_chinese(sample['Question']):
            f.write(json.dumps({
                'input': sample['Question'],
                'output': sample['Response']
            }, ensure_ascii=False) + '\n')

with open(os.path.join(output_path, 'test.jsonl'), 'w') as f:
    for sample in tqdm.tqdm(dataset['test'], desc="writing eval json"):
        if is_chinese(sample['Question']):
            f.write(json.dumps({
                'input': sample['Question'],
                'output': sample['Response']
            }, ensure_ascii=False) + '\n')

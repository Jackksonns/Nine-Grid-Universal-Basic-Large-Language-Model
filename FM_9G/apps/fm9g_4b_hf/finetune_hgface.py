# -*- coding: utf-8 -*-
import sys
import json
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

sys.path.append("../../")
from fm9g.dataset.indexed_dataset import IndexedDataset


import importlib.util
import os

def load_transform_func(config, config_path):
    # 从配置中获取 transforms 文件路径
    transform_path = os.path.join(os.path.dirname(config_path),config.get("transforms"))
    if not transform_path or not os.path.exists(transform_path):
        raise FileNotFoundError(f"Transforms 文件未找到：{transform_path}")
    
    # 通过文件名生成模块名称（去掉扩展名）
    module_name = os.path.splitext(os.path.basename(transform_path))[0]
    
    # 加载模块的 spec
    spec = importlib.util.spec_from_file_location(module_name, transform_path)
    if spec is None:
        raise ImportError(f"无法加载模块规格：{module_name}")
    
    # 利用 spec 创建模块对象
    module = importlib.util.module_from_spec(spec)
    # 执行模块，将模块中的代码加载到 module 对象中
    spec.loader.exec_module(module)
    
    # 返回模块中定义的 transform 函数
    if hasattr(module, "transform"):
        return module.transform
    else:
        raise AttributeError(f"模块 {module_name} 中没有找到 transform 函数")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/")

@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the indexed training data dir."},        
    )
    train_data_config_path: str = field(
        default=None,
        metadata={"help": "Path to the training config file. Will overwrite 'train_data_path'"},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the indexed test data dir."},
    )


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=False)
    qlora: bool = field(default=False)
    output_dir: str = field(default="outputs")
    lora_modules: str = field(default="[\"q_proj\", \"v_proj\", \"k_proj\", \"o_proj\"]")
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)


class SupervisedDataset(IndexedDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        transform_func=None
    ):
        super(SupervisedDataset, self).__init__(path=data_path)
        # self.data = IndexedDataset(path=data_path)
        self.tokenizer = tokenizer
        self.transform_func = transform_func

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = json.loads(super().__getitem__(idx).decode('utf-8'))
        if self.transform_func is not None:
            res = self.transform_func(sample,0,0)
            user_inputs = res["input"]
            ai_response = res["output"]
            return {
            'input_ids': user_inputs,
            'labels': ai_response
            }
        else:
            if "input" in sample and len(sample["input"]) > 0:
                user_inputs = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample["input"]}], 
                    tokenize=False,
                    add_generation_prompt=True
                )
                ai_response = sample["output"] + self.tokenizer.eos_token
            else:
                user_inputs = ""
                ai_response = sample['output'] + "</s>"
            return {
                'input_ids': user_inputs,
                'labels': ai_response
            }
        

class MixedDataset(Dataset):

    def __init__(self, datasets, probs):
        lengths = []
        for d in datasets:
            lengths.append(len(d))
        lengths = np.asarray(lengths, dtype=np.int32)
        probs = np.asarray(probs)
        normed_probs = np.round(probs/probs.min()).astype(np.int32)
        self.sample_cycle = normed_probs.sum()
        self.length = int(np.min(lengths/normed_probs)) * self.sample_cycle
        # print(normed_probs)
        self.dataset_schedule = []
        self.id_schedule = []
        for dataset_id, normed_len in enumerate(normed_probs):
            self.dataset_schedule.extend([dataset_id]*normed_len)
            self.id_schedule.extend(range(0, normed_len))
        self.dataset_lengths_per_cycle = normed_probs
        self.datasets = datasets
        print("{} datasets (length={}), using sampling cycle={} and prob={}".format(
            len(self.datasets), lengths.tolist(), self.sample_cycle, self.dataset_lengths_per_cycle.tolist()
        ))

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        cycle_count = index // self.sample_cycle
        index_in_cycle = index % self.sample_cycle
        dataset_id = self.dataset_schedule[index_in_cycle]
        data_id = self.dataset_lengths_per_cycle[dataset_id] * cycle_count + self.id_schedule[index_in_cycle]
        dataset_id = int(dataset_id)
        data_id = int(data_id)
        return self.datasets[dataset_id][data_id]
    
    def __len__(self):
        return self.length
    

class SupervisedCollector:
    
    def __init__(
            self, 
            tokenizer: PreTrainedTokenizer, 
            ignore_idx: int = -100, 
            max_model_length: int = 4096):
        self.tokenizer = tokenizer
        self.ignore_idx = ignore_idx
        self.max_model_length = max_model_length
    
    def collect_data(self, batch):
        if batch[0]['input_ids'] is None:
            total_inputs = [b['labels'] for b in batch]
            inputs = self.tokenizer(total_inputs, padding=True, max_length=self.max_model_length, truncation=True, return_tensors='pt')
            target = inputs.input_ids.clone()
        else:
            user_inputs = [b['input_ids'] for b in batch]
            total_inputs = [b['input_ids'] + b['labels'] for b in batch]
            inputs = self.tokenizer(total_inputs, padding=True, max_length=self.max_model_length, truncation=True, return_tensors='pt')
            users = self.tokenizer(user_inputs, padding=True, max_length=self.max_model_length, truncation=True, return_tensors='pt')
            users_len = users.attention_mask.sum(dim=1, keepdims=False)
            target = inputs.input_ids.clone()
            for i in range(len(batch)):
                target[i, :users_len[i]] = self.ignore_idx
        
        return {
            'input_ids': inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            'labels': target
        }
        

def load_model_and_tokenizer(
    model_path: str,
    max_length: int = 4096,
    use_lora: bool = True,
    qlora: bool = False,
    bf16: bool = False,
    fp16: bool = False,
    lora_modules: list = [],
    lora_r: int = 64,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
):
    """load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    if qlora:
        assert use_lora, "use_lora must be True when use_qlora is True"
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
            torch_dtype=dtype,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            task_type=TaskType.CAUSAL_LM,
            target_modules=(lora_modules),
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer


if __name__ == "__main__":

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        use_lora=training_args.use_lora,
        qlora=training_args.qlora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        lora_modules=json.loads(training_args.lora_modules),
        lora_r = training_args.lora_r,
        lora_alpha = training_args.lora_alpha,
        lora_dropout = training_args.lora_dropout
    )

    if data_args.train_data_config_path is not None:
        with open(data_args.train_data_config_path) as f:
            dataset_configs = json.load(f)
        sub_datasets = []
        probabilities = []
        norm_factor = 0
        for dc in dataset_configs:
            transform_func = load_transform_func(dc, data_args.train_data_config_path)
            sub_datasets.append(
                SupervisedDataset(data_path=dc["path"], tokenizer=tokenizer, transform_func=transform_func)
            )
            probabilities.append(dc['abs_weight'])
            norm_factor += dc['abs_weight']
        probabilities = [p/norm_factor for p in probabilities]
        train_dataset = MixedDataset(sub_datasets, probs=probabilities)
    elif data_args.train_data_path is not None:
        train_dataset = SupervisedDataset(
            data_path=data_args.train_data_path,
            tokenizer=tokenizer,
            transform_func=None
        )
    else:
        raise ValueError("must give the path to the training dataset jsonl file or training config file.")
    if data_args.eval_data_path is not None:
        eval_dataset = SupervisedDataset(
            data_path=data_args.eval_data_path,
            tokenizer=tokenizer,
        )
    else:
        eval_dataset = None
        training_args.eval_strategy = 'no'
    
    collector = SupervisedCollector(tokenizer=tokenizer, ignore_idx=-100, max_model_length=training_args.model_max_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collector.collect_data
    )

    trainer.train()
    # save the incremental PEFT weights, more details can be found in https://huggingface.co/blog/peft
    model.save_pretrained(training_args.output_dir)
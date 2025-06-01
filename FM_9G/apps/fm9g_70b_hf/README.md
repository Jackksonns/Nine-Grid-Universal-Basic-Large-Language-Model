<style type="text/css">
    h1 { counter-reset: h2counter; }
    h2 { counter-reset: h3counter; }
    h3 { counter-reset: h4counter; }
    h4 { counter-reset: h5counter; }
    h5 { counter-reset: h6counter; }
    h6 { }
    h2:before {
      counter-increment: h2counter;
      content: counter(h2counter) ".\0000a0\0000a0";
    }
    h3:before {
      counter-increment: h3counter;
      content: counter(h2counter) "."
                counter(h3counter) ".\0000a0\0000a0";
    }
    h4:before {
      counter-increment: h4counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) ".\0000a0\0000a0";
    }
    h5:before {
      counter-increment: h5counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) ".\0000a0\0000a0";
    }
    h6:before {
      counter-increment: h6counter;
      content: counter(h2counter) "."
                counter(h3counter) "."
                counter(h4counter) "."
                counter(h5counter) "."
                counter(h6counter) ".\0000a0\0000a0";
    }
</style>

# 九格大模型使用文档

本文档介绍九格大模型70B版本的 transformers 全量微调与lora微调方法。

## 目录

<!-- - [仓库目录结构](#仓库目录结构) -->

- [九格大模型使用文档](#九格大模型使用文档)
  - [目录](#目录)
  - [模型下载](#模型下载)
  - [环境配置](#环境配置)
  - [数据处理流程](#数据处理流程)
  - [单机全量微调](#单机全量微调)
  - [多机全量微调](#多机全量微调)
  - [查看训练情况](#查看训练情况)
  - [lora微调](#lora微调)
  - [Qlora微调](#Qlora微调)

## 模型下载

可在
https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/FM9G_70B_SFT_MHA.tar
下载70B模型。下载完成后解压tar包，并将其中所有文件放置于**此README文件所在的目录ckpt文件夹**下。

## 环境配置

完成模型下载后，需要安装所需的各项依赖。除本文介绍的70B模型外，九格还有4B和70B两种不同的版本可供选用，4B、7B、70B模型的依赖完全相同，如果已经配置完成其中任意一种，即可跳过此环境配置步骤。环境配置步骤分为Conda环境安装、Pytorch安装、其余依赖项安装三步。

### conda 环境安装

#### 使用python 3.10.16 创建conda环境

```shell
conda create -n fm-9g python=3.10.16
```

#### 激活环境

```shell
conda activate fm-9g
```

#### 安装Pytorch

如果不使用vllm推理，可使用如下方法安装Pytorch

```shell
# 需要先查看CUDA版本，根据CUDA版本挑选合适的pytorch版本 (测试CUDA版本为12.2)
pip3 install torch==2.3.0
```

#### 安装bmtrain
```shell
pip install bmtrain
```

#### 安装transformers
```shell
pip install transformers==4.44.0
```

#### 安装其他依赖包

```shell
pip install h5py
pip install accelerate==0.34.0
pip install tensorboardX
pip install scipy
pip install datamodel_code_generator
pip install jsonschema
```

## 数据处理流程
### 单个数据集处理
预训练语料为无监督形式，不需要区分问题与答案，但需要将数据转为index后进行模型训练。我们拿到的原始数据可能是两种形式：
- 文件格式为.txt的原始文本，处理流程为：数据→jsonl格式的数据→index数据
- 文件格式为.jsonl的文本数据，处理流程为:数据→index数据
1. 参考以下脚本，将txt数据处理为jsonl格式：
``` python
# convert_txt2jsonl.py
import json
import sys
for line in sys.stdin:
    if line.strip() == "":
        continue
    temp_json = {"input": "", "output": line.strip()}#预训练计算Loss时只计算output部分，所以input字段为空
    print(json.dumps(temp_json, ensure_ascii=False))
```
脚本使用方法如下，其中pretrain.txt是原始txt数据，pretrain.jsonl是输出的jsonl格式数据：
```shell
cat pretrain.txt | python convert_txt2jsonl.py > pretrain.jsonl
```
输出的jsonl文件中，其中每一行有两个字段：input字段与output字段。例如：
```JSON
{"input":"","output":"中国的首都是北京。"}
```

2. 其他数据格式转换参考
使用模版对数据进行转换，可以参考脚本./prepare_data.py，按照具体数据格式进行修改。我们提供了一份对[medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)数据进行格式转换的示例代码，原始数据可在Huggingface处找到，或[点击此链接](https://drive.weixin.qq.com/s?k=AKIAqQfNADgRYnIYHF)下载。完成下载后，可将以下代码中`data_path`变量替换为medical_o1_sft_Chinese.json文件所在的路径。

```python
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
```
转换前：
```json
{
    "Question": "根据以下症状：38岁女性患者，出现颈前肿痛、胸闷不适、口苦咽干、急躁易怒、心悸多汗、舌苔薄黄、脉弦数，请判断其对应的中医证型是什么？",
    "Complex_CoT": "这个38岁的女性患者，嗯，颈前肿痛让我首先想到的是甲状腺问题。她还有胸闷不适，感觉好像肝功能也受到了影响，可能有什么事情让她的情绪不太平衡。再看她说的口苦咽干，似乎有些内火，不知道是不是肝火上升。 \n\n她还提到了急躁易怒，这基本上就是肝郁的一个标志了。然后她心悸多汗，这大概是热象在作怪，整个儿有火上扰心的迹象。舌苔薄黄、脉弦数这些更是反馈出体内有热。我想这些都在说明她的内部火气过盛。\n\n对这种情况，我得先想想是不是肝胆出了问题，肝火旺自己的可能性很大。但那个颈前肿，我还是倾向于甲状腺，应该还有一些气机不畅的问题。 \n\n我们再来理理，肝郁确实是个可行的证型，她很可能长期情绪压抑，但那个口苦还可能说明胆的参与，可能胆也有火。\n\n再来，这种心烦易怒，还心悸多汗，总讓人觉得心火挺旺，再看看甲状腺，嗯，有可能她的心火是真的因为肝火影响到了心。 \n\n哦对了，那些胸闷不适，她可能是胃气上逆。胃部的热很可能对她整体的火气有贡献。再对比她其他症状，胃热和火气看来联系很大呢。\n\n这样一想，这个患者的情况可能是肝郁胃热比较贴切。再检查一下，不错，符合这些症状集的表现。而且她的热象和心的症状似乎也是彼此关联的，感觉这个诊断很合理。\n\n最后，结合所有信息，肝郁和胃热的结合更能说明她的身体状态。大概没落下什么重大的症状。",
    "Response": "根据患者的症状分析，这位38岁的女性可能对应的中医证型是“肝郁化火”和“胃热”结合的情况。\n\n首先，患者的颈前肿痛提示甲状腺可能有问题，而口苦咽干与中医的肝胆相关，尤其是胆火上扰。此外，急躁易怒显示出典型的肝火亢盛的特征，这是因情志不畅，肝气郁结，进一步化火的表现。心悸多汗、舌苔薄黄、脉弦数等症状都指向体内火气上涨，尤其是在肝火影响下，容易出现心火的症状。\n\n结合这些症状，患者的“胸闷不适”和相关的胃气上逆也可能是胃热表现的一部分。整体来看，她的症候结构中体现出的肝郁、心火亢盛、胃热上扰等相互交织，这与“肝郁化火”伴随“胃热”的证型较为贴合。\n\n因此，综合考虑患者的症状表现及其相互关系，“肝郁化火”和“胃热”是较为合适的中医证型判断。针对这种情况，治疗上可能需要疏肝解郁，清肝泻火，同时兼顾和胃降逆的措施。"
}
```
转换后：
```json
{"input": "根据以下症状：38岁女性患者，出现颈前肿痛、胸闷不适、口苦咽干、急躁易怒、心悸多汗、舌苔薄黄、脉弦数，请判断其对应的中医证型是什么？", "output": "根据患者的症状分析，这位38岁的女性可能对应的中医证型是“肝郁化火”和“胃热”结合的情况。\n\n首先，患者的颈前肿痛提示甲状腺可能有问题，而口苦咽干与中医的肝胆相关，尤其是胆火上扰。此外，急躁易怒显示出典型的肝火亢盛的特征，这是因情志不畅，肝气郁结，进一步化火的表现。心悸多汗、舌苔薄黄、脉弦数等症状都指向体内火气上涨，尤其是在肝火影响下，容易出现心火的症状。\n\n结合这些症状，患者的“胸闷不适”和相关的胃气上逆也可能是胃热表现的一部分。整体来看，她的症候结构中体现出的肝郁、心火亢盛、胃热上扰等相互交织，这与“肝郁化火”伴随“胃热”的证型较为贴合。\n\n因此，综合考虑患者的症状表现及其相互关系，“肝郁化火”和“胃热”是较为合适的中医证型判断。针对这种情况，治疗上可能需要疏肝解郁，清肝泻火，同时兼顾和胃降逆的措施。"}
```


2. jsonl格式转index。脚本位于./data_preprocess.sh，内容如下：

```shell
python convert_json2index.py \
  --path ../../data_test/converted_data/alpaca/alpaca_zh.jsonl \#jsonl文件
  --language zh \ #只能选择zh（中文）或者en（英文）
  --output ../../data_test/indexed_data/data_ralpaca_zh #存放生成的index的目录，与原先存放jsonl文件的目录不能相同
```

<!-- 脚本运行成功时，会有如下显示：（不需要用hadoop所以不用管hadoop: not found的警告信息）

![脚本运行成功后的显示](./055bf7ce-faab-403b-a7ee-896279bee11f.png) -->

转完后，在index的目录下会生成四个文件：data.jsonl（原先的jsonl数据）、index、index.h5、meta.json（记录数据集信息，包含 "language", "nlines", "nbytes", "length_distribute", "avg_token_per_line", "hdfs_path", "data_sample"字段）。
这里有一个meta.json的例子：
```JSON
{"language": "zh", "nlines": 48818, "nbytes": 33836014, "length_distribute": {"less_4k": 48818, "4k-8k": 0, "8k-16k": 0, "16k-32k": 0, "32k-64k": 0, "64k-128k": 0, "128k-256k": 0, "more_256k": 0}, "avg_token_per_line": 204.21905649555583, "data_sample": {"input": "<用户>保持健康的三个提示。<AI>", "output": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心..."}}
```

### 多个数据集混合
我们支持多个数据集的混合读入，并设置不同数据集的比例。为此，需要准备一个数据混合的json文件，来指导训练过程中的数据读取策略，示例如下：
```JSON
[
    {
        "dataset_name": "medical_o1_reasoning",
        "task_name": "medical_o1_reasoning",
        "abs_weight": 0.5,
        "path": "/FM9G4770B/data_test/indexed_data/medical_train",
        "transforms": "./add_template.py",
        "allow_repeat": true,
        "nlines": 17278,
        "ave_tokens_per_line": 351,
        "total_tokens": 0.0006
    },
    {
        "dataset_name": "ralpaca_zh",
        "task_name": "ralpaca_zh",
        "abs_weight": 0.5,
        "path": "/FM9G4770B/data_test/indexed_data/data_ralpaca_zh",
        "transforms": "./script_cpmc.py",
        "allow_repeat": true,
        "nlines": 48818,
        "ave_tokens_per_line": 204,
        "total_tokens": 0.001
    }
]
```
其中abs_weight需要自行设计；path、nlines、ave_tokens_per_line可以参考生成index时的meta.json进行填写；allow_repeat为数据集是否需要复制；total_tokens为估计的数据集token总数，以b（十亿）为单位，例如0.1代表0.1b个token，transforms为读入训练数据的脚本路径，该脚本可以参考以下代码：
```python
# script_cpmc.py
import random

def rand(n: int, r: random.Random):
    return int(r.random() * n)

def transform(data, num_sample: int, r: random.Random):
    if 'input' in data:
        _input = data['input']
    else:
        _input = ""
    
    if 'output' in data:
        _output = data['output']
    else:
        _output = ""
    return {"input": _input, 
            "output": _output,
            }
```
以下是chat模板
```python
# add_template.py

import random

def rand(n: int, r: random.Random):
    return int(r.random() * n)

def transform(data, num_sample: int, r: random.Random):
    if 'input' in data:
        user_input = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>User\n{}<|im_end|>\n<|im_start|>assistant\n".format(data['input'])
        ai_output = "{}<|im_end|>".format(data['output'])
    elif 'output' in data:
        user_input = ""
        ai_output = data['output']
    else:
        user_input = ''
        ai_output = ''

    return {
        "input": user_input, 
        "output": ai_output,
        }

```


## 单机全量微调
Transformers 全量微调优化程度较低，硬件需求高。建议使用 bmtrain 进行全量微调。 70B 模型文件较大，建议使用 Transformers 进行 qlora 微调。

训练脚本为
/FM9G4770B/apps/fm9g_70b_hf/run_huggingface.sh。
修改训练脚本中的的训练参数，如下所示：
```shell
#! /usr/bin/env bash

# 启用Tokenizer的并行处理以提高数据预处理效率
export TOKENIZERS_PARALLELISM=true
# 训练数据配置路径（包含多个子数据集的路径和权重等信息）
train_data_config_path="/FM9G4770B/apps/fm9g_70b_hf/dataset_configs/data_test.json"
# 验证数据路径，可以为空
eval_data_path="/FM9G4770B/data_test/indexed_data/indexed_test_data"
# 每张卡的训练 batch 大小
batch_size=2
# 最大长度
model_max_length=4096
# 梯度累积步数（相当于有效 batch_size = batch_size × gradient_acc_steps × GPU数 × 节点数）
gradient_acc_steps=1
# 学习率
lr=1e-4
# 本次实验的标识符，用于区分不同实验保存的路径
identity=test
# 预训练模型的路径
model_name_or_path="./ckpt"
# 是否使用LoRA微调
use_lora=false
# 是否使用QLoRA量化训练，开启时必须同时开启use_lora
qlora=false
# 实验输出目录
exp_dir="./data/checkpoints/huggingface_70b"
# 遇到未定义变量或错误即退出脚本，防止出错后继续执行
set -ue

# 使用 torchrun 启动分布式训练任务（8卡单机）
torchrun \
    --nproc_per_node 8 \                         # 每节点使用GPU数量
    --nnodes 1 \                                 # 节点数量
    --node_rank 0 \                              # 当前节点编号
    --master_addr localhost \                    # 主节点地址
    --master_port 6004 \                         # 主节点通信端口
    -m finetune_hgface \                         # 启动 Python 模块 finetune_hgface

    # 模型与数据相关参数
    --model_name_or_path $model_name_or_path \   # 模型路径
    --train_data_config_path $train_data_config_path \  # 多数据集配置文件
    --eval_data_path $eval_data_path \           # 验证集路径
    --use_lora $use_lora \                       # 是否启用LoRA
    --qlora $qlora \                             # 是否启用QLoRA
    --bf16 true \                                # 使用bfloat16训练

    # 输出与训练控制
    --output_dir "$exp_dir/$identity" \          # 模型保存路径
    --num_train_epochs 1 \                       # 训练轮数
    --per_device_train_batch_size $batch_size \  # 每张卡的batch大小
    --model_max_length $model_max_length \       # 最大长度
    --gradient_accumulation_steps $gradient_acc_steps \  # 梯度累积步数

    # 保存策略
    --save_strategy steps \                      # 按步数保存
    --save_steps 50 \                            # 每50步保存一次
    --save_total_limit 5 \                       # 最多保留5个checkpoint

    # 验证策略
    --eval_strategy "epoch" \                    # 每轮训练后进行一次评估

    # 优化器与学习率调度
    --learning_rate $lr \                        # 学习率
    --weight_decay 0.01 \                        # 权重衰减
    --adam_beta2 0.95 \                          # Adam优化器的 β2 参数
    --warmup_ratio 0.1 \                         # 学习率预热比例
    --lr_scheduler_type "linear" \               # 线性学习率衰减

    # 日志与监控
    --logging_steps 10 \                         # 每10步记录一次日志
    --report_to "tensorboard" \                  # 使用 TensorBoard 进行日志记录
    --logging_dir "$exp_dir/$identity/tblogs" \  # TensorBoard 日志保存路径

    --dataloader_num_workers 4                   # 每个 dataloader 使用的 CPU 线程数
```

3. 激活自己的训练环境：
```shell
conda activate fm-9g
```

4. 指定要用的GPU：
```shell
export CUDA_VISIBLE_DEVICES=0,1
```

5. 切换到fm9g_70b目录下，运行训练脚本：
```shell
cd /FM9G4770B/apps/fm9g_70b_hf
bash ./run_huggingface.sh
```

## 多机全量微调
Transformers 全量微调优化程度较低，硬件需求高。建议使用 bmtrain 进行全量微调。 70B 模型文件较大，建议使用 Transformers 进行 qlora 微调。

需要保证机器之间能够通信，且每台机器上的训练环境、代码、数据等一致。

1. 登录主节点，激活训练环境：
```shell
ssh g3006 #登录节点
conda activate fm-9g #激活训练环境
export CUDA_VISIBLE_DEVICES=0,1 #指定要用的GPU
```

3. 修改主节点训练脚本，修改
/FM9G4770B/apps/fm9g_70b_hf/run_huggingface.sh
中机器配置参数，，并将脚本重命名 run_huggingface_0.sh，方便区分：
```shell
    --nproc_per_node 8 \                         # 每节点使用GPU数量
    --nnodes 2 \                                 # 节点数量
    --node_rank 0 \                              # 当前节点编号
    --master_addr g3006 \                    # 主节点地址
    --master_port 6004 \                         # 主节点通信端口
    -m finetune_hgface \                         # 启动 Python 模块 
```

4. 提交主节点训练脚本：
```shell
bash run_huggingface_0.sh
```

5. 启动从节点、激活训练环境，指定要用的卡，方法与主节点一样。

6. 修改从节点训练脚本：将单机多卡的训练脚本重命名为 run_huggingface_1.sh，修改主节点名称、端口、机器数量、GPU数量：
```shell
    --nproc_per_node 8 \                         # 每节点使用GPU数量
    --nnodes 2 \                                 # 节点数量
    --node_rank 1 \                              # 当前节点编号
    --master_addr g3006 \                    # 主节点地址
    --master_port 6004 \                         # 主节点通信端口
    -m finetune_hgface \                         # 启动 Python 模块 
```

7. 提交从节点训练脚本：
```shell
cd /FM9G4770B/apps/fm9g_70b_hf
bash run_huggingface_1.sh
```

8. 如果有三台及以上的机器，重复5-7，注意修改 node_rank 编号
9.  开始训练后，每个iter的loss、lr等信息将在主节点上显示



## 查看训练情况
1. 用tensorboard查看各个loss曲线与学习率等变化情况：
```shell
tensorboard –-logdir /FM9G4770B/apps/fm9g_70b_hf/data/checkpoints/huggingface_70b/test/tblogs #存放.events文件的路径，修改为自己的路径
```


## lora微调

### 安装依赖包

```shell
pip install peft
```

### 微调方式
单机、多机 lora 微调与全量微调脚本和方式一致，修改
/FM9G4770B/apps/fm9g_70b_hf/run_huggingface.sh
脚本中的的相关参数即可，如下所示：
```shell
#! /usr/bin/env bash
use_lora=true
#设置需要注入 LoRA 的层
lora_modules="[\"q_proj\", \"v_proj\", \"k_proj\"]"
#LoRA 的秩（rank），控制注入矩阵的低秩维度，值越大表示模型容量越强，通常 4~64
lora_r=64
# LoRA 的缩放因子（α），影响更新幅度。实际缩放 = α / r，常用于稳定训练
lora_alpha=32
# LoRA dropout，用于训练时正则化，防止过拟合。通常设置为 0 或 0.05
lora_dropout=0.1
```

### lora模型格式转换
模型训练完成后，需将lora merge 到原始模型用于模型推理。
参考脚本
/FM9G4770B/apps/fm9g_70b_hf/merge_hugginface_lora.py
方法如下：
```python

import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer

# 原始基础模型路径（非LoRA模型）
model_path = "/data/public/kqa/9G_4_7_70/70b"
# 训练好的LoRA adapter权重的路径
adapters_path = "/FM9G4770B/apps/fm9g_70b_hf/data/checkpoints/huggingface_70b/test/checkpoint-100"
# 合并后的完整模型输出路径（用于推理时加载）
output_path = "./data/checkpoints/huggingface_70b/merged"
# 加载分词器（tokenizer），用于后续推理时的文本预处理
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 加载原始模型（非LoRA
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map="auto", 
    trust_remote_code=True
)
# 将LoRA adapter 加载到基础模型中
model = PeftModel.from_pretrained(model, adapters_path)
# 合并LoRA权重到原始模型权重中，得到完整的可推理模型
model = model.merge_and_unload()
# 保存合并后的模型和分词器，供推理时直接使用
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
```

## Qlora微调
### 安装依赖包
```shell
pip install bitsandbytes
```
### 微调方式
单机、多机 Qlora 微调与全量微调脚本和方式一致，修改
/FM9G4770B/apps/fm9g_70b_hf/run_huggingface.sh
脚本中的的相关参数即可，如下所示：
```shell
#! /usr/bin/env bash
use_lora=true
qlora=true
#设置需要注入 LoRA 的层
lora_modules="[\"q_proj\", \"v_proj\", \"k_proj\"]"
#LoRA 的秩（rank），控制注入矩阵的低秩维度，值越大表示模型容量越强，通常 4~64
lora_r=64
# LoRA 的缩放因子（α），影响更新幅度。实际缩放 = α / r，常用于稳定训练
lora_alpha=32
# LoRA dropout，用于训练时正则化，防止过拟合。通常设置为 0 或 0.05
lora_dropout=0.1
```
### qlora模型格式转换
模型训练完成后，需将lora merge 到原始模型用于模型推理。参考脚本
/FM9G4770B/apps/fm9g_70b_hf/merge_hugginface_qlora.py
修改一下参数：
```python

#预训练模型路径
model_path = "/data/public/kqa/9G_4_7_70/70b"
#Qlora qlora模型保存路径
adapters_path = "/FM9G4770B/apps/fm9g_70b_hf/data/checkpoints/huggingface_70b/test/checkpoint-50"
#合并后模型保存路径
output_path ="./data/checkpoints/huggingface_70b/q_merged"
```


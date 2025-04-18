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

本文档介绍九格大模型7B版本的bmtrain训练方式。

## 目录

<!-- - [仓库目录结构](#仓库目录结构) -->

- [九格大模型使用文档](#九格大模型使用文档)
  - [目录](#目录)
  - [模型下载](#模型下载)
  - [环境配置](#环境配置)
  - [数据处理流程](#推理脚本示例)
  - [模型转换](#模型转换)
  - [单机训练](#单机训练)
  - [多机训练](#多机训练)
  - [参数详细介绍](#参数详细介绍)
  - [查看训练情况](#查看训练情况)
  - [模型格式转换](#模型格式转换)
  - [全量微调](#全量微调)
  - [lora微调](#lora微调)

## 模型下载

可在
https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/9G7B_MHA.tar
下载7B模型。下载完成后解压tar包，并将其中所有文件放置于**此README文件所在的目录ckpt文件夹**下。

## 环境配置

完成模型下载后，需要安装所需的各项依赖。除本文介绍的7B模型外，九格还有4B和70B两种不同的版本可供选用，4B、7B、70B模型的依赖完全相同，如果已经配置完成其中任意一种，即可跳过此环境配置步骤。环境配置步骤分为Conda环境安装、Pytorch安装、其余依赖项安装三步。

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

如果使用vllm，则需要安装与我们预编译的vllm whl文件匹配的pytorch。可在[此链接处](https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=0a12aa9aa6bc442dff8823ac8b48d991fd0771562eaa38593f9c8196d65f7007)下载对应版本的Pytorch安装包。

#### 安装bmtrain
```shell
pip install bmtrain
```

#### 安装flash_attn
```shell
pip install flash_attn==2.7.0.post2
```
如果使用 pip 直接安装 flash-attn 时遇到错误，建议从[此网站](https://github.com/Dao-AILab/flash-attention/releases)直接下载，请根据你的 Python、PyTorch、CUDA 版本选择对应的 wheel 文件。

使用以下命令确认你的 PyTorch 和 C++ ABI 配置：
```shell
python -c "import torch; print(torch.__version__); print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```
输出示例：
```graphql
2.3.0
True
```
第一行为 PyTorch 版本（例如 2.3.0）
第二行为 C++ ABI 兼容性，可以选择对应的wheel版本。

本文档测试环境为 cuda 12.2、torch 2.3.0、 python 3.10 ,并用如下版本安装：
```shell
pip install flash_attn-2.7.0.post2+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```


#### 安装其他依赖包

```shell
pip install transformers==4.44.0
pip install h5py
pip install matplotlib
pip install tensorboardX
pip install datasets
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


## 模型转换
在训练微调之前，我们需要把hf格式的模型转换为pt格式，方便九格架构读取。
转换脚本为 ./hgface_2_bmtrain.py
``` python

import torch
from transformers import AutoModelForCausalLM

# === 加载模型 ===
model = AutoModelForCausalLM.from_pretrained(
    "./ckpt", 
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
torch.save(new_state_dict, "./9g_models/7b.pt")
print("✅ 已保存为 ./9g_models/7b.pt")

```


## 单机训练
1. 修改./train_configs/7b.json中的训练参数，这一部分的参数设置会覆盖掉shell脚本中的相应部分。
2. 修改FM9G4770B/apps/fm9g_7b/run_bmtrain.sh中最后部分的训练参数，如下所示：
```shell
GPUS_PER_NODE=${args["n_gpus"]} #该节点上需要的GPU数量,可以在./train_configs/7b.json中修改
NNODES=1 #单机训练无需修改这个参数
RANK=0 #单机训练无需修改这个参数
MASTER_ENDPOINT=g3006 #该节点名称
MASTER_PORT=12345 #该节点端口，注意避免端口冲突
```

3. 激活自己的训练环境：
```shell
conda activate fm-9g
```

4. 指定要用的GPU：
```shell
export CUDA_VISIBLE_DEVICES=0,1
```

5. 切换到fm9g_7b目录下，运行训练脚本：
```shell
cd /FM9G4770B/apps/fm9g_7b
bash run_bmtrain.sh
```


## 多机训练
需要保证机器之间能够通信，且每台机器上的训练环境、代码、数据等一致。
1. 修改./train_configs/7b.json中的训练参数，这一部分的参数设置会覆盖掉shell脚本中的相应部分。

2. 登录主节点，激活训练环境：
```shell
ssh g3006 #登录节点
conda activate fm-9g #激活训练环境
export CUDA_VISIBLE_DEVICES=0,1 #指定要用的GPU
```

3. 修改主节点训练脚本，修改FM9G4770B/apps/fm9g_7b/run_bmtrain.sh中最后部分的训练参数，，并将脚本重命名run_bmtrain_node_0.sh，方便区分：
```shell
GPUS_PER_NODE=${args["n_gpus"]} #该节点上需要的GPU数量,可以在./train_configs/7b.json中修改
NNODES=2 #机器数量
RANK=0 #0为主节点，1/2/3…为从节点
MASTER_ENDPOINT=g3006 #该节点名称
MASTER_PORT=12345 #该节点端口，注意避免端口冲突
```

4. 提交主节点训练脚本：
```shell
bash run_bmtrain_node_0.sh
```

5. 启动从节点、激活训练环境，指定要用的卡，方法与主节点一样。

6. 修改从节点训练脚本：将单机多卡的训练脚本重命名为run_bmtrain_node_1.sh，在末尾修改主节点名称、端口、机器数量、GPU数量：
```shell
 GPUS_PER_NODE=${args["n_gpus"]} #该节点上需要的GPU数量,可以在./train_configs/7b.json中修改
  NNODES=2 #机器数量
  RANK=1 #0为主节点，1/2/3…为从节点
  MASTER_ENDPOINT=g3006 #主节点名称
  MASTER_PORT=12345 #主节点端口号，注意避免端口冲突
```

7. 提交从节点训练脚本：
```shell
cd /FM9G4770B/apps/fm9g_7b
bash run_bmtrain_node_1.sh
```

8. 如果有三台及以上的机器，重复5-7，注意修改RANK编号
9.  开始训练后，每个iter的loss、lr等信息将在从节点上显示


## 参数详细介绍
``` python
#是否需要接着预训练模型微调，需要的话写pt格式九格模型路径
args["pretrained_model"]=""

#若训练中断，需要恢复训练，写resume的模型step；注意：恢复训练前后训练配置需要一致
args["resume_ckpt"]=""

#tokenizer路径
args["tokenizer_path"]="./ckpt"

#在dataset_config/目录下，数据集的设置
args["dataset_config"]="data_test"

#训练的名称，模型和log等信息会存储在该文件夹中
args["model_unique"]="7b"

#config位置，在./model_configs/目录中
args["config"]="7b"

#无需更改
args["n_gpus"]=2
args["flash"]="cuda"
args["max_length"]="4096"
args["local"]="False"
args["dataloader"]="indexed"
args["save"]="True"
args["tokenizer_path"]="./tokenizer/tokenizer.model" #  /user/tc_agi/klara/baichuan2/baichuan2.tokenizer.model
args["load_grad"]="False"
args["grad_ckpt_num"]="160"
args["exp_group"]=""
args["ignore_cuda_oom"]="1"
args["tensorboard_all_tasks"]="0"
args["stop_when_end"]="0"
args["only_run_dataloader"]="0"
args["eps"]="1e-6"
args["inspect_iters"]="100"
args["strict_state_dict"]="1"
args["resume_no_optimze"]="0"
args["tp_size"]="1"
args["async_save"]="False"

#训练batch size
args["batch_size"]="1"

#多久存一次
args["save_iters"]="500"

#总的iteration
args["train_iters"]="10000"

#dataloder 的加载线程的设置，如果配置较好，可以适量提高
args["dataloader_num_threads"]=1
args["dataloader_prefetch"]=1
args["dataloader_prefetch_factor"]=1
args["dataloader_num_workers"]=1
args["parallel_load_datastate"]="8"

#学习率
args["lr"]="1e-2"

#warmup的次数
args["warmup_iters"]="20"

#drop的比例
args["drop_iters"]="0.1"

#看是否仅load model
args["only_load_model"]="1"

#学习率下降方法
args["lr_scheduler"]="Cosine"

#可以直接resume训练数据信息
args["load_dataloader_ckpt"]="0"

#drop比例
args["drop_begin"]="-1"
args["drop_rate"]="0.5"
#是use checkpoint，建议使用
args["use_checkpoint"]="0"
```

## 查看训练情况
1. 用tensorboard查看各个loss曲线与学习率等变化情况：
```shell
tensorboard –-logdir /FM9G4770B/apps/fm9g_7b/data/tensorboard/7b #存放.events文件的路径
```
## 模型格式转换
模型训练完成后，需将pt格式模型文件转换为hf模型文件用于模型推理。方法如下：

```shell
#! /usr/bin/env bash 

set -ue

#原始hf模型路径，用于复制配置信息
huggingface_ref_dir=./ckpt
#训练保存的pt格式九格模型
bmtrain_ckpt_path="/FM9G4770B/apps/fm9g_7b/data/checkpoints/7b/10/fm9g_live_checkpoint-10.pt"
# hf格式模型保存路径
huggingface_save_dir="/FM9G4770B/apps/fm9g_7b/9g_models/hf-10"

python bmtrain_2_hgface.py \
    --huggingface-ref-dir $huggingface_ref_dir \
    --bmtrain-ckpt-path $bmtrain_ckpt_path \
    --huggingface-save-dir $huggingface_save_dir

```

## 全量微调
全参数微调训练与原始模型训练方法基本一致，需要额外注意以下几点：
1.数据集类型
训练数据集通常包含大量、多样化的数据，覆盖广泛的主题和语言现象，用于学习广泛的知识和技能。通过无监督学习，训练数据集可能不包含显式标签，模型通过预测下一个词或填补缺失词语来学习模式。
微调数据集更专注于特定的领域或任务，通常是有标签的，并且标签与目标任务直接相关。例如，微调分类模型时，数据集中的每条数据都有对应的分类标签；微调翻译模型时，数据集中包含源语言和目标语言的句子对。
需要根据具体微调任务设计与选择合适的微调数据集。

2.预训练模型的引入
修改训练脚本参数文件：/FM9G4770B/apps/fm9g_7b/run_bmtrain.sh，引入args["load"]参数，里面补充基于微调的预训练模型的路径即可：
```python
#基于微调的预训练模型路径
args["pretrained_model"]="./9g_models/7b.pt"
args["resume_ckpt"]=""
```

3.恢复训练
```python
args["pretrained_model"]=""
args["resume_ckpt"]="5" #写resume的模型step；注意：恢复训练前后训练配置需要一致
```



## lora微调
### 安装依赖包

```shell
pip install opendelta
```
若安装遇到问题，可以下载[github 主页](https://github.com/thunlp/OpenDelta)zip文件解压后本地安装。
```shell
cd OpenDelta
python setup.py install
```


### 单机微调
1. 修改./train_configs/7b_lora.json中的训练参数，这一部分的参数设置会覆盖掉shell脚本中的相应部分。
2. 修改/fm9g_7b/run_bmtrain_lora.sh中最后部分的训练参数，如下所示：
```shell
GPUS_PER_NODE=${args["n_gpus"]} #该节点上需要的GPU数量,可以在./train_configs/7b.json中修改
NNODES=1 #单机训练无需修改这个参数
RANK=0 #单机训练无需修改这个参数
MASTER_ENDPOINT=g3006 #该节点名称
MASTER_PORT=12345 #该节点端口，注意避免端口冲突
```

3. 激活自己的训练环境：
```shell
conda activate fm-9g
```

4. 指定要用的GPU：
```shell
export CUDA_VISIBLE_DEVICES=0,1
```

5. 切换到fm9g_7b目录下，运行训练脚本：
```shell
cd /FM9G4770B/apps/fm9g_7b
bash run_bmtrain_lora.sh
```


### 多机微调
需要保证机器之间能够通信，且每台机器上的训练环境、代码、数据等一致。
1. 修改./train_configs/7b_lora.json中的训练参数，这一部分的参数设置会覆盖掉shell脚本中的相应部分。

2. 登录主节点，激活训练环境：
```shell
ssh g3006 #登录节点
conda activate fm-9g #激活训练环境
export CUDA_VISIBLE_DEVICES=0,1 #指定要用的GPU
```

3. 修改主节点训练脚本，修改FM9G4770B/apps/fm9g_7b/run_bmtrain_lora.sh中最后部分的训练参数，，并将脚本重命名 run_bmtrain_lora_0.sh，方便区分：
```shell
GPUS_PER_NODE=${args["n_gpus"]} #该节点上需要的GPU数量,可以在./train_configs/7b.json中修改
NNODES=2 #机器数量
RANK=0 #0为主节点，1/2/3…为从节点
MASTER_ENDPOINT=g3006 #该节点名称
MASTER_PORT=12345 #该节点端口，注意避免端口冲突
```

4. 提交主节点训练脚本：
```shell
bash run_bmtrain_lora_0.sh
```

5. 启动从节点、激活训练环境，指定要用的卡，方法与主节点一样。

6. 修改从节点训练脚本：将单机多卡的训练脚本重命名为 run_bmtrain_lora_1.sh，在末尾修改主节点名称、端口、机器数量、GPU数量：
```shell
 GPUS_PER_NODE=${args["n_gpus"]} #该节点上需要的GPU数量,可以在./train_configs/7b.json中修改
  NNODES=2 #机器数量
  RANK=1 #0为主节点，1/2/3…为从节点
  MASTER_ENDPOINT=g3006 #主节点名称
  MASTER_PORT=12345 #主节点端口号，注意避免端口冲突
```

7. 提交从节点训练脚本：
```shell
cd /FM9G4770B/apps/fm9g_7b
bash run_bmtrain_lora_1.sh
```

8. 如果有三台及以上的机器，重复5-7，注意修改RANK编号
9.  开始训练后，每个iter的loss、lr等信息将在从节点上显示


### lora 新增参数详细介绍
``` python

#设置需要注入 LoRA 的层
args["lora_layer"]= "[\"project_q\",\"project_k\",\"project_v\",\"w_0\",\"w_1\",\"w_out\"]"
#LoRA 的秩（rank），控制注入矩阵的低秩维度，值越大表示模型容量越强，通常 4~64
args["lora_r"]=16
# LoRA 的缩放因子（α），影响更新幅度。实际缩放 = α / r，常用于稳定训练
args["lora_alpha"]=32
# LoRA dropout，用于训练时正则化，防止过拟合。通常设置为 0 或 0.05
args["lora_dropout"]=0.0

# lora 暂不支持恢复训练
args["resume_ckpt"]=""
```

### lora模型格式转换
模型训练完成后，需将pt格式模型文件转换为hf模型文件用于模型推理。方法如下：
```shell

#! /usr/bin/env bash 

set -ue

#原始hf模型路径，用于复制配置信息
huggingface_ref_dir=./ckpt
#预训练的pt格式九格模型
bmtrain_ckpt_path="/FM9G4770B/apps/fm9g_7b/9g_models/7b.pt"
# hf格式模型保存路径
huggingface_save_dir="/FM9G4770B/apps/fm9g_7b/9g_models/hf-lora-10"
# lora pt 格式模型
lora_ckpt_path="/FM9G4770B/apps/fm9g_7b/data/checkpoints/7b_lora/5/fm9g_live_checkpoint-10.pt"
# lora 训练配置
lora_config_path="/FM9G4770B/apps/fm9g_7b/train_configs/7b_lora.json"

python bmtrain_2_hgface.py \
    --huggingface-ref-dir $huggingface_ref_dir \
    --bmtrain-ckpt-path $bmtrain_ckpt_path \
    --huggingface-save-dir $huggingface_save_dir \
    --lora-ckpt-path $lora_ckpt_path \
    --lora-config-path $lora_config_path
```

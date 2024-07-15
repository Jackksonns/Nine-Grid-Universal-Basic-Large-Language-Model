# 九格大模型使用文档
## 目录
- [环境配置](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md?tab=readme-ov-file#环境配置)
- [开源模型](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md?tab=readme-ov-file#开源模型)
- [数据构建](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md?tab=readme-ov-file#数据构建)
- [模型训练](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md?tab=readme-ov-file#模型训练)
- [模型推理](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md?tab=readme-ov-file#模型推理)
- [多机训练](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_DISTRIBUTED.md)
- [FAQs](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md?tab=readme-ov-file#FAQs)

帮助您快速了解CPM-9G的使用，我们准备了一个快速入门教程，目标是基于CPM-9G基座模型通过指令微调的方式构建一个Chat模型。      
## 环境配置：
[环境配置、算力资源](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ENV.md)

## 开源模型
1 8B的百亿SFT模型，v2版本是在v1基础上精度和对话能力的优化模型，下载链接：
 [8b_sft_model_v1](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/checkpoints-epoch-1.tar.gz), [8b_sft_model_v2](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/sft_8b_v2.zip)

2 端侧2B模型，下载链接：
[2b—sft-model]


## 数据构建

本教程使用的数据是Alpaca Zh，一个开源中文指令微调数据集。[数据集](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json)

### 数据预处理
#### 1. 准备jsonl文件
首先需要将原始数据处理位jsonl形式（每一行一个json），每行包含两个字段，分别是input和output，以下是一个例子：
{"input":"", "output": "我爱北京。"}实际预训练时，数据多为txt格式，可以参照以下过程将txt格式的文件转换为jsonl格式：
``` python
# convert_txt2jsonl.py
import json
import sys

for line in sys.stdin:
    if line.strip() == "":
        continue
    temp_json = {"input": "", "output": line.strip()}
    print(json.dumps(temp_json, ensure_ascii=False))
```
使用方式为：
```js
cat pretrain.txt | python convert_txt2jsonl.py > pretrain.jsonl
```
在本Quick Start教程中，已准备好jsonl数据,路径为raw_data/alpaca_zh.jsonl，示例如下：
```json
{"input": "<用户>保持健康的三个提示。<AI>", "output": "以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。"}
{"input": "<用户>三原色是什么？<AI>", "output": "三原色通常指的是红色、绿色和蓝色（RGB）。它们是通过加色混合原理创建色彩的三种基础颜色。在以发光为基础的显示设备中（如电视、计算机显示器、智能手机和平板电脑显示屏）, 三原色可混合产生大量色彩。其中红色和绿色可以混合生成黄色，红色和蓝色可以混合生成品红色，蓝色和绿色可以混合生成青色。当红色、绿色和蓝色按相等比例混合时，可以产生白色或灰色。\n\n此外，在印刷和绘画中，三原色指的是以颜料为基础的红、黄和蓝颜色（RYB）。这三种颜色用以通过减色混合原理来创建色彩。不过，三原色的具体定义并不唯一，不同的颜色系统可能会采用不同的三原色。"}
```

#### 2. 数据二进制化
为了提升数据读取的效率，方便进行大规模分布式预训练，我们以二进制的形式读取训练数据。因此，在训练开始前，需要将上一步准备好的jsonl格式的数据文件二进制化，需要的代码路径为quick_start/data_binarize.py，使用前需要将环境变量设置为您的本地路径：
```js
sys.path.insert(0, "/data/public/CPM-9G/9G-Train")
```
以下是一个使用示例：
假设当前的数据在raw_data路径下：raw_data/alpaca_zh.jsonl
```js
python data_binarize.py --input [PATH to raw_data] --data_type json --output_path [PATH to raw_data_bin] --output_name data
```
处理完成后，在输出路径（即OUTPUT PATH）下会生成data和meta.bin两个文件，其中data是二进制后的数据文件，meta.bin则记录了这份数据的规模、大小等信息，示例如下：
```js
{"file_name": "data", "block_begin": 0, "block_end": 45, "nbytes": 738321350, "nlines": 4432310, "mask": false, "block_size": 16777216}
```
**请注意，当前的框架需要保证block_end数大于所用的GPU总数。**
例如，用32卡训练时，需满足block_end>32，如果文件较小，可以在二进制化之前对多个小文件进行拼接，以满足大规模训练的需求。

在本Quick Start中，我们为jsonl数据到二进制数据的转换过程准备了脚本：
``` python
for i in {1..10};do
cat raw_data/alpaca_zh.jsonl >> raw_data/alpaca_zh_repeat.jsonl
done
```
``` shell
mkdir raw_data_repeat
mv raw_data/alpaca_zh_repeat.jsonl raw_data_repeat/data.jsonl

python data_binarize.py --input raw_data_repeat --data_type json --output_path bin_data_repeat --output_name data
```
#### 3. 准备数据读取脚本
鉴于不同的预训练数据所包含的字段可能有所差别，我们还兼容了字段转换的环节，如果按照上述标准流程做的数据预处理，那么转换方式将十分简单，代码如下：
```js
# transform_script.py
import random

def rand(n: int, r: random.Random):
    return int(r.random() * n)

def transform(data, num_sample: int, r: random.Random):
    return {"input": data["input"], "output": data["output"]}我们还支持多个数据集的混合读入，并设置不同数据集的比例。为此，需要准备一个数据混合的json文件，来指导训练过程中的数据读取策略，示例如下：
[
    {
        "dataset_name": "alpaca_zh",
        "task_name": "alpaca_zh",
        "weight": 1.0,
        "path": "/data/public/CPM-9G/quick_start/bin_data_repeat",
        "incontext_weight": [
            1.0
        ],
        "transforms": "/data/public/CPM-9G/quick_start/transform_data.py"
    }
]
```
该文件中各字段的解释如下：
- dataset_name:数据集名称;
- task_name:数据集所属任务，task_name+dataset_name 将作为训练过程中识别数据集的标签，task_name 则可用于训练过程中针对任务分别汇总 loss 信息、token 吞吐量等;
- weight:浮点数，采样权重;(注意此权重仅代表每个数据集的样本数配比，实际 token 吞吐量的配比还与每个样本的平均 token数量有关)
- path:meta.bin、二进制数据的父目录，即前文所述的 raw_data_bin;
- transforms:数据转换脚本对应的路径;
- incontext_weight: 训练样本叠加方式，[1.0] 表示 100% 的概率采样一个样本，[0.8, 0.2] 表示 80% 的概率采样一个样本, 20% 概率采样两个样本进行拼接，[0.75, 0.1, 0.15] 表示 15% 概率采样三个样本、 10% 的概率采样两个样本进行拼接、75% 采样一个样本;
- 数据集的配比(即 weight 参数)需要重点调整，对于模型的训练稳定性和最终在各类数据上的能力表现有重大影响；
- 我们在此文件中指定了数据文件的路径、转换脚本路径等信息，后续训练仅需要系统该文件的路径即可。

## 模型训练
模型训练列举了三种训练
- [pretrain训练](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md?tab=readme-ov-file#pretrain训练)  
- [SFT全参数微调训练](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md?tab=readme-ov-file#SFT全参数微调训练)  
- [LoRA微调训练](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_LORA.md)  

### pretrain训练：
模型训练代码的位置：9G-Train/apps/cpm9g/pretrain_cpm9g.py
需要将代码中环境变量设置为您的代码路径：
``` python
#9G-Train/apps/cpm9g/pretrain_cpm9g.py:17
sys.path.insert(0, "/data/public/CPM-9G/9G-Train")
```

```shell
#! /bin/bash

# use 8 GPU for example, pretrain may need 32 GPU
export MASTER_ADDR=`hostname`
export MASTER_PORT=12345

EXP_PATH=. # 修改为您的实验路径，用于存储训练日志和模型
CODE_PATH=/data/public/CPM-9G/9G-Train # 修改为您的代码路径
DATA_PATH=/data/public/CPM-9G/quick_start/datasets.json # 修改为您的datasets.json路径
CHECKPOINT=/data/public/CPM-9G/models/model.pt # 修改为您的基座模型路径

mkdir -p ${EXP_PATH}/logs/debug
mkdir -p ${EXP_PATH}/logs/tensorboard/cpm9g/
CONFIG_NAME="${CODE_PATH}/apps/cpm9g/config/"
# --------------- 运行参数 ---------------
OPTS=""
OPTS+=" --model-config ${CONFIG_NAME}/config.json"
OPTS+=" --vocab ${CONFIG_NAME}/vocab.txt"
OPTS+=" --batch-size 12"
OPTS+=" --train-iters 2000" # 训练步数，达到此步数后，学习率降到最小值
OPTS+=" --save-iters 100" # 存储步数，每隔此步数，存储一个模型文件
OPTS+=" --save-name cpm9g_checkpoint" # 模型名称前缀
OPTS+=" --max-length 4096" # 最多token数量
OPTS+=" --lr 1.5e-5" # 峰值学习率
OPTS+=" --inspect-iters 100" # 检查步数，每隔此步数，输出一次模型梯度的详细信息
OPTS+=" --warmup-iters 50" # 热启动步数
OPTS+=" --lr-decay-style noam" # 学习率变化策略
OPTS+=" --weight-decay 0.1" # 正则化参数
OPTS+=" --clip-grad 1.0" # 正则化参数
OPTS+=" --loss-scale 1048576" # 和训练稳定性相关，一般情况下不需修改
OPTS+=" --loss-scale-steps 32" # 和训练稳定性相关，一般情况下不需修改
OPTS+=" --offload" # 使用cpu offload将优化器参数转移到cpu，一般情况下无需修改
OPTS+=" --flash cuda"

# --------------- 写文件路径 ---------------
OPTS+=" --save ${EXP_PATH}/checkpoints/cpm9g/"
OPTS+=" --save-model ${EXP_PATH}/models/cpm9g/"

OPTS+=" --log-dir ${EXP_PATH}/logs/train/"
OPTS+=" --tensorboard ${EXP_PATH}/tensorboard/cpm9g/"`date +"%Y%m%d%H%M%S"`

# --------------- 读文件路径 ---------------
OPTS+=" --dataset ${DATA_PATH}"
OPTS+=" --load ${CHECKPOINT}"
OPTS+=" --start-step 1"

# --------------- 透传参数 ---------------
OPTS+=" $@"

# --------------- 最终指令 ---------------
CMD="torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CODE_PATH}/apps/cpm9g/pretrain_cpm9g.py ${OPTS}"
echo "${CMD}"
$CMD
```

### SFT全参数微调训练
``` shell
export MASTER_ADDR=`hostname`
export MASTER_PORT=12345

CPM_PATH="/data/groups/QY_LLM_Core/arq_project/code/9G-Train"
CKPT_NAME="/data/public/anrongqiao/models/"
EXP_PATH=./exp/8b/

mkdir -p $EXP_PATH
MODEL_NAME="cpm9g-8b-sft"

OPTS=""
OPTS+=" --model-config ${CKPT_NAME}/config.json"
OPTS+=" --vocab ${CKPT_NAME}/vocab.txt"

OPTS+=" --train-iters 10000" # 训练步数，达到此步数后，学习率降到最小值
OPTS+=" --inspect-iters 200" # 存储步数，每隔此步数，存储一个模型文件
OPTS+=" --warmup-iters 20" # 热启动步数

OPTS+=" --lr-decay-style cosine" # 学习率变化策略
OPTS+=" --weight-decay 0.1" # 正则化参数
OPTS+=" --clip-grad 1.0" # 正则化参数
OPTS+=" --loss-scale 1048576" # 和训练稳定性相关，一般情况下不需修改
OPTS+=" --max-loss-scale 33554432" #和训练稳定性相关，一般情况下不需修改
OPTS+=" --min-loss-scale 1" #和训练稳定性相关，一般情况下不需修改
OPTS+=" --loss-scale-steps 32" # 和训练稳定性相关，一般情况下不需修改

OPTS+=" --offload" # 使用cpu offload将优化器参数转移到cpu，一般情况下无需修改
OPTS+=" --batch-size 1"
OPTS+=" --max-length 4096" #上下文长度
OPTS+=" --lr 1e-5" #学习率
OPTS+=" --start-step 0" #初始steps
OPTS+=" --epoch 1" # 训练多少个epoch

OPTS+=" --load ${CKPT_NAME}/model.pt" # 修改成自己的预训练模型
OPTS+=" --dataset ../dataset/qy_sft_20230129_bin/" # 和pretrain脚本不同，sft数据量少，直接输入bin文件即可
OPTS+=" --save ${EXP_PATH}/checkpoints" # 模型存储
OPTS+=" --save-name ${MODEL_NAME}" #待存储模型的前缀
OPTS+=" --tensorboard /data/logs/tensorboard/${MODEL_NAME}/${CUR_DATE}/" #
OPTS+=" --gradient-accumulation-steps 4" # 梯度累积更新步数
OPTS+=" $@"

#运行指令
CMD="torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/cpm9g/sft_cpm9g.py ${OPTS}"
echo "${CMD}"
$CMD
```

## 模型推理
```python
import os

from libcpm import CPM9G

import argparse, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, help="the path of ckpt")
    parser.add_argument("--config", type=str, help="the path of config file")
    parser.add_argument("--vocab", type=str, help="the path of vocab file")
    args = parser.parse_args()

    model_config = json.load(open(args.config, 'r'))
    model_config["new_vocab"] = True

    model = CPM9G(
        "",
        args.vocab,
        -1,
        memory_limit = 30 << 30,#memory limit 左边的参数根据gpu的显存设置，如果是A100，可以设置为 72 << 30，这样的话就可以用到更多的显存
        model_config=model_config,
        load_model=False,
    )
    model.load_model_pt(args.pt)

    datas = [
        '''<用户>马化腾是谁？<AI>''',
        '''<用户>你是谁？<AI>''',
        '''<用户>我要参加一个高性能会议，请帮我写一个致辞。<AI>''',
    ]

    for data in datas:
        res = model.inference(data, max_length=4096)
        print(res['result'])

if __name__ == "__main__":
    main()
```

## FAQs

常见问题汇总，持续补充ing

### 训练相关
1 推荐大家使用docker，避免大家在conda 环境安装时候遇到的问题
2 pretrain训练的脚本和sft训练的脚本基本类似，在apps/cpm_9g目录下
3 尽量避免在window机器下修改脚本，window中的编码和格式linux是有差别的，容易在脚本执行中报错
4 SFT如何调参训练
  ```
  回答：如果数据量少于10w条，全参数微调的时候多训练几个epoch，把学习率调低一些，比如说5e-6等；更建议使用lora 微调的方式
      数据量很多呢，比如说达到百万级别，那可以选择全参数微调，但训练最多2个epoch足够，注意过拟合的问题
  ```
5 微调训练中，train_iters如何计算？
  ```
  回答：因为模型上下文是4096的token数目，通常情况存在训练数据不足4096的长度，所以会对多条数据进行merge，因此送入模型条数要少于实际的数据条数
  ```
6 打印出来的Iter信息有缺失
  ```
  回答：debug下看看是否是出现drop_last的情况
  ```
7 现有代码是否需要验证集合？
  ```
  回答：不需要，参数中出现的val_datasets忽略即可
  ```
8 加载模型遇到：invalid header or archive is carrupted，这种一般是模型没有下载完导致的，目前红山上的模型确定是完整的，首先自查自己的模型是否下载成功。
9 存储模型的时候遇到failed write file data ，一般先检查下文件路径和权限、磁盘空间吧，存储模型基本不会报错
10 是否支持图像模态：
```
  回答：不支持图像模态，仅支持文本模态
```
### 数据相关
1 历史对话的传入：
``` json
datas = [
    '''<用户>问题1<AI>答案1<用户>问题2<AI>答案2<用户>问题2<AI>'''
    ]
```


## TODO
1 发布8B-32k上下文的模型
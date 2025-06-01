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

本文档介绍九格大模型4B版本的推理方式。本仓库支持三种推理方式：

1. 使用原生huggingface transformer的generate函数进行推理；
2. 使用性能更好的vllm框架进行推理；
3. 使用vllm将模型部署为服务，可使用OpenAI API发送请求来进行推理。

推理代码示例可参见后文说明，也可在[代码仓库中 inference_samples/4b 目录](../inference_samples/4b)中找到。完成模型下载并依照步骤安装所需的各项依赖后，即可使用全部的三种推理方式。

## 目录

<!-- - [仓库目录结构](#仓库目录结构) -->

- [九格大模型使用文档](#九格大模型使用文档)
  - [目录](#目录)
  - [模型下载](#模型下载)
  - [环境配置](#环境配置)
  - [推理脚本示例](#推理脚本示例)

## 模型下载

可在 https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/9G4B.tar 下载4B模型。

## 环境配置

完成模型下载后，需要安装所需的各项依赖。除本文介绍的4B模型外，九格还有7B和70B两种不同的版本可供选用，4B、7B、70B模型的依赖完全相同，如果已经配置完成其中任意一种，即可跳过此环境配置步骤。环境配置步骤分为Conda环境安装、Pytorch安装、其余依赖项安装三步。

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
conda install pytorch==2.3.0
```

如果使用vllm，则需要安装与我们预编译的vllm whl文件匹配的pytorch。可在[此链接处](https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=0a12aa9aa6bc442dff8823ac8b48d991fd0771562eaa38593f9c8196d65f7007)下载对应版本的Pytorch安装包。

#### 安装其他依赖包

```shell
pip install transformers==4.44.0
pip install datamodel-code-generator
pip install accelerate
pip install jsonschema
pip install pytrie
pip install sentencepiece
pip install protobuf
```

#### 安装vllm依赖

使用vllm进行推理需要使用我们预编译的vllm whl安装包。此安装包在CUDA12.2、python3.10环境下编译，适配预编译版本为cuda 12.x的torch 2.3.0，可安装后执行推理。安装包可在 [https://drive.weixin.qq.com/s?k=AKIAqQfNADgsGqHMan](https://drive.weixin.qq.com/s?k=AKIAqQfNADgsGqHMan) 下载。

请注意此版本的Vllm安装包适用于FM 9G 4B、7B和70B模型，不适用于FM9G2B、8B模型，如果需要推理FM9G2B、8B模型，请参考[此说明文档](../quick_start_clean/readmes/quick_start.md)下载对应版本的Vllm whl安装包。

```shell
cd ..
pip install vllm-0.5.0.dev0+cu122-cp310-cp310-linux_x86_64.whl
```

## 推理脚本示例

### transformers原生代码推理脚本示例

此代码适用于4B模型单卡推理。在指定路径时，需指定pytorch_model.bin文件**所在目录**的路径，注意不是pytorch_model.bin文件本身的路径。

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    model_path = "XXXX"  # 请替换为你的pytorch_model.bin文件所在的目录的路径
    prompt = "山东最高的山是？"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    model.to(device)
    model.eval()

    prompt = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)
    with torch.no_grad():
        res = model.generate(**inputs, max_new_tokens=256)
    responses = tokenizer.decode(res[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    ai_answer = responses.strip()
    print(ai_answer)
```

### vllm离线批量推理脚本示例

此脚本适用于4B模型vllm离线推理。同样，在指定路径时，需指定pytorch_model.bin文件**所在目录**的路径。

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

if __name__ == '__main__':

    # 提示用例，定义了一个包含多个问题的列表，这些问题将被用于生成回答
    prompts = [
        "你是谁？",
        "山东最高的山是？",
        "介绍一下大模型的旋转位置编码。",
    ]

    # 模型路径，指定了模型文件所在的目录路径
    model_path = "XXXX"

    # 设置采样参数以控制生成文本，更多参数详细介绍可见/vllm/sampling_params.py
    # temperature越大，生成结果的随机性越强，top_p过滤掉生成词汇表中概率低于给定阈值的词汇，控制随机性
    # max_tokens表示生成文本的最大长度
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

    # tensor_parallel_size是模型张量并行的GPU数量，用于加速模型的计算.
    # 4B模型可在单块A100 40G上推理；对于显存较小的显卡，可考虑使用多块GPU并行推理
    tensor_parallel_size = 1
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, tokenizer_mode='auto')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 初始化一个空列表，用于存储对话
    conversations = []
    for prompt in prompts:
        # 使用分词器的apply_chat_template方法将提示用例转换为对话格式
        conversations.append(
            tokenizer.apply_chat_template(conversation=[{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        )

    # 根据提示生成文本，将对话列表和采样参数传递给LLM的generate方法
    outputs = llm.generate(conversations, sampling_params)

    # 打印输出结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### 部署OpenAI API服务推理

vLLM可以为 LLM 服务进行部署，这里提供了一个示例：

#### 启动服务：

```shell
#! /usr/bin/env bash

set -ue

python -m vllm.entrypoints.openai.api_server \
    --model modelpath \
    --tokenizer-mode auto \
    --dtype auto \
    --trust-remote-code \
    --served-model-name 9g \
    --api-key fm9g \
    --gpu-memory-utilization 0.9 \
    --port 8020  \
    --tensor-parallel-size 1
    # tensor_parallel_size是模型张量并行的GPU数量，用于加速模型的计算；
    # 4B模型可在单块A100 40G上推理；对于显存较小的显卡，可考虑使用多块GPU并行推理
```

执行对应指令后，在http://localhost:8020 地址上启动服务，启动成功后终端会出现如下提示：

```shell
INFO:     Started server process [3511795]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8020 (Press CTRL+C to quit)
```

#### 调用推理API：

启动服务端成功后，重新打开一个终端，可参考执行以下python脚本：

```python
from openai import OpenAI

client = OpenAI(
    api_key="fm9g",
    base_url="http://localhost:8020/v1",
)
messages = [{"role": "user", "content": "介绍一下大语言模型的旋转位置编码"}]

response = client.chat.completions.create(
    model="9g",
    messages=messages,
    stream=True, # 流式输出
    # 其他可选推理参数
    # max_tokens=200,
    # n = 1,
    # stream = False,
    # frequency_penalty = 0.8,
    # presence_penalty = 0.9,
    # logit_bias = {}
)

for chunk in response:
    try: 
        content = chunk.choices[0].delta.content
    except:
        content = None
    if content is not None:
        print(content, end="")
print() 
```

#### 调用多轮对话API：

启动服务端成功后，重新打开一个终端，可参考执行以下python脚本：

```python
from openai import OpenAI

client = OpenAI(
    api_key="fm9g",
    base_url="http://localhost:8020/v1",
)
messages = []

while True:
    print("开始对话（输入 'quit' 结束）：")
    user_input = input("请输入内容：")
    if user_input.strip().lower() == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="9g",
        messages=messages,
        # 其他生成的超参数，看需求来加
        stream=True,
        # max_tokens=200,
        # n = 1,
        # frequency_penalty = 0.8,
        # presence_penalty = 0.9,
        # logit_bias = {}
    )

    full_reply = ""
    for chunk in response:
        try: 
            content = chunk.choices[0].delta.content
        except:
            content = None
        if content is not None:
            print(content, end="")
            full_reply += content
    print() 

    messages.append({"role": "assistant", "content": full_reply})
```
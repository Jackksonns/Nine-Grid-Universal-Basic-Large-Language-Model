# 九格通用基础大模型

## 简介  
启元九格大模型由启元实验室牵头，联合清华大学、哈尔滨工业大学、中国科学院计算技术研究所、北京大学、南开大学等顶尖科研单位共同研发。该模型具备 **高效训练与推理**、**高效适配与部署** 的技术特点，支持多种 **自然语言处理（NLP）** 和 **多模态** 任务，包括 **文本问答、文本分类、机器翻译、文本摘要、图文理解等**。

---

## 更新信息  

### 🔥 **最新发布（2025.04.18）**：

- **模型**：开源了**4B、7B、70B**三种不同尺寸的基础语言模型，能力再上台阶。
- **训练**：4B、7B、70B模型的**全参数微调和Lora微调**训练代码已经开源。
- **推理**：支持原生**Huggingface Transformers**推理和**vllm**快速推理，环境配置方法和离线批量推理/在线多轮对话的示例代码均已开源。

    模型下载链接和推理、微调训练的说明文档和对应示例代码可在下表链接中找到，对应推理环境和微调环境的配置方法也已放置在文档中：

    |模型|模型下载链接|推理说明及示例代码|微调训练说明及示例代码|
    |----|-------|----------------|---|
    |4B|[🔗点击此处下载](https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/9G4B.tar)|[📄点击此处阅读](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/quick_start_4_7_70/inference_4b.md)|[🚆BMTrain微调](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/FM_9G/apps/fm9g_4b/README.md) / [🤗Transformers微调](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/FM_9G/apps/fm9g_4b_hf/README.md)|
    |7B|[🔗点击此处下载](https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/9G7B_MHA.tar)|[📄点击此处阅读](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/quick_start_4_7_70/inference_7b.md)|[🚆BMTrain微调](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/FM_9G/apps/fm9g_7b/README.md) / [🤗Transformers微调](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/FM_9G/apps/fm9g_7b_hf/README.md)|
    |70B|[🔗点击此处下载](https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/FM9G_70B_SFT_MHA.tar)|[📄点击此处阅读](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/quick_start_4_7_70/inference_70b.md)|[🚆BMTrain微调](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/FM_9G/apps/fm9g_70b/README.md) / [🤗Transformers微调](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/FM_9G/apps/fm9g_70b_hf/README.md)|


### 🚀 **历史版本更新**  

#### **2025.02.25**：[**FM9G-4B**](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM9G-V/quick_start_clean/readmes/quick_start.md)

- **模型**：[**4B 模型**](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/9G4B.tar)将上下文长度扩展至32k支持长文推理，并采用了GQA结构以降低KV cache的显存需求。
- **训练**：训练代码将即将开源。（2025.04.18更新：训练代码已开源）
- **推理**：推理代码可以直接复用2.4B模型。增加了[2.4B模型](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/gguf/fm9g-2b-q4_k_m.gguf)和[4B模型](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/gguf/fm9g-4b-q4_k_m.gguf)的q4_k_m量化版gguf模型。

> 🚨 **FM9G 4B模型** 是一款专为长上下文处理设计的模型，支持32k+的文本窗口，并通过搭载[MapReduce](https://github.com/thunlp/LLMxMapReduce/tree/main)能力，能够处理超过100k的上下文。该模型还支持工具调用和代码解释，具备卓越的数学推理能力和中英文指令执行能力，能够高效处理复杂任务。


#### **2025.01.12**：[**FM9G-V**](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM9G-V/quick_start_clean/readmes/quick_start.md)
- **模型**：**13B 多模态基础大模型**，支持 **单图文推理**。
- **训练**：开源了 **多模态基础大模型** 的训练代码。
- **推理**：支持 **单图文推理**，提升了图文理解和生成能力。

#### **2024.08.19**：[**FM9G**](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/quick_start_clean/readmes/quick_start.md)
- **2B 模型**：经过多数据集验证，发现 **LoRA训练效果不及全参数微调**，因此 **2B模型采用了全参数微调**，效果显著提升。
- **8B 模型**：LoRA微调仍在 **master 分支** 进行，正在进行更细致的优化。
- **QUICK START**：更新了 **2B 全参数微调** 的详细信息，帮助用户更好地理解和应用该模型。
---

### 📚 其他信息
- 若仍在使用旧版本的九格模型训练和推理，请切换分支至 [master](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md) 分支。

---

### 📌 开源模型参数级别

| 模型            | 主要能力              | 参数规模 | 代码分支  |
|---------------|-------------------|--------|--------|
| **FM9G-8B**  | **文本处理（NLP）**   | 80 亿  | FM9G  |
| **FM9G-2B**  | **文本处理（NLP）**   | 20 亿  | FM9G   |
| **FM9G-V（13B）** | **多模态（文本+图像）** | 130 亿 | FM9G-V   |
| **FM9G-4B**  | **文本处理（NLP）**   | 40 亿  | FM9G   |

                                                   
# 迈向通用智能的大模型技术系列课程
系列课程全方位介绍人工智能和大模型技术的基础知识和前沿课题，理论学习和实践应用相结合。课程既有“人工智能与大模型通论”和“神经网络与预训练模型”等基础知识，也有“九格大模型生态体系”和“领域大模型实战”等实战主题，基本内容包括大模型训练、微调、知识增强、伦理安全、多模态、具身智能、自主智能体等话题，高级选题包括多语言处理、面向科学研究的大模型应用、高效计算技术、评测与数据科学等话题。课程旨在通过一系列精心设计的单元为学习者提供大型通用人工智能的学习之旅。

## 人工智能大模型通论
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%80%9A%E8%AE%BA-%E5%AD%99%E8%8C%82%E6%9D%BE%E8%80%81%E5%B8%88-1124_DeWatermark.mp4
" width="800px" height="600px" controls="controls"></video>
[人工智能与大模型通论-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/1.%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%80%9A%E8%AE%BA-PPT.pdf)
                                                      
## 大模型技术的重要特性与发展趋势
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8A%80%E6%9C%AF%E7%9A%84%E9%87%8D%E8%A6%81%E7%89%B9%E6%80%A7%E4%B8%8E%E5%8F%91%E5%B1%95%E8%B6%8B%E5%8A%BF-%E5%88%98%E7%9F%A5%E8%BF%9C%E8%80%81%E5%B8%88-1201_DeWatermark.mp4
" width="800px" height="600px" controls="controls"></video>
[大模型技术的重要特性与发展趋势-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/2.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8A%80%E6%9C%AF%E7%9A%84%E9%87%8D%E8%A6%81%E7%89%B9%E6%80%A7%E4%B8%8E%E5%8F%91%E5%B1%95%E8%B6%8B%E5%8A%BF-PPT.pdf)
                 
## 大语言模型的适配与对齐技术
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/2023-12-22-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%80%82%E9%85%8D%E4%B8%8E%E5%AF%B9%E9%BD%90%E6%8A%80%E6%9C%AF-%E4%B8%81%E5%AE%81_DeWatermark.mp4
" width="800px" height="600px" controls="controls"></video>
[大语言模型的适配与对齐技术-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/3.%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%80%82%E9%85%8D%E4%B8%8E%E5%AF%B9%E9%BD%90%E6%8A%80%E6%9C%AF-PPT.pdf)
                  
## 大模型领域适配原理与实战
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/2023-12-29%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A2%86%E5%9F%9F%E9%80%82%E9%85%8D%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E6%88%98-%E7%8E%8B%E7%A1%95_DeWatermark.mp4
" width="800px" height="600px" controls="controls"></video>
[大模型领域适配原理与实战-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/4.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A2%86%E5%9F%9F%E9%80%82%E9%85%8D%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E6%88%98-PPT.pdf)
                
## 知识增强的大语言模型
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%9A%84%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.mp4
" width="800px" height="600px" controls="controls"></video>
[知识增强的大语言模型-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/5.%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%9A%84%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-PPT.pdf)
                   
## 大模型工具学习
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B7%A5%E5%85%B7%E5%AD%A6%E4%B9%A0.mp4
" width="800px" height="600px" controls="controls"></video>
[大模型工具学习-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/6.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B7%A5%E5%85%B7%E5%AD%A6%E4%B9%A0-PPT.pdf)
                 
## 检索增强生成的基本实现
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%AE%9E%E7%8E%B0.mp4
" width="800px" height="600px" controls="controls"></video>
[检索增强生成的基本实现-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/7.%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%AE%9E%E7%8E%B0-PPT.pdf)
              
## 多模态语义检索与检索增强技术
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%AD%E4%B9%89%E6%A3%80%E7%B4%A2%E4%B8%8E%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E6%8A%80%E6%9C%AF.mp4
" width="800px" height="600px" controls="controls"></video>
[多模态语义检索与检索增强技术-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/8.%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%AD%E4%B9%89%E6%A3%80%E7%B4%A2%E4%B8%8E%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E6%8A%80%E6%9C%AF-PPT.pdf)
              
## 大语言模型驱动的多智能体协作与演化
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/400_0121.mp4
" width="800px" height="600px" controls="controls"></video>
[大语言模型驱动的多智能体协作与演化-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/9.%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E9%A9%B1%E5%8A%A8%E7%9A%84%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E5%8D%8F%E4%BD%9C%E4%B8%8E%E6%BC%94%E5%8C%96-PPT.pdf)
                        
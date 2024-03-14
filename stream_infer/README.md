# demo使用方法

# 环境安装
docker 路径：

由于流式输出需要特定的环境依赖，因此在新的env下进行推理和输出
conda activate stream_info

# 流程：
1 将模型进行convert处理，将训练模型转换成流式输出支持的格式
   python convert.py

2 模型推理: python deploy_llm_8b_demo.py
   (1) 设置CUDA_VISIBLE_DEVICES的数目
   (2) 修改LocalLoader 中的实际使用模型的属性
   (3) 在修改LocalLoader调用的时候，修改流式输出模型位置及其词表

3 测试请求：python request_demo.py
  若不清楚请求的ip port，可以在推理阶段保存的log文件（error_8b.log）中找到



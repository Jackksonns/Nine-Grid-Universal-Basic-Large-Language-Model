# demo使用方法

# 环境安装
基于方便，采用了镜像形式进行使用
docker 路径：https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/cpmlive-flash-0.0.5.tar
docker的使用：见QuickStart手册.pdf

启动docker之后，在命令行中执行conda activate stream_info，后续的处理都在stream_info环境下进行执行

# 流程：
1 将模型进行convert处理，将训练模型转换成流式输出支持的格式
   (1) 需要修改的主要参数如下：

    输入已有的源模型 src_model_path = "./checkpoints-epoch-1/cpm9g-8b-sft-epoch-1.pt"
    格式转换后的模型地址 dst_model_path = "model_8b.ckpt" 
    模型的layers数目，需要根据提供的config文件中的layer字段确定，
        在百亿模型中layer_num = 32
        在千亿模型中layer_num = 80

   (2)执行python convert.py得到的dst model即为后续用的模型

2 模型推理: 
   (1) 设置CUDA_VISIBLE_DEVICES的数目
   
   (2) 修改LocalLoader 类中模型的属性，将下面这6个函数依据提供的模型config文件中的字段进行修改
   
    def num_layers(self):
    def dim_model(self):
    def num_heads(self):
    def num_kv_heads(self):
    def dim_head(self):
    def dim_ff(self):
    
   (3) 在修改LocalLoader类别调用的时候 将上一步生成的模型文件和词表位置输入，memory_limit无需修改
     model = libcpm.CPMCaterpillar(
    LocalLoader(
        "model_8b.ckpt",
        "vocabs.txt",
    )
   (4) 执行python deploy_llm_8b_demo.py

3 测试请求：
  (1) 修改url，本机请求的话是localhost 或者127.0.0.1，url是该及其的ip
     
  (2) 可以修改payload中的content内容，举例：
     payload = json.dumps({
      "content": "<用户>好久不见！<AI>"})
     payload = json.dumps({
      "content": "<用户>你是谁？<AI>"})

  (3) 执行python request_demo.py
  (4) 最终结果如以下形式，即正确的请求结果
     https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/stream_infer/result.png



# Docker使用
我们提供可以运行模型训练和推理的docker，便于在新环境下快速使用九格大模型。您也可以使用Conda配置运行环境。Conda配置方式请见下一节。
#### [docker 路径](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/cpmlive-flash-0.0.4.tar)
## 镜像加载
### rootless 启动
允许用户在不影响主机系统的情况下运行应用程序和服务，并且可以轻松地共享和分发环境
```shell
srun -p gpu1 --nodelist=g2001 -N 1 -n 8 -c 8 --gres=gpu:8 --pty bash
module load rootless-docker/default 
```

**注意使用bash（不能用zsh）**
start_rootless_docker.sh运行成功的话，此时执行docker ps可以看到当前没有正在运行的容器，如果有正在运行的容器，说明rootless模式没有启动成功，请联系管理员。

### 加载镜像
```shell
docker load -i cpmlive-flash-0.0.4.tar
docker tag [IMAGE_ID] cpmlive-flash:0.0.4
```

如果加载镜像遇到：archive/tar invailid tar header的问题，是因为docker下载不全，check下docker下载结果。以红山上上传的docker为准

### 启动容器
```
docker run -it -d -v [HOST_PATH1]:[DOCKER_PATH1] -v [HOST_PATH2]:[DOCKER_PATH2] --gpus all --shm-size=4g --sh cpmlive-flash:0.0.4 bash
```
如果有docker权限、且rootless执行错误的情况下，可以尝试下非rootless启动

## 非rootless 启动
### 启动容器
```
docker run -it -d -v [HOST_PATH1]:[DOCKER_PATH1] -v [HOST_PATH2]:[DOCKER_PATH2] --gpus all --network host --shm-size=4g cpmlive-flash:0.0.4 bash
```

参数解释如下：
- -v [HOST_PATH1]:[DOCKER_PATH1]: 这个选项用于将主机（宿主机）文件系统中的目录或文件挂载到容器中的目录。[HOST_PATH1] 是主机路径，[DOCKER_PATH1] 是容器中对应的路径；
- --gpus all: 这个选项用于在容器中启用 GPU 支持，并将所有可用的 GPU 分配给容器。需要在 Docker 守护程序中启用 NVIDIA Container Toolkit 才能使用此选项；
- --network host: 这个选项用于让容器共享主机网络命名空间，使容器可以直接访问主机上的网络接口和服务；
- --shm-size 容器的share memory，根据主机的情况设置，如果训练推理需要的内存比较多，可以增大share memory值；
### 进入容器
```shell
docker exec -it [CONTAINER_ID] bash
```
### 退出容器
```shell
Ctrl+d
```
### 删除容器
```shell
docker stop [CONTAINER_ID]
```
### 查看正在运行容器
```shell
docker ps
```
### 环境安装
```shell
pip install tensorboardX
```

## Conda环境配置
### 训练环境配置
```shell
1. 使用python 3.8.10创建conda环境
conda create -n cpm-9g python=3.8.10 

2.安装Pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia  

3. 安装BMTrain
pip install bmtrain==0.2.3.post2  

4. 安装flash-attn
pip install flash-attn==2.0.8  

5. 安装其他依赖包
pip install einops
pip install pytrie
```

如果需要自己配置conda的训练，供参考的配置：
驱动版本：Driver Version: 470.57.02
cuda：11.4-11.6之间都可以

### 推理环境配置
```js
1. 安装nvidia-nccl
pip install nvidia-nccl-cu11==2.19.3   

2. 配置环境变量
nccl_root=`python -c "import nvidia.nccl;import os; print(os.path.dirname(nvidia.nccl.__file__))"`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$nccl_root/lib
echo $LD_LIBRARY_PATH2. 安装LibCPM
pip installlibcpm-1.0.0-cp38-cp38-linux_x86_64.whl
```

# 算力资源
## 推荐配置：
### 千亿大模型
   - 预训练、全参数微调：8 * 512G以上内存，64 * 80G以上显存
   - 高效微调（LoRA）与推理: 512G 以上内存，8 * 80G以上显存

### 百亿大模型
   - 预训练、全参数微调：2 * 512G以上内存，16 * 80G以上显存
   - 高效微调（LoRA）与推理: 128G 以上内存，2 * 80G以上显存

## 极限配置
最极限的资源配置，仅供参考，在大模型训练中其实并不推荐，因为其效果一般不佳，训练时长也比较久

| 模型        | 资源   |  最小算力  | 
| :--------  | :-----  | :----:  |
| 百亿模型 |内存 |训练:140G, 推理:1G|
| 百亿模型 |显存 |训练:49G, 推理:20G|
| 千亿模型 |内存 |训练: 200G, 推理:2G|
| 千亿模型 |显存 |训练: 8*80G , 推理:4 * 50G|

另外
- 该表格是百亿、千亿模型需要的最小的资源，batch size为1.
- 百亿模型是在单卡A100上测试
- 千亿的训练是用8卡A100，但是训到过程中out of memory，所以建议至少用2台A100或者至少两台
- 千亿的推理是用4卡A100训练

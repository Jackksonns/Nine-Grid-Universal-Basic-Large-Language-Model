# 分布式多机训练


-  首先保证机器之间能够通信
-  每台机器上的训练环境、代码、数据等一致

## 简单模式
这种方式只适用于机器很少的提交方法，比如说两台机器debug调试的时候可以如下操作
以sft_cpm9g_8b.sh举例
```shell
# 这儿指定主节点的IP值
export MASTER_ADDR=g3002

#中间省略各种参数配置

#--nnodes 指定用几台机器，提交任务后主节点会一直等待通信满足4台机器，直到time out
#--nproc_per_node 每张机器多少张卡
CMD="torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${CPM_PATH}/apps/cpm9g/sft_cpm9g.py ${OPTS}"
```
接下来，在这两个机器中都执行bash sft_cpm9g_8b.sh，这样就完成一次最简单的多机训练
不过机器多了之后不推荐这种方式

## slurm 集群多机任务提交

算力平台使用Slurm调度，常用Slurm命令包括：
``` shell
Slurm命令	功能
sinfo	查看集群分区状态
squeue	查看作业队列
srun, salloc	交互式运行作业
sbatch	提交作业
scancel	取消作业
scontrol	查看和修改作业参数
sacct	查看已完成作业
```

### 单机任务
参考脚本
前面"#SBATCH"是Slurm配置参数，解释如下：
``` shell
●--partition: 使用的队列名称
●--nodes: 节点数量，用多少台机器
●--ntasks-per-node：每个节点的进程数，和每节点的GPU数量保持一致
●--gres=gpu:8：每个节点分配的GPU数量
●--cpus-per-task：每个任务分配的CPU数量（建议不要修改），该节点的cpu总数为任务数乘以每个任务的cpu数，这个示例脚本中的cpu总数为8x8=64
```

#### 具体示例：

train.sh:
```
#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --nodelist=g1001
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

python main.py
```

提交任务
```
sbatch train.sh
```

### 多机任务
已测试通过torchrun的方式多机训练，需要设置"MASTER_ADDR"和"MASTER_PORT"两个环境变量，先提交一个主节点的任务，获取"MASTER_ADDR"，在提交从节点任务。一个4台机器的多机任务的操作示例如下：

注意：#SBATCH的nodes参数设置为1，slurm的多节点通信与bmtrain的环境变量有冲突，且srun不稳定，推荐采用slurm提交多个单节点任务，用torchrun的方式实现多节点通信。

##### 第一步：启动主节点
train_master.sh:
```
#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
MASTER_ADDR=`hostname`
MASTER_PORT=12345
echo $MASTER_ADDR
torchrun --nnodes=4 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py
```

提交主节点：
```
sbatch train_master.sh
```

在输出的log（slurm-xxx.log）中查看主节点的名称，例如此时查到主节点是"g1001"

##### 第二步：启动从节点
train_slave.sh:
```
#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
MASTER_ADDR=g1001
MASTER_PORT=12345
echo $MASTER_ADDR
torchrun --nnodes=4 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py
```

提交从节点，示例是一个4台机器的任务，因此再提交3个从节点程序
```
for i in {1..3};do
    sbatch train_slave.sh
done
```


#### TODOs
1 完善dockers、K8s集群的分布式多机任务训练
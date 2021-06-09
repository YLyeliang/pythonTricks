## Multi-GPU Training

## Notes

DDP实现模块和架构,

**ProcessGroup**: 该部分包含`ProcessGroup.hpp`和`Store.hpp`。前者包含进程组实现的抽象api. `c10d`提供三个实现:ProcessGroupGloo, ..NCCL, ..MPI. 后者为通信服务

**DistributedDataParallel**: 包含`distributed.py` , `comm.h`, `reducer.h`. 

第一个为python接口，其中`_sync_param`函数实现进程内参数同步，并将模型缓存从rank为0的进程广播到其他进程,进程间参数同步则由`Reducer.cpp`实现。 

第二个为广播辅助函数，在forward之前触发。

第三个为梯度同步的核心实现 in the backward pass.包含三个接口：

- `Reducer`: 该部分的在`distributed.py`调用构造函数，同时注册钩子函数`Reducer::autograd_hook()`来做梯度累加
- `autograd_hook()`:当梯度准备好时触发
- `prepare_for_backward()`: 在forward pass结束后调用，遍历autograd graph。

![](https://user-images.githubusercontent.com/16999635/72313120-4e7c1c80-3658-11ea-9c6d-44336b2daeac.png)



## Getting started

构建函数用于多进程调用，其中函数中第一个参数为rank，函数体中使用`init_process_group`初始化进程组，创建模型。在main中使用`mp.spawn(function,args=(),nprocs=world_size)`来进行调用，实现并发，可以省略launch步骤。



## Writing Distributed Applications

### Point-to-point Communication

![Send and Recv](https://pytorch.org/tutorials/_images/send_recv.png)

点对点通讯，使用`dist.send`，`dist.recv()`来收发数据，`dist.isend`和`dist.irecv`来收发数据，前者是阻塞式的；后者是非阻塞式的，返回一个work对象，该对方可以调用`wait()`来等待数据传输完成。

### Collective Communication

| [![Scatter](https://pytorch.org/tutorials/_images/scatter.png)](https://pytorch.org/tutorials/_images/scatter.png)Scatter | [![Gather](https://pytorch.org/tutorials/_images/gather.png)](https://pytorch.org/tutorials/_images/gather.png)Gather |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Reduce](https://pytorch.org/tutorials/_images/reduce.png)](https://pytorch.org/tutorials/_images/reduce.png)Reduce | [![All-Reduce](https://pytorch.org/tutorials/_images/all_reduce.png)](https://pytorch.org/tutorials/_images/all_reduce.png)All-Reduce |
| [![Broadcast](https://pytorch.org/tutorials/_images/broadcast.png)](https://pytorch.org/tutorials/_images/broadcast.png)Broadcast | [![All-Gather](https://pytorch.org/tutorials/_images/all_gather.png)](https://pytorch.org/tutorials/_images/all_gather.png)All-Gather |

通常,任何数学交换操作可调用操作符标志。pytorch支持4种，at element-wise level:

- `dist.ReduceOp.SUM`
- `dist.ReduceOp.PRODUCE`
- `dist.ReduceOp.MAX`
- `dist.ReduceOp.MIN`

## Distributed Training

`DistributedDataParallel`

首先需要对数据集进行切分，为每个rank提供等量的数据集。

**初始化方法**

- `MASTER_PORT` 主导rank 0的进程的机器端口
- `MASTER_ADDR`: ... IP地址
- `WORLD_SIZE`: 总进程数
- `RANK`: 每个进程的级别/序号，来区分是否是主进程；

**Shared File System**

当所有进程可以访问共享文件系统时，可以使用该方法进行启动：

```python
dist.init_process_group(
    init_method='file:///mnt/nfs/sharedfile',
    rank=args.rank,
    world_size=4)
```

**TCP**

使用TCP初始化，需要提供rank 0的ip地址和端口号，所有的workers将能连接到该进程并交换信息

```python
dist.init_process_group(
	init_method='tcp://10.1.1.20:23456',
	rank=args.rank,
	world_size=4)
```



## Backend

`torch.distributed`支持三种内置backends: `gloo`,  `mpi`, `nccl`。

经验上来讲，使用NCCL做分布式GPU训练，使用Gloo做分布式CPU训练。

目前，NCCL能发挥最大的分布式训练性能，

### Launch utility

`torch.distributed.launch`:该函数用来创建单节点多进程训练流程，其中每个节点会创建一个或多个进程。通过`--nproc_per_node`来指定每个节点创建的进程数量，示例如下：

1. 单节点多进程分布式训练:

   ```shell	
   python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py (--args)
   ```

2. 多节点多进程分布式训练(e.g. two nodes)

   Node 1:(IP:192.168.1.1, port:1234)

   ```shell
   python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 train.py (--args)
   ```

   Node 2: 

   ```shell
   python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 train.py (--args)
   ```

**注意事项**

1. 分布式训练目前使用NCCL后台来实现最大性能，因此推荐首选NCCL作为训练后台

2. 在训练项目中，必须解析参数``local_rank=LOCAL_PROCESS_RANK`，示例如下:

   ```python
   import argparse
   parser= argparse.ArgumentParser()
   parser.add_argument("--local_rank",type=int)
   args=parser.parser_args()
   ```

   并将device设置成local rank:

   ```python
   torch.cuda.set_device(args.local_rank)
   ```

3. 在训练项目中，需要在训练前初始化分布式后台，并保证`init_method='env://')`:

   ```python
   torch.distributed.init_process_group(backend="NCCL",init_method='env://')
   ```

4. 训练项目中的模型分布式函数设置如下:

   ```python
   model = torch.nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[args.local_rank],
                                                    output_device=args.local_rank)
   ```

   


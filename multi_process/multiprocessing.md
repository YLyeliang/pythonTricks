# Multi processing 多进程
## Introduction
multiprocessing与threading包类似。不同之处，在于其可以绕过Python的GIL，获取更高的效率。
此外，还引入了`Pool`API,该API提供一种便捷方法，赋予函数并行化处理一系列输入值的能力，可以将输入数据分配给不同进程处理（数据并行)。
如`1_.py`所示.

## Process类
通过创建Process类，并调用`start()`方法来生成进程。`1_.py`

## Contexts and start methods

multiprocessing提供三种方式来启动进程：

- spawn: 父进程开启一个全新的python解释器进程。子进程只继承能够运行`run()`方法的必要资源。该方法相比`fork`和`forksever`效率要低
- fork：父进程使用`os.fork()`来分叉一个python解释器。子进程开始时，有效等同于父进程。所有资源均被子进程继承。
- forkserver：服务器进程启动后，当一个新进程需要时，父进程链接到服务器并请求它来fork一个新进程。The fork服务器进程是单线程的，所以可以安全使用`os.fork()`

例子见`2_.py`。
  
## 进程间交换对象
进程间交换对象，可以使用队列或管道。具体例子见`3.py`

## 进程间同步
可以通过在进程内加锁的方式，来保证进程间的输出有序，见`3.py`

## 进程间共享状态
在进行并发编程时，通常最好尽量避免使用共享状态。使用多个进程时尤其如此。如果真的需要使用一些共享数据，那么 multiprocessing 提供了两种方法。

**共享内存**
可以使用 `Value` 或 `Array` 将数据存储在共享内存映射中。

**服务进程**
由 Manager() 返回的管理器对象控制一个服务进程，该进程保存Python对象并允许其他进程使用代理操作它们。

Manager() 返回的管理器支持类型： list 、 dict 、 Namespace 、 Lock 、 RLock 、 Semaphore 、 BoundedSemaphore 、 Condition 、 
Event 、 Barrier 、 Queue 、 Value 和 Array 。
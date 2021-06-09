## contextlib

此模块为涉及 [`with`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#with) 语句的常见任务提供了实用的工具。

@contextlib.**contextmanager**

这个函数是一个 [decorator](https://docs.python.org/zh-cn/3/glossary.html#term-decorator) ，它可以定义一个支持 [`with`](https://docs.python.org/zh-cn/3/reference/compound_stmts.html#with) 语句上下文管理器的工厂函数， 而不需要创建一个类或区 [`__enter__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__enter__) 与 [`__exit__()`](https://docs.python.org/zh-cn/3/reference/datamodel.html#object.__exit__) 方法。

```python
from contextlib import contextmanager

@contextmanager
def managed_resource(*args, **kwds):
    # Code to acquire resource, e.g.:
    resource = acquire_resource(*args, **kwds)
    try:
        yield resource
    finally:
        # Code to release resource, e.g.:
        release_resource(resource)

>>> with managed_resource(timeout=3600) as resource:
...     # Resource is released at the end of this block,
...     # even if code in the block raises an exception
```



## eval函数

eval() 函数用来执行一个字符串表达式，并返回表达式的值。

```python
x = 7
eval('3 * x')
21
```

同时，也可以用于表示对象的字符串名称，比如类、函数等。

```python
def hello():
    print("hello")
f = eval("hello")
f()
hello
```



## glob模块

[`glob`](https://docs.python.org/zh-cn/3/library/glob.html#module-glob) 模块可根据 Unix 终端所用规则找出所有匹配特定模式的路径名，但会按不确定的顺序返回结果。 波浪号扩展不会生效，但 `*`, `?` 以及表示为 `[]` 的字符范围将被正确地匹配。这是通过配合使用 [`os.scandir()`](https://docs.python.org/zh-cn/3/library/os.html#os.scandir) 和 [`fnmatch.fnmatch()`](https://docs.python.org/zh-cn/3/library/fnmatch.html#fnmatch.fnmatch) 函数来实现的，而不是通过实际发起调用子终端。

## Multiprocessing

### Contexts and start methods

multiprocessing提供三种方式来启动进程：

- spawn: 父进程开启一个全新的python解释器进程。子进程只继承能够运行`run()`方法的必要资源。该方法相比`fork`和`forksever`效率要低
- fork：父进程使用`os.fork()`来分叉一个python解释器。子进程开始时，有效等同于父进程。所有资源均被子进程继承。
- forkserver：服务器进程启动后，当一个新进程需要时，父进程链接到服务器并请求它来fork一个新进程。The fork服务器进程是单线程的，所以可以安全使用`os.fork()`

 
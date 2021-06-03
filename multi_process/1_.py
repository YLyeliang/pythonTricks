from multiprocessing import Pool, Process
import os


def f(x):
    return x ** 2


def f2(name):
    info("function f")
    print("hello", name)


# 获取信息
def info(title):
    print(title)
    print(f"module name: {__name__}")
    print(f"parent process: {os.getppid()}")
    print(f"process id: {os.getpid()}")


if __name__ == '__main__':
    # Pool API
    # with Pool(5) as p:
    #     print(p.map(f, (1, 2, 3)))

    info("main line")
    for i in range(10):
        p = Process(target=f2, args=('bob Zhu',))
        p.start()
        p.join()

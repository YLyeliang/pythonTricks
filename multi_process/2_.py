import multiprocessing as mp


def foo(q):
    q.put("hello")


if __name__ == '__main__':
    mp.set_start_method("spawn")
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()
    # 或者，使用get_context()来获取上下文对象。
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()

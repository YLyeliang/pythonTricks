from multiprocessing import Process, Pipe, Queue, Lock
from multiprocessing import Value, Array, Manager


def fPipe(conn):
    conn.send([42, None, 'hello'])
    conn.close()


def fQueue(q):
    q.put([42, None, 'hello'])


def fSync(l, i):
    l.acquire()
    try:
        print("hello world", i)
    finally:
        l.release()


def fShare(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]


def fManager(d, l):
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()


if __name__ == '__main__':
    # 1. 进程间交换
    # Pipe
    parent_conn, child_conn = Pipe()
    p = Process(target=fPipe, args=(child_conn,))
    p.start()
    print(parent_conn.recv())  # prints "[42, None, 'hello']"
    p.join()

    # Queue
    q = Queue()
    p = Process(target=fQueue, args=(q,))
    p.start()
    print(q.get())  # prints "[42, None, 'hello']"
    p.join()

    # 2. 进程间同步
    # 不使用锁的情况下，来自于多进程的输出很容易产生混淆。
    lock = Lock()
    for num in range(10):
        Process(target=fSync, args=(lock, num)).start()

    # 3. 进程间共享状态
    # value & array
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=fShare, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])

    # manager
    with Manager() as manager:
        d = manager.dict()
        l = manager.list(range(10))

        p = Process(target=fManager, args=(d, l))
        p.start()
        p.join()

        print(d)
        print(l)

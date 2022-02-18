"""
multi-processing多进程卡死问题。
使用queue时，多个进程同时写，然后在join部分会阻塞。
解决：
1. 在Join之前消费掉queue里的元素
"""
import multiprocessing as mp


def test_param(a, num):
    b = []
    for i in range(num * 10000, (num + 1) * 10000):
        b.append(i)
    if isinstance(a, list):
        a.extend(b)
    else:
        a.put(b)


class dd:
    a = []


if __name__ == '__main__':
    # test_for_loop()
    # test_close()
    gl = []
    a = [8, 8, 8]
    # manager = mp.Manager()
    # a = manager.list(a)
    queue = mp.Queue()
    print(type(queue))
    print(type(a))
    print(isinstance(queue, list))
    for i in range(5):
        gl.append(mp.Process(target=test_param, args=(queue, i)))
        gl[i].start()

    for g in gl:
        while g.is_alive():
            while not queue.empty():
                a.extend(queue.get())
    for g in gl:
        g.join()
    # test_param(a)
    print("done")
    # while not queue.empty():
    #     a.extend(queue.get())
    print(a)

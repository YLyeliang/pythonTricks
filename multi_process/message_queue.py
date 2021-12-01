from multiprocessing import Process, Queue, Pool
import time, os


def produce(nums):
    for num in nums:
        messge = f"get num : {num}, and plus 100 on it, and get {num + 100}"
        print(messge)
        num += 100
        time.sleep(1)
        yield num


def broker(num):
    messge = f"get num from producer, and the num is: {num}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    print(messge)
    time.sleep(0.1)


class messgeQueue(object):
    def __init__(self, process_num=3):
        self.process_num = process_num
        self.bag_queue = Queue()
        self.bag_download_func = produce
        self.parse_pool = Pool(6)

    def wrapper(self, func, q, *args, **kwargs):

        for res in func(*args, **kwargs):
            q.put(res)

    def producer(self, in_num_list):
        length = len(in_num_list)
        step = length // self.process_num
        for i in range(self.process_num):
            if i + 1 == self.process_num:
                in_nums = in_num_list[i * step:]
            else:
                in_nums = in_num_list[i * step:(i + 1) * step]
            p = Process(target=self.wrapper, args=(self.bag_download_func, self.bag_queue, in_nums))
            p.daemon = True
            p.start()

    def broker(self, func):
        while True:
            if not self.bag_queue.empty():
                in_num = self.bag_queue.get()
                self.parse_pool.apply_async(func, args=(in_num,))

    def run(self, num_list):
        self.producer(num_list)
        self.broker(broker)


if __name__ == '__main__':
    num_list = list(range(1000))
    messge = messgeQueue()
    messge.run(num_list)

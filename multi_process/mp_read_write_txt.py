import multiprocessing as mp
import time


def transform_func(num_list, que, index):
    print("start")
    que.put({str(index): list(map(lambda x: format(int(x), ',')+"\n", num_list))})
    print("end")


def for_loop():
    out_list = list(map(lambda x:format(int(x),',')+"\n",raw))
    out_f.writelines(out_list)

def mp_loop():
    mp_num = 16
    num = len(raw) // mp_num
    p_list = []

    que = mp.Queue()
    for i in range(mp_num):
        sub_raw = raw[i * num:(i + 1) * num]
        p_list.append(mp.Process(target=transform_func, args=(sub_raw, que, i)))
        p_list[i].start()

    whole_dict = {}
    for g in p_list:
        while g.is_alive():
            while not que.empty():
                sub_dict = que.get()
                for key, val in sub_dict.items():
                    whole_dict[key] = val

    for p in p_list:
        p.join()
    print("transform end")
    dict_len = len(whole_dict)
    for i in range(dict_len):
        out_f.writelines(whole_dict[str(i)])

if __name__ == '__main__':
    # with open("file.txt","w") as f:
    #     f.writelines(list(map(lambda x:str(x)+"\n",list(range(10000000)))))
    debug =1
    with open("file.txt", "r") as f:
        raw = f.read().splitlines()
    out_file = "out_f.txt"
    out_f = open(out_file, "w")
    s =time.time()
    mp_loop()
    e=time.time()
    print(e-s)

    s=time.time()
    for_loop()
    e=time.time()
    print(e-s)

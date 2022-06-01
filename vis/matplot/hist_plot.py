import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Heiti TC'


def plot_hist(arr):
    weights = np.ones_like(arr) / 10
    binwidth = 0.1
    print(np.arange(min(arr), max(arr) + binwidth, binwidth))
    plt.hist(arr, bins=np.arange(min(arr), max(arr) + binwidth, binwidth),
             edgecolor="tab:orange", alpha=0.5)

    plt.legend(['区间宽度: 0.1'])
    plt.xlabel("yaw值区间")
    plt.ylabel("出现频次")
    plt.title("yaw角数值直方图")

    plt.show()


np.tan()


if __name__ == '__main__':
    with open("tmp.txt") as f:
        lines = f.readlines()

    yaw_arr = []
    for line in lines:
        if "yaw" in line:
            yaw = line.split("yaw:")[-1].rstrip('\n')
            yaw = eval(yaw)
            yaw_arr.append(yaw)
    print(yaw_arr)
    plot_hist(yaw_arr)

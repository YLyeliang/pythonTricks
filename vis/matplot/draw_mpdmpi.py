import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import pandas as pd

titleA = "highway"

titleB = "Urban"

mpd_highway_dev = [320, 44.27]
mpd_highway_ep = [175, 46.25]
mpd_urban_dev = [24, 23.74]
mpd_urban_ep = [35, 20.56]

mpi_highway_dev = [12, 14.63]
mpi_highway_ep = [26.1, 13.43]
mpi_urban_dev = [22, 9.85]
mpi_urban_ep = [15.8, 10.37]


# for copy
def init_plot_params():
    plt.figure()  # 创建画布
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，不然中文无法显示

    plt.rcParams['figure.figsize'] = (8.0, 4.0)  # 设置figure_size尺寸
    # figsize(12.5, 4) # 设置 figsize
    plt.rcParams['savefig.dpi'] = 300  # 保存图片分辨率
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
    # 指定dpi=200，图片尺寸为 1200*800
    # 指定dpi=300，图片尺寸为 1800*1200

    plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
    plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style


def draw_plot(mpd_highway_dev,
              mpd_highway_ep,
              mpi_highway_dev,
              mpi_highway_ep,
              name = "highway"):
    init_plot_params()
    x = list(range(len(mpd_highway_ep)))
    plt.plot(x, mpd_highway_dev, 'r', label='mpd-highway-dev')
    plt.plot(x, mpd_highway_ep, 'g', label='mpd_highway_ep')
    plt.plot(x, mpi_highway_dev, 'b', label='mpi_highway_dev')
    plt.plot(x, mpi_highway_ep, 'yellow', label='mpi_highway_ep')
    plt.plot(x, mpd_highway_dev, 'ro-', x, mpd_highway_ep, 'g+-', x, mpi_highway_dev, 'b^-', x, mpi_highway_ep, "y+-")
    plt.title(f"{name}-MPI-MPD")
    plt.xlabel("Time")
    plt.ylabel("Number")
    ax = plt.gca()
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 5)
    plt.legend()
    plt.savefig(name+'.png')
    # plt.show()


if __name__ == '__main__':
    # draw_plot(mpd_highway_dev, mpd_highway_ep, mpi_highway_dev, mpi_highway_ep)
    draw_plot(mpd_urban_dev, mpd_urban_ep, mpi_urban_dev, mpi_urban_ep,"urban")

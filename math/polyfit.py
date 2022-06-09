import traceback

import numpy as np
import matplotlib.pyplot as plt
import sys


def sin_polyfit():
    xxx = np.arange(0, 1000)  # x值，此时表示弧度
    yyy = np.sin(xxx * np.pi / 180)  # 函数值，转化成度

    z1 = np.polyfit(xxx, yyy, 3)  # 用7次多项式拟合，可改变多项式阶数；
    p1 = np.poly1d(z1)  # 得到多项式系数，按照阶数从高到低排列
    yvals = p1(xxx)  # 可直接使用yvals=np.polyval(z1,xxx)
    print(z1)
    print(p1)  # 显示多项式

    plt.plot(xxx, yyy, '*', label='original values')
    plt.plot(xxx, yvals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
    plt.title('polyfitting')
    plt.show()


def poly_angle():
    x = [100, 101, 102, 103]
    y = [99.4, 99.6, 99.8, 100]
    start = np.polyfit(x, y, 1)
    angle = np.degrees(abs(np.arctan(start[0])))
    # a = 1 + "nene" + None
    debug = 1


def interp():
    x = [0, 1, 1.5, 2.72, 3.5]
    xp = [1, 2, 3]
    yp = [3, 2, 0]
    y = np.interp(x, xp, yp) # array(. . . )



if __name__ == '__main__':
    f = open("tmp.log", "w")
    try:
        # poly_angle()
        sin_polyfit()
    except Exception as e:  # traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** format_exception:")
        f = repr(traceback.format_exception(exc_type, exc_value,
                                            exc_traceback))

    print("hehe")

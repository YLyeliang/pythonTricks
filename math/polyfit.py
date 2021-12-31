import numpy as np
import matplotlib.pyplot as plt

xxx = np.arange(0, 1000)  # x值，此时表示弧度
yyy = np.sin(xxx*np.pi/180)  #函数值，转化成度

z1 = np.polyfit(xxx, yyy, 1) # 用7次多项式拟合，可改变多项式阶数；
p1 = np.poly1d(z1) #得到多项式系数，按照阶数从高到低排列
yvals=p1(xxx) # 可直接使用yvals=np.polyval(z1,xxx)
print(z1)
print(p1)  #显示多项式

plt.plot(xxx, yyy, '*',label='original values')
plt.plot(xxx, yvals, 'r',label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4) # 指定legend在图中的位置，类似象限的位置
plt.title('polyfitting')
plt.show()
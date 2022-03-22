import matplotlib.pyplot as plot
import numpy as np
from numpy.lib.scimath import logn
import matplotlib as mpl

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 确定坐标轴
plot.xlim((0, 100))
plot.ylim((0, 100))

# 设置坐标轴名称
plot.xlabel('输入规模')
plot.ylabel('时间')

# 产生等差数列
x = np.linspace(1, 100, 100)


def f(x):
    y = x
    return y


plot.plot(x, logn(2, x), 'r-', linewidth=1, label='lgx')
plot.plot(x, 2*f(x), 'g--', linewidth=1, label='f(x)')

plot.legend(['lgx', 'f(x)'], loc='upper left')
plot.show()

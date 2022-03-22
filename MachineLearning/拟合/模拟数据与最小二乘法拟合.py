import matplotlib.pyplot as plot
import numpy as np
from numpy.lib.scimath import logn
import matplotlib as mpl

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

plot.xlim((0, 100))
plot.ylim((0, 100))

plot.xlabel("x")
plot.ylabel("y")

# linespace(start, end, number) 生成100由0递进到100的数
# random.randint(start, end, number) 生成100个-5到5之间的随机数
x = np.linspace(0, 100, 100)
y = x + np.random.randint(-5, 5, 100)

print(x)
print(y)
plot.plot(x, y, 'o')

# 最小二乘法拟合数据
num = len(y)
mu_x = np.sum(x) / num
mu_y = np.sum(y) / num
mu_xx = np.sum(np.multiply(x, x)) / num
mu_xy = np.sum(np.multiply(x, y)) / num
w1 = (mu_xy - mu_x * mu_y) / (mu_xx - mu_x * mu_x)
w0 = mu_y - w1 * mu_x


def h(x):
    y = w1 * x + w0
    return y


plot.plot(x, h(x), 'r-')

print(h(150))
plot.show()

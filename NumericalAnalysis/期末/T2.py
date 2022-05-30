import random

import matplotlib.pyplot as plt
import numpy as np

# 已知条件：求积节点x0、x1、……、xn，对应函数值y0、y1、……、yn，积分上下限a、b；
# 生成11个x，随机生成11个y，构造一个10次的拉格朗日函数
a, b = 0, 10
xs = np.linspace(0, 10, 11)
ys = [random.uniform(5, 10) for i in range(11)]


def lagrange(x):
    Lx = 0
    for i in range(11):
        li = 1
        for j in range(11):
            if i != j:
                li *= (x - xs[j]) / (xs[i] - xs[j])
        Lx += li * ys[i]
    return Lx

x_pred, y_pred = np.linspace(0,10,100), []
for x in x_pred:
    y_pred.append(lagrange(x))

plt.plot(xs, ys, '.')
plt.plot(x_pred, y_pred)
plt.show()
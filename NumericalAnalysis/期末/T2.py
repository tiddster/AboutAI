import math
import random

import matplotlib.pyplot as plt
import numpy as np

# 已知条件：求积节点x0、x1、……、xn，对应函数值y0、y1、……、yn，积分上下限a、b；
# 生成11个x，构造一个10次的拉格朗日函数
from scipy import integrate


def f(x):
    return math.sin(x) * math.cos(x)
a, b = 0, 1
xs = np.linspace(0, 1, 11)
ys = [f(x) for x in xs]


def lagrange(a, b):
    Lx = 0
    for i in range(len(xs)):
        Ai = integral(a, b, i)  # 四舍五入
        Lx += Ai * ys[i]
    return Lx


def lagrangeB(x, i):
    li = 1
    for j in range(len(xs)):
        if i != j:
            li *= (x - xs[j]) / (xs[i] - xs[j])
    return li


def integral(a, b, k):
    N, h, S = 1000, (b - a) / 1000, 0
    for i in range(N):
        x = a + h * i
        S += h * lagrangeB(x, k)
    return S


def quad(a, b):
    res = integrate.quad(f, a, b)[0]
    return res


print(f"利用高斯求积公式的结果{quad(0,1)}")
print(f"利用拉格朗日型插值求积分结果{lagrange(0, 1)}")

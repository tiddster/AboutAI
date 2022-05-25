import math

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 3 * x


def q(x):
    return 4


def yf(x):
    return 3 * (math.exp(2 * x) - math.exp(-2 * x)) / (4 * (math.exp(2) - math.exp(-2))) - 3 * x / 4


def difference(N, a, b, alpha, beta):
    """
    通过差分公式直接转换成三对角方程组, 所求解的答案便是每一个yi
    a,b: 左右边界
    alpha, beta: 分别对应y(a), y(b)的值
    """
    h = (b - a) / N
    fi, mat = np.zeros((N - 1, 1)), np.zeros((N - 1, N - 1))
    y0, yN = alpha, beta
    xs = []
    y_ori = [0] * (N - 1)
    # 初始化三对角方程
    for i in range(0, N - 1):
        xi = a + (i + 1) * h
        xs.append(xi)
        y_ori[i] = yf(xi)
        if i + 1 == 1:
            fi[i] = h ** 2 * f(xi) - y0
            mat[i, i], mat[i, i + 1] = -(2 + q(xi) * h ** 2), 1
        elif i + 1 == N - 1:
            fi[i] = h ** 2 * f(xi) - yN
            mat[i, i - 1], mat[i, i] = 1, -(2 + q(xi) * h ** 2)
        else:
            fi[i] = h ** 2 * f(xi)
            mat[i, i - 1], mat[i, i], mat[i, i + 1] = 1, -(2 + q(xi) * h ** 2), 1
    # 求解三对角方程：
    y_pred = np.linalg.inv(mat) @ fi
    print(y_pred.ravel())

    plt.plot(xs, y_pred.ravel())
    plt.plot(xs, y_ori, '.')
    plt.show()


difference(10, 0, 1, 0, 0)

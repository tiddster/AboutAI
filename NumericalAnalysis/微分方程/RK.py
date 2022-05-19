import numpy as np
from matplotlib import pyplot as plt


def f_(x, y):
    return y - 2 * x / y if y!=0 else 1

def f(x):
    return (1 + 2 * x) ** 0.5

def RK(a,b,N,y0,f_):
    h = (b - a) / N
    xs = np.zeros(N + 1)
    y_pred = np.zeros(N + 1)
    ys = np.zeros(N + 1)
    y_pred[0], xs[0], ys[0] = y0, a, y0
    for i in range(1, N + 1):
        xs[i] = xs[i-1] + h
        k1 = f_(xs[i-1], y_pred[i-1])
        k2 = f_(xs[i-1] + h/2, y_pred[i-1] + h/2 * k1)
        k3 = f_(xs[i-1] + h/2, y_pred[i-1] + h/2 * k2)
        k4 = f_(xs[i-1] + h, y_pred[i-1] + h * k3)
        y_pred[i] = y_pred[i-1]+h/6*(k1 + 2*k2 + 2*k3 + k4)
        ys[i] = f(xs[i])
        print(f"第{i}个 4阶经典龙格-库塔法方法函数值为{y_pred[i]} 真实值为{ys[i]}")

    print(y_pred)
    plt.plot(xs, y_pred)
    plt.plot(xs, ys, '.')
    plt.show()

RK(0,1,5,1,f_)
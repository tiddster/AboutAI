import numpy as np
from matplotlib import pyplot as plt


def f_(x, y):
    return y - 2 * x / y


def f(x):
    return (1 + 2 * x) ** 0.5


# 显示欧拉法
def explicit_euler(a, b, N, y0, f_):
    h = (b - a) / N
    xs = np.zeros(N + 1)
    y_pred = np.zeros(N + 1)
    ys = np.zeros(N + 1)
    y_pred[0], xs[0], ys[0] = y0, a, y0
    for i in range(1, N + 1):
        xs[i] = xs[i - 1] + h
        ys[i] = f(xs[i])
        y_pred[i] = y_pred[i - 1] + h * f_(xs[i - 1], y_pred[i - 1])

    print(y_pred)
    plt.plot(xs, y_pred)
    plt.plot(xs, ys, '.')
    plt.show()


def implicit_euler(a, b, N, y0, f_):
    h = (b - a) / N
    xs = np.zeros(N + 1)
    y_pred = np.zeros(N + 1)
    ys = np.zeros(N + 1)
    y_pred[0], xs[0], ys[0] = y0, a, y0
    for i in range(1,N+1):
        xs[i] = xs[i-1] + h
        y1 = y_pred[i-1] + h * f_(xs[i-1], y_pred[i-1])
        y2 = y_pred[i-1] + h * f_(xs[i], y1)
        while abs(y2 - y1) > 0.01:
            y1 = y2
            y2 = y_pred[i] + h * f_(xs[i], y1)
        y_pred[i] = y2
        ys[i] = f(xs[i])
    print(y_pred)
    plt.plot(xs, y_pred)
    plt.plot(xs, ys, '.')
    plt.show()


def improve_euler(a, b, N, y0, f_):
    h = (b - a) / N
    xs = np.zeros(N + 1)
    y_pred = np.zeros(N + 1)
    ys = np.zeros(N + 1)
    y_pred[0], xs[0], ys[0] = y0, a, y0
    for i in range(1, N + 1):
        xs[i] = xs[i - 1] + h
        yp = y_pred[i - 1] + h * f_(xs[i - 1], y_pred[i - 1])
        yc = y_pred[i - 1] + h * f_(xs[i], yp)
        y_pred[i] = (yp + yc) / 2
        ys[i] = f(xs[i])

    print(y_pred)
    plt.plot(xs, y_pred)
    plt.plot(xs, ys, '.')
    plt.show()


implicit_euler(0, 1, 10, 1, f_)

import math

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return math.cos(x)


def q(x):
    return 1


def p(x):
    return 2

def NormalDifference(N, a, b, alpha, beta):
    h = (b - a) / N
    fi, mat = np.zeros((N - 1, 1)), np.zeros((N - 1, N - 1))
    y0, yN = alpha, beta
    xs = []

    for i in range(0, N-1):
        xi = a + (i+1) * h
        xs.append(xi)
        if i+1 == 1:
            fi[i] = 2 * h**2 * f(xi) - (2-h*p(xi)) * y0
            mat[i, i], mat[i, i+1] = -2 * (2-h**2*q(xi)), 2 + h*p(xi)
        elif i+1 == N-1:
            fi[i] = 2 * h**2 * f(xi) - (2+h*p(xi)) * yN
            mat[i, i-1], mat[i, i] = 2 - h*p(xi), -2 * (2-h**2*q(xi))
        else:
            fi[i] = 2 * h**2 * f(xi)
            mat[i, i - 1], mat[i, i], mat[i, i+1] = 2-h*p(xi), -2*(2-h**2*q(xi)), 2+h*p(xi)

    y_pred = np.linalg.inv(mat) @ fi
    print(y_pred.ravel())

    plt.plot(xs, y_pred)
    plt.show()

NormalDifference(10, 0, math.pi/2, -0.3, -0.1)
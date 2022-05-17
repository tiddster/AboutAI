import numpy as np


def costFunc(theta, X, y):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(1,n), linear regression parameters
    """
    m = X.shape[0]
    #print(X.shape, theta.shape)
    hx = X @ theta.T
    cost = np.sum((hx - y) ** 2) / (2 * m)

    return cost


def reg_costFunc(theta, X, y, l=1):
    cost = costFunc(theta, X, y)
    reg = l * np.sum(theta[1:] ** 2) / (2 * len(y))
    return cost + reg

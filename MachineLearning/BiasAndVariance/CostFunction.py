import numpy as np


def cost_function(theta, X, y):
    """
    X: R(m*n), m records, n features
    y: R(m)
    theta : R(n), linear regression parameters
    """

    m = X.shape[0]

    print(theta.shape)
    print(X.shape)

    hx = X @ theta
    cost = np.sum((hx - y) ** 2) / (2*m)

    return cost
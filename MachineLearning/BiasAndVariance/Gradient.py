import numpy as np
from scipy.optimize import minimize

from MachineLearning.BiasAndVariance.CostFunction import reg_costFunc


def gradient(theta, X, y):
    theta = np.matrix(theta)
    m = X.shape[0]

    hx = X @ theta.T
    grad = (hx - y).T @ X

    return grad / m


def reg_gradient(theta, X, y, l=1):
    m = X.shape[0]

    reg_grad = theta.copy()
    reg_grad[0] = 0
    reg_grad = (l / m) * reg_grad

    return gradient(theta, X, y) + reg_grad


def findBest(X, y, l=1):
    theta = np.ones(X.shape[1])
    res = minimize(fun=reg_costFunc, x0=theta, args=(X, y, l), method='TNC', jac=reg_gradient)
    return res

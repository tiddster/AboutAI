import numpy as np
import MachineLearning.LogisiticRegression.Sigmoid as Sigmoid

'''
X: (m,n)矩阵
y: (m,1)矩阵
theta: (n)标量 或 (n,1)矩阵
'''


def costFunc(theta, X, y):
    m = len(y)
    hx = Sigmoid.sigmoid(X @ theta)
    cost = np.sum((-y * np.log(hx) - (1 - y) * np.log(1 - hx))) / m
    return cost


def regCostFunc(theta, X, y, l=1):
    hx = Sigmoid.sigmoid(X @ theta)
    cost = np.sum(np.multiply(-y, np.log(hx)) - np.multiply(1 - y, np.log(1 - hx))) / len(y)
    regulation = l * np.sum(theta[1:] ** 2) / (2 * len(y))
    return cost + regulation

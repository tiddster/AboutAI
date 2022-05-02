import numpy as np
import MachineLearning.LogisiticRegression.Sigmoid as Sigmoid
import MachineLearning.LogisiticRegression.CostFunction as CostFunction
import scipy.optimize as opt

'''
X: (m,n)矩阵
y: (m,1)矩阵
theta: (n)标量 或 (n,1)矩阵
'''


def gradient(theta, X, y):
    m = len(y)
    hx = Sigmoid.sigmoid(X @ theta)
    error = hx.T - y
    gradTheta = error.T @ X / len(y)
    return gradTheta


def regGradient(theta, X, y, l=1):
    m = len(y)
    tempTheta = l * theta / m
    tempTheta[0] = 0
    return gradient(theta, X, y) + tempTheta


def findBest(m, X, y, l=0):
    theta = np.zeros(m)
    X = np.matrix(X)
    Y = np.matrix(y)
    result = opt.minimize(fun=CostFunction.regCostFunc, x0=theta, args=(X, Y, l), jac=regGradient, method='TNC')
    return result

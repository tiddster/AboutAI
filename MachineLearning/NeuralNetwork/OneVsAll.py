import numpy as np
from scipy.optimize import minimize
import MachineLearning.LogisiticRegression.CostFunction as CostFunction
import MachineLearning.LogisiticRegression.Gradient as Gradient


def getAllLabels(y):
    labels = np.unique(y)
    return labels


def oneVsAll(X, y, labelNum, l):
    r, c = X.shape

    allTheta = np.zeros([labelNum, c])

    for i in range(1, labelNum + 1):
        theta = np.zeros(c)
        yi = np.array([1 if label == i else 0 for label in y])

        res = minimize(fun=CostFunction.regCostFunc, x0=theta, args=(X, yi, l), jac=Gradient.regGradient, method='TNC')
        allTheta[i - 1, :] = res.x
    return allTheta

import numpy as np

import MachineLearning.LogisiticRegression.Sigmoid as Sigmoid


def Layer1(X, theta1):
    """
    :param X: 不带常量1的X
    :param theta1:
    :return:
    """
    a1 = X
    ones = np.ones([a1.shape[0], 1])
    a1 = np.hstack((ones, a1))

    z2 = a1 @ theta1.T
    a2 = Sigmoid.sigmoid(z2)
    return a2


def Layer2(X, theta1, theta2):
    a2 = Layer1(X, theta1)
    ones = np.ones([a2.shape[0], 1])
    a2 = np.hstack((ones, a2))

    z3 = a2 @ theta2.T
    a3 = Sigmoid.sigmoid(z3)
    return a3

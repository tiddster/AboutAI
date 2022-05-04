import numpy as np

from MachineLearning.LogisiticRegression.Sigmoid import sigmoid


def forwardP(X, theta1, theta2):
    m = X.shape[0]
    ones = np.ones((m,1))

    a1 = np.hstack((ones, X))
    z2 = a1 @ theta1
    a2 = np.hstack((ones, sigmoid(z2)))
    z3 = a2 @ theta2
    h = sigmoid(z3)

    return a1, z2, a2, z3, h
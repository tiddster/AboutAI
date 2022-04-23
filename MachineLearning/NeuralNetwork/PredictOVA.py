import MachineLearning.LogisiticRegression.Sigmoid as Sigmoid
import numpy as np


def predictOVA(X, thetas):
    r, c = X.shape[0], X.shape[1]
    labelNums = thetas.shape[0]
    ps = Sigmoid.sigmoid(X @ thetas.T)
    predictions = np.argmax(ps, axis=1)
    return predictions + 1


def getAccuracy(X, y, thetas):
    predictions = predictOVA(X, thetas)
    y = y.ravel()
    total,right = len(y), 0
    for p, o in zip(predictions, y):
        if p == o:
            right += 1

    return right / total

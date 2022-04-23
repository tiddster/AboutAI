import MachineLearning.LogisiticRegression.Sigmoid as Sigmoid
import numpy as np


def predictOVA(X, thetas):
    r, c = X.shape[0], X.shape[1]

    ones = np.ones([r,1])
    oneX = np.hstack((ones,X))

    labelNums = thetas.shape[0]
    ps = Sigmoid.sigmoid(oneX @ thetas.T)
    predictions = np.argmax(ps, axis=1)
    return predictions + 1


def getAccuracy(X, y, thetas):
    predictions = predictOVA(X, thetas)
    print(predictions)
    y = y.ravel()
    total,right = len(y), 0
    for p, o in zip(predictions, y):
        if p == o:
            right += 1

    return right / total

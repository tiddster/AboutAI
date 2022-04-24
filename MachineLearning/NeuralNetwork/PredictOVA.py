import MachineLearning.LogisiticRegression.Sigmoid as Sigmoid
import numpy as np


def predictOVA(X, thetas, pros):
    """
    :param X: 不带常量1的X
    :param pros: 针对于NN，已经存在概率，直接进行预测即可
    :param thetas: 针对LR，不存在概率，需用sigmoid和X去计算概率值，再进行预测
    :return:
    """
    r, c = X.shape[0], X.shape[1]
    if pros is None:
        ones = np.ones([r,1])
        oneX = np.hstack((ones,X))

        labelNums = thetas.shape[0]
        pros = Sigmoid.sigmoid(oneX @ thetas.T)
    predictions = np.argmax(pros, axis=1)
    return predictions + 1


def getAccuracy(X, y, pros=None, thetas=None):
    """
    :param X: 不带常量1的X
    :param y: 真实值
    :param pros: 针对于NN，NN算出来直接是概率，传入predictOVA，不用再次sigmoid就可以直接用
    :param thetas: 针对LR，需要在predictOVA中，用sigmoid去计算概率值
    :return:
    """
    predictions = predictOVA(X, thetas, pros)
    print(predictions)

    y = y.ravel()
    total,right = len(y), 0
    for p, o in zip(predictions, y):
        if p == o:
            right += 1

    return right / total

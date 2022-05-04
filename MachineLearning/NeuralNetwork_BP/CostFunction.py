import numpy as np

from MachineLearning.NeuralNetwork_BP.ForwardPropagate import forwardP


def costFunction(thetas, inputSize, hiddenSize, numLabels, X, Y_OH, l=0):
    """
    :param thetas: 全部theta以1*n维的形式作为形参，所以要用reshape分开
    :param inputSize: 输入层的大小
    :param hiddenSize: 隐层的大小
    :param numLabels: 输出层的大小（因为输出层的个数就对应着10个数字标签）
    :param X:
    :param Y_OH:
    :param l:
    :return:
    """
    m = X.shape[0]

    theta1 = np.reshape(thetas[:hiddenSize * (inputSize + 1)], (inputSize+1, hiddenSize))
    theta2 = np.reshape(thetas[hiddenSize * (inputSize + 1):], (hiddenSize+1, numLabels))

    a1, z2, a2, z3, h = forwardP(X, theta1, theta2)
    print(h)

    J = 0
    for i in range(m):
        cost = np.sum((-Y_OH[i, :] * np.log(h[i,:])) - ((1 - Y_OH[i, :]) * np.log(1 - h[i,:])))
        J += cost

    return J/m
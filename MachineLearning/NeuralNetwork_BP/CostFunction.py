import numpy as np

from MachineLearning.NeuralNetwork_BP.ForwardPropagate import forward_prop


def cost_function(thetas, inputSize, hiddenSize, numLabels, X, Y_OH):
    """
    :param thetas: 全部theta以1*n维的形式作为形参，所以要用reshape分开
    :param inputSize: 输入层的大小
    :param hiddenSize: 隐层的大小
    :param numLabels: 输出层的大小（因为输出层的个数就对应着10个数字标签）
    :param X:
    :param Y_OH:
    :return:
    """
    m = X.shape[0]

    theta1 = np.reshape(thetas[:hiddenSize * (inputSize + 1)], (inputSize + 1, hiddenSize))
    theta2 = np.reshape(thetas[hiddenSize * (inputSize + 1):], (hiddenSize + 1, numLabels))

    a1, z2, a2, z3, h = forward_prop(X, theta1, theta2)

    J = 0
    for i in range(m):
        cost = np.sum((-Y_OH[i, :] * np.log(h[i, :])) - ((1 - Y_OH[i, :]) * np.log(1 - h[i, :])))
        J += cost

    return J / m


def reg_cost_function(thetas, inputSize, hiddenSize, numLabels, X, Y_OH, l=0.1):
    m = X.shape[0]

    J = cost_function(thetas, inputSize, hiddenSize, numLabels, X, Y_OH)

    theta1 = np.reshape(thetas[:hiddenSize * (inputSize + 1)], (inputSize + 1, hiddenSize))
    theta2 = np.reshape(thetas[hiddenSize * (inputSize + 1):], (hiddenSize + 1, numLabels))

    J += float(l) / (2 * m) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:,1:]**2))

    return J
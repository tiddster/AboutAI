import numpy as np

from MachineLearning.NeuralNetwork_BP.CostFunction import reg_cost_function
from MachineLearning.NeuralNetwork_BP.ForwardPropagate import forward_prop
from MachineLearning.NeuralNetwork_BP.Sigmoid import sigmoid_grad


# bp算法的准备操作，或者说是正则化bp算法和普通bp算法的公共部分,以减少代码量
def pre_back_prop(thetas, input_size, hidden_size, num_labels, X, Y_OH, l=1):
    m = X.shape[0]

    one = np.ones((1, 1))

    theta1 = np.reshape(thetas[:(input_size + 1) * hidden_size], (input_size + 1, hidden_size))  # (401, 25)
    theta2 = np.reshape(thetas[(input_size + 1) * hidden_size:], (hidden_size + 1, num_labels))  # (26, 10)

    delta1 = np.zeros(theta1.shape)  # (401, 25)
    delta2 = np.zeros(theta2.shape)  # (26, 10)

    a1, z2, a2, z3, h = forward_prop(X, theta1, theta2)
    a1, z2, a2, z3, h = np.matrix(a1), np.matrix(z2), np.matrix(a2), np.matrix(z3), np.matrix(h)

    J = reg_cost_function(thetas, input_size, hidden_size, num_labels, X, Y_OH, l)

    for t in range(m):
        a1_temp = a1[t, :]  # (1, 401)
        z2_temp = z2[t, :]  # (1, 25)
        a2_temp = a2[t, :]  # (1, 26)
        h_temp = h[t, :]  # (1, 10)
        y_temp = Y_OH[t, :]  # (1, 10)

        d3_temp = h_temp - y_temp  # (1, 10)
        z2_temp = np.hstack((one, z2_temp))  # (1, 26)
        d2_temp = np.multiply((d3_temp @ theta2.T), sigmoid_grad(z2_temp))  # (1 ,26)

        delta1 += a1_temp.T @ d2_temp[:, 1:]
        delta2 += a2_temp.T @ d3_temp

    delta1 /= m
    delta2 /= m

    return J, delta1, delta2


def back_prop(thetas, input_size, hidden_size, num_labels, X, Y_OH, l=1):
    J, delta1, delta2 = pre_back_prop(thetas, input_size, hidden_size, num_labels, X, Y_OH)
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return J, grad


def reg_back_prop(thetas, input_size, hidden_size, num_labels, X, Y_OH, l=1):
    m = X.shape[0]

    J, delta1, delta2 = pre_back_prop(thetas, input_size, hidden_size, num_labels, X, Y_OH)

    theta1 = np.reshape(thetas[:(input_size + 1) * hidden_size], (input_size + 1, hidden_size))  # (401, 25)
    theta2 = np.reshape(thetas[(input_size + 1) * hidden_size:], (hidden_size + 1, num_labels))  # (26, 10)

    delta1[1:, :] += (theta1[1:,:] * l) / m
    delta2[1:, :] += (theta2[1:,:] * l) / m

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


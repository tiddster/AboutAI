import numpy as np

from MachineLearning.NeuralNetwork_BP import config
from MachineLearning.NeuralNetwork_BP.ForwardPropagate import forward_prop

input_size = config.inputSize
hidden_size = config.hiddenSize
num_labels = config.numLabel


def predict(thetas, X):
    theta1 = np.reshape(thetas[:(input_size + 1) * hidden_size], (input_size + 1, hidden_size))
    theta2 = np.reshape(thetas[(input_size + 1) * hidden_size:], (hidden_size + 1, num_labels))

    a1, z2, a2, z3, h = forward_prop(X, theta1, theta2)

    y_pred = np.array(np.argmax(h, axis=1) + 1)
    return y_pred


def getAccuracy(y_pred, y):
    total = len(y)
    cnt = 0
    for p, y in zip(y_pred, y):
        if p == y:
            cnt += 1
    return cnt / total

import numpy as np


def sigmoid(z):
    return np.exp(z) / (1 + np.exp(z))

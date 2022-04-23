from scipy.io import loadmat
import numpy as np


def textRead(file):
    data = loadmat(file)
    return data


class DP:
    def __init__(self, file):
        data = textRead(file)
        X_data = data['X']
        self.Y = data['y']

        ones = np.ones(self.Y.shape)
        self.X = np.hstack((ones, X_data))



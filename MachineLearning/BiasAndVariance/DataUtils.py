import numpy as np
from scipy.io import loadmat

data_path = 'dataset\\ex5data1.mat'


def load_data(file_path):
    data = loadmat(file_path)
    return data


class DataUtils:
    def __init__(self):
        data = load_data(data_path)
        self.X, self.y = data['X'], data['y']
        self.X_test, self.y_test = data['Xtest'], data['ytest']
        self.X_val, self.y_val = data['Xval'], data['yval']

        self.oneX, self.oneX_test, self.oneX_val = self.add_ones(self.X), self.add_ones(self.X_test), self.add_ones(self.X_val)

        self.theta = np.ones((1, self.oneX.shape[1]))

    def add_ones(self, X):
        m = X.shape[0]

        ones = np.ones((m,1))
        X = np.hstack((ones, X))

        return X

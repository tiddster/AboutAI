import numpy as np
from scipy.io import loadmat

class NN:
    def __init__(self, file):
        self.data = self.getData(file)
        self.X = np.matrix(self.data['X'])
        self.Y = np.matrix(self.data['y'])

    def getData(self,file):
        data = loadmat(file)
        return data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def cost(self, theta, X, y, l=1):
        m = len(y)
        hx = self.sigmoid(X @ theta)

        cost = np.sum(-y * np.log(hx) - (1-y) * np.log(1 - hx)) / m
        regulation = l * np.sum(theta[1:] ** 2) / (2*m)

        return cost + regulation

    def gradient(self, theta, X, y):
        m = len(y)
        hx = self.sigmoid(X @ theta)

        error = hx.T - y
        grad = error * X / m

        return grad

    def regularizedGradient(self, theta, X, y, l=1):
        punishTheta = l / len(y) * theta
        punishTheta[0] = 0
        return self.gradient(theta,X,y) + punishTheta

    def oneVsAll(self):



if __name__ == '__main__':
    nn = NN('dataset\\ex3data1.mat')

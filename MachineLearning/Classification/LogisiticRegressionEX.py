import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


class LREX:
    def __init__(self, path):
        self.positive = []
        self.negative = []
        data = np.matrix(self.textRead(path))

        row, col = data.shape

        self.X = data[:, :col-1]
        self.Y = data[:, col-1:]

        ones = np.ones((len(self.X), 1))
        self.X = np.hstack((ones, self.X))

        self.theta = np.zeros(col)

    def textRead(self, path):
        data = pd.read_csv(path, header=None, names=['x1', 'x2', 'y'])

        self.positive = data[data['y'].isin([1])]
        self.negative = data[data['y'].isin([0])]

        plt.scatter(self.positive['x1'], self.positive['x2'], marker='o')
        plt.scatter(self.negative['x1'], self.negative['x2'], marker='x')
        plt.show()

        return data

    def g(self, z):
        return 1 / (1 + np.exp(z))

    def h(self, theta, X):
        return self.g(np.dot(X, theta.T))

    def costFunc(self, theta, X, y):

        hx = self.g(np.dot(X, theta))
        cost = -np.mean(np.multiply(y, np.log(hx)) + np.multiply(1 - y, np.log(1 - hx)))

        return cost

    def gradientDescent(self, theta=None, X=None, Y=None):
        theta = self.theta if theta is None else theta
        X = np.matrix(self.X) if X is None else X
        Y = np.matrix(self.Y) if Y is None else Y

        gradTheta = np.zeros(len(theta))

        error = self.h(theta,X).T - Y

        for i in range(len(theta)):
            term = np.multiply(error, X[:, i])
            gradTheta[i] = np.sum(term) / len(X)

        return gradTheta

    def findBestTheta(self):
        theta = np.zeros(len(self.theta))
        X = np.matrix(self.X)
        Y = np.matrix(self.Y)

        print(X.shape, theta.shape, Y.shape)
        print(np.shape(self.gradientDescent()))

        result = opt.minimize(fun=self.costFunc, x0=theta, args=(X, Y), method='TNC', jac=self.gradientDescent)
        return result

    def predict(self):
        def predict(theta, X):
            probability = self.g(np.dot(X, theta))
            return [1 if x >= 0.5 else 0 for x in probability]

if __name__ == '__main__':
    lrex = LREX('dataset\ex2data1.txt')
    print(lrex.costFunction())
    print(lrex.gradientDescent())
    print(lrex.findBestTheta())
    lrex.costFunction()

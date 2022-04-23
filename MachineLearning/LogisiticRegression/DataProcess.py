import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DataProcess:
    def __init__(self, file, degree):
        self.positive = []
        self.negative = []
        data = np.matrix(self.textRead(file))
        row, col = data.shape
        self.X_data = data[:, :col - 1]
        self.Y = data[:, col - 1:]
        self.degree = degree

        if degree == 0:
            ones = np.ones((len(self.X_data), 1))
            self.X = np.hstack((ones, self.X_data))
            self.theta = np.zeros(col)
        else:
            self.X = self.polyTrans(self.X_data[:,0].ravel(), self.X_data[:,1].ravel(), degree)
            row, col = self.X.shape
            self.theta = np.ones(col)

    def textRead(self, path):
        data = pd.read_csv(path, header=None, names=['x1', 'x2', 'y'])

        self.positive = data[data['y'].isin([1])]
        self.negative = data[data['y'].isin([0])]

        plt.scatter(self.positive['x1'], self.positive['x2'], marker='o')
        plt.scatter(self.negative['x1'], self.negative['x2'], marker='x')
        plt.show()

        return data

    def polyTrans(self, x1, x2, degree=2):
        # 若  m = x1.shape[0]
        # x1,x2为一维数组，否则算出来的theta会膨胀
        temp = np.mat(x1)
        m = temp.shape[1]

        out = np.ones(m)
        for i in range(1, degree + 1):
            for j in range(i + 1):
                add = np.multiply(np.power(x1, i - j), np.power(x2, j))
                out = np.vstack([out, add])
        return out.T

    def plotBound(self, theta):
        u = np.linspace(np.min(self.X_data[:, 0]), np.max(self.X_data[:, 0]), 100)
        v = np.linspace(np.min(self.X_data[:, 1]), np.max(self.X_data[:, 1]), 100)

        uu, vv = np.meshgrid(u, v)

        z = np.dot(self.polyTrans(uu.ravel(), vv.ravel(), self.degree), theta.T)

        z = z.reshape(uu.shape)

        plt.scatter(self.positive['x1'], self.positive['x2'], marker='o')
        plt.scatter(self.negative['x1'], self.negative['x2'], marker='x')
        plt.contour(u, v, z, 0, colors='b')
        plt.show()

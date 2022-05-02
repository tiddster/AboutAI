import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model

def getModel(x, y):
    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    model.fit(x, y.ravel())
    return model

'''
class LREX:
    def __init__(self, path, degree=0):
        self.positive = []
        self.negative = []
        data = np.matrix(self.textRead(path))

        row, col = data.shape

        self.X_data = data[:, :col-1]
        self.Y = data[:, col-1:]
        self.degree = degree

        if degree == 0:
            ones = np.ones((len(self.X_data), 1))
            self.X = np.hstack((ones, self.X_data))
            self.theta = np.zeros(col)
        else:
            self.X = self.polyTrans(self.X_data[:,0].ravel(), self.X_data[:,1].ravel(), degree)
            row, col = self.X.shape
            self.theta = np.ones(col)

        self.findBestTheta()

    def textRead(self, path):
        data = pd.read_csv(path, header=None, names=['x1', 'x2', 'y'])

        self.positive = data[data['y'].isin([1])]
        self.negative = data[data['y'].isin([0])]

        plt.scatter(self.positive['x1'], self.positive['x2'], marker='o')
        plt.scatter(self.negative['x1'], self.negative['x2'], marker='x')
        plt.show()

        return data

    def sigmod(self, z):
        return np.exp(z) / (1 + np.exp(z))

    # 损失函数
    def costFunc(self, theta, X, y, l=0):
        hx = self.sigmod(X @ theta)
        cost = np.sum(np.multiply(y, np.log(hx)) + np.multiply(1 - y, np.log(1 - hx))) / len(y)
        regulation = l * np.sum(theta[1:] ** 2) / (2*len(y))
        return cost + regulation

    def gradientDescent(self, theta=None, X=None, Y=None):
        hx = self.sigmod(X @ theta)
        error = hx.T - Y
        gradTheta = error.T * X / len(Y)
        return gradTheta

    def regularizedGradient(self, theta, X, Y, l=1):
        punishTheta = l / Y.size * theta
        punishTheta[0] = 0  # 第一项不惩罚设为0
        return self.gradientDescent(theta, X, Y) + punishTheta

    # 利用scipy得到最优theta
    def findBestTheta(self):
        theta = np.zeros(len(self.theta))
        X = np.matrix(self.X)
        Y = np.matrix(self.Y)

        result = opt.minimize(fun=self.costFunc, x0=theta, args=(X, Y),jac=self.regularizedGradient, method='TNC')
        self.theta = result.x
        return result.x

    def predict(self):
        def predict(theta, X):
            probability = self.sigmod(X @ theta)
            return [1 if x >= 0.5 else 0 for x in probability]

    #多项式逻辑回归
    def polyTrans(self, x1, x2,degree=2):
        # 若  m = x1.shape[0]
        # x1,x2为一维数组，否则算出来的theta会膨胀

        temp = np.mat(x1)
        m = temp.shape[1]

        out = np.ones(m)
        # 每个Featuer的最高次数等于6
        for i in range(1, degree + 1):
            for j in range(i + 1):
                add = np.multiply(np.power(x1, i - j),np.power(x2, j))
                out = np.vstack([out, add])
        return out.T

    # 绘制多项式逻辑回归的决策边界
    def plotBound(self):
        u = np.linspace(np.min(self.X_data[:, 0]), np.max(self.X_data[:, 0]), 100)
        v = np.linspace(np.min(self.X_data[:, 1]), np.max(self.X_data[:, 1]), 100)

        uu, vv = np.meshgrid(u,v)

        z = np.dot(self.polyTrans(uu.ravel(), vv.ravel(), self.degree), self.theta)

        z = z.reshape(uu.shape)

        plt.scatter(self.positive['x1'], self.positive['x2'], marker='o')
        plt.scatter(self.negative['x1'], self.negative['x2'], marker='x')
        plt.contour(u, v, z, 0, colors='b')
        plt.show()

if __name__ == '__main__':
    lrex = LREX('dataset\ex2data2.txt', 6)
    print(lrex.findBestTheta())
    lrex.plotBound()
'''


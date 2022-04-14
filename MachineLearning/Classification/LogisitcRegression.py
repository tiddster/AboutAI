import numpy as np
import math
import matplotlib.pyplot as plt

class LR:
    def __init__(self, n, xs, ys):
        self.thetas = np.zeros(n)
        self.ys = ys
        self.xs = xs

    # 拟合函数,也就是将聚类分开的那条线
    def g(self, x, reCal=False):
        if reCal:
            X = np.array([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])
            return np.dot(self.thetas, x.T)
        else:
            return np.dot(self.thetas, x.T)

    def h(self, x):
        res = self.g(x)
        print(res, 1 / (1 + math.exp(-res)))
        return 1 / (1 + math.exp(-res))

    def J(self):
        res = 0
        m = len(self.ys)
        for x, y in zip(self.xs, self.ys):
            res += self.costFunction(x, y)
        return res / m

    def costFunction(self, x, y):
        """
        返回下列式子也可
        :return: -y*math.log(self.h(x)) - (1-y)*math.log(1-self.h(x))
        """
        if y == 1:
            return -math.log(self.h(x))
        elif y == 0:
            return -math.log(1 - self.h(x))

    def GradientDescent(self, alpha=0.2, k=10e-3):
        m = len(self.ys)
        X = self.xs
        Y = self.ys
        tempThetas = self.thetas.copy()
        lastCost = self.J()

        for i in range(len(tempThetas)):
            res = 0
            for x,y in zip(X,Y):
                res += (self.h(x) - y) * x[i]
            tempThetas[i] -= alpha*res/m

        self.thetas = tempThetas.copy()
        nowCost = self.J()

        if abs(nowCost - lastCost) > k:
            return self.GradientDescent(alpha,k)
        else:
            return self.thetas

    def plot(self, xs):
        x0Y0, x0Y1, x1Y0, x1Y1 = [], [], [], []
        for x, y in zip(xs, self.ys):
            if y == 0:
                x0Y0.append(x[0])
                x1Y0.append(x[1])
            else:
                x0Y1.append(x[0])
                x1Y1.append(x[1])

        plt.plot(x0Y0, x1Y0, '.')
        plt.plot(x0Y1, x1Y1, '.')

        plt.show()

def fitFunction(xs):
    X = np.array([])
    for index, x in enumerate(xs):
        tempX = np.array([1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])
        if index == 0:
            X = tempX
        else:
            X = np.vstack((X, tempX))
    return X

if __name__ == '__main__':
    xs = [
        [1, 0.5],
        [1, 1.5],
        [2, 1],
        [3, 1]
    ]
    ys = [0, 0, 1, 0]
    print(fitFunction(xs))
    lr = LR(6, fitFunction(xs), ys)
    print(lr.J())
    print(lr.GradientDescent())
    lr.plot(xs)
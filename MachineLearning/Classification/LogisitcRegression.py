import numpy as np
import math


class LR:
    def __init__(self, thetas, xs, ys):
        self.thetas = thetas
        self.ys = ys
        self.X = xs.copy()

        tempOnes = np.ones([len(self.ys), 1])
        self.X = np.hstack((tempOnes, self.X))

    def h(self, x):
        power = np.dot(self.thetas, x.T)
        return 1 / (1 + math.exp(power))

    def J(self):
        res = 0
        m = len(self.ys)
        for x, y in zip(self.X, self.ys):
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

    def GradientDescent(self, alpha=0.1, k=10e-6):
        m = len(self.ys)
        X = self.X
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


if __name__ == '__main__':
    thetas = [0, 1]
    xs = [
        [1],
        [0],
        [1],
        [0]
    ]
    ys = [1, 1, 0, 0]
    lr = LR(thetas, xs, ys)

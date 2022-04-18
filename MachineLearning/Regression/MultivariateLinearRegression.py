import numpy as np


# 多变量的线性回归函数
# xs 是二维, x 是一维
class MLR():
    def __init__(self, theta, xs=[], ys=[]):
        self.theta = theta
        self.xs = xs
        self.ys = ys
        self.paddedXs = self.xs.copy()

        self.length = len(ys)

        # 扩充一列x0 = 0， 便于与theta数组做矩阵内积计算
        Ones = np.ones((self.length, 1))
        self.paddedXs = np.hstack((Ones, self.paddedXs))

    def h(self, x):
        return np.dot(self.theta, x)

    def J(self):
        m = self.length
        total = 0
        for x, y in zip(self.paddedXs, self.ys):
            total += (self.h(x) - y) ** 2
        return total / (2 * m)

    def regularizedJ(self, Lambda=0):
        m = self.length
        punish = Lambda * np.sum(np.power(self.theta[1:], 2))
        return self.J() + punish / (2*m)

    def JInMatrix(self):
        m = self.length
        res = np.sum((np.dot(self.theta, self.paddedXs.T) - self.ys) ** 2)
        # 正规化 在代价函数中增加了惩罚
        # 且规定一般不将theta0加入惩罚措施
        return res / (2 * m)

    def regularizedJInMatrix(self, Lambda=0):
        m = self.length
        punish = Lambda * np.sum(np.power(self.theta[1:], 2))
        return self.J() + punish / (2 * m)

    def GradientDescent(self, alpha=0.1, k=10e-6,Lambda=0):
        lastCost = self.J()

        m = self.length
        tempThetas = self.theta.copy()

        for i in range(len(self.theta)):
            total = 0
            for x, y in zip(self.paddedXs, self.ys):
                total += (self.h(x) - y) * x[i]
            if i == 0:
                self.theta[i] -= alpha * total / m
                tempThetas[i] -= self.theta[i]
            else:
                tempThetas[i] -= alpha * (total - Lambda * tempThetas[i]) / m

        print(tempThetas)

        self.theta = tempThetas.copy()
        nowCost = self.J()

        if abs(nowCost - lastCost) < k:
            return self.theta
        else:
            return self.GradientDescent(alpha=alpha)

    def GradientDescentInMatrix(self, alpha=0.1, k=10e-6, Lambda=0):
        lastCost = self.JInMatrix()

        m = self.length
        tempThetas = self.theta - alpha / m * np.dot((np.dot(self.theta,self.paddedXs.T) - self.ys), self.paddedXs)

        self.theta = tempThetas
        nowCost = self.JInMatrix()

        if abs(nowCost - lastCost) < k:
            return self.theta
        else:
            return self.GradientDescentInMatrix(alpha=alpha)

if __name__ == '__main__':
    thetas = [0, 1]
    xs = [
        [5],
        [3],
        [0],
        [4]
    ]
    ys = [4, 4, 1, 3]
    mlr = MLR(thetas, xs, ys)
    print(mlr.J())
    print(mlr.regularizedJ(1))
    print(mlr.JInMatrix())
    print(mlr.regularizedJInMatrix(1))
    print(mlr.GradientDescent(Lambda=1))
    print(mlr.GradientDescentInMatrix())

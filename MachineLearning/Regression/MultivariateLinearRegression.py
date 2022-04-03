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
        print(self.paddedXs)

    def h(self,x):
        return np.dot(self.theta, x)

    def J(self):
        m = self.length
        total = 0
        for x, y in zip(self.paddedXs, self.ys):
            total += (self.h(x) - y) ** 2
        return total / (2*m)

    def GradientDescent(self, alpha=0.1, k=10e-6):
        lastCost = self.J()

        m = self.length
        total = 0
        tempThetas = self.theta.copy()

        for i in range(len(self.theta)):
            for x, y in zip(self.paddedXs,self.ys):
                total += (self.h(x) - y) * x[i]
            tempThetas[i] -= alpha * total / m

        self.theta = tempThetas.copy()
        nowCost = self.J()

        if abs(nowCost - lastCost) < k:
            return self.theta
        else:
            return self.GradientDescent(alpha=alpha)

if __name__ == '__main__':
    thetas = [0,1]
    xs = [
        [5],
        [3],
        [0],
        [4]
    ]
    ys = [4,4,1,3]
    mlr = MLR(thetas,xs,ys)
    print(mlr.J())
    print(mlr.GradientDescent())
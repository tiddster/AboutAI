import numpy as np
# 多变量的线性回归函数
# xs 是二维, x 是一维
class MLR():
    def __init__(self, theta, xs, ys):
        self.theta = theta
        self.xs = xs
        self.ys = ys

        self.length = len(ys)

        # 扩充一列x0 = 0， 便于与theta数组做矩阵内积计算
        Ones = np.ones(shape)((self.length, 1))
        self.xs = np.hstack((Ones, self.xs))

    def h(x):
        return np.dot(self.theta, x.T)

    def J(self):
        m = self.length
        total = 0
        for x, y in zip(self.xs, self.ys):
            total += (self.h(x) - y) ** 2
        return total / (2*m)

    def GradientDescent(self, alpha = 1, k = 10e-12):
        m = self.length
        total = 0
        tempTheta = self.theta.copy()
        
        for x, y in zip(xs,ys):
            total += (self.h(x) - y)

        for i in range(len(self.theta)):
            tempTheta[i] -= alpha * total * self.xs[i] / m

        for rec, theta in zip(recTheta, self.theta):
            if abs(rec, theta) > k:
                return self.GradientDescent()
        
        return self.theta

if __name__ == '__main__':
    print()
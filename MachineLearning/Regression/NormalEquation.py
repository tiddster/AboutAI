import numpy as np


class NE():
    def __init__(self, xs, ys):
        self.xs = np.matrix(xs)
        self.ys = ys
        self.length = len(ys)

        self.A = np.zeros([self.xs[0].shape[0]+1, self.xs[0].shape[0]+1])
        for i in range(1, self.xs[0].shape[0]+1):
            self.A[i,i] = 1

        ones = np.ones([self.length, 1])
        self.X = np.hstack((ones, self.xs))

    def NormalEquation(self, Lambda=0):
        X = self.X
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) - Lambda * self.A), X.T), self.ys)


if __name__ == '__main__':
    xs = [
        [5],
        [3],
        [0],
        [4]
    ]
    ys = [4, 4, 1, 3]
    ne = NE(xs, ys)
    print(ne.NormalEquation(1))

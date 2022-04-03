import numpy as np

class NE():
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.length = len(ys)

        ones = np.ones([self.length,1])
        self.X = np.hstack((ones,self.xs))

    def NormalEquation(self):
        X = self.X
        print(X,X.T)
        print(np.outer(X.T,X))

if __name__ == '__main__':
    xs = [
        [5],
        [3],
        [0],
        [4]
    ]
    ys = [4, 4, 1, 3]
    ne = NE(xs,ys)
    ne.NormalEquation()

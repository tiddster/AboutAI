import numpy as np


class NewtonInterpolation():
    def __init__(self,xs=[],ys=[],k=0):
        self.xs = xs
        self.ys = ys
        self.k = k
        self.C = []


    def f(self,x):
        index = self.xs.index(x)
        return self.ys[index]

    def DiffQuotient(self,xs):
        m = len(xs)
        if m == 2:
            return (self.f(xs[1]) - self.f(xs[0])) / (xs[1] - xs[0])
        else:
            res = self.DiffQuotient(xs[:-1])
            self.C.append(res)
            return (res - self.DiffQuotient(xs[1:])) / (xs[0] - xs[-1])

    def getC(self,xs):
        self.C.append(self.DiffQuotient(xs))
        return self.C

    def getValue(self,x):
        res = self.ys[0]
        m = len(self.C)
        for i in range(1,m):
            temp = 1
            for tempX in self.xs[:i]:
                temp *= (x - tempX)
            res += temp
            print(res)
        return res

if __name__ == '__main__':
    xs = [8.1,8.2,8.6,8.7]
    ys = [16.94410, 17.56492, 18.50515, 18.82091]
    NI = NewtonInterpolation(xs, ys)
    print(NI.getC(xs))
    print(NI.getValue(8.4))
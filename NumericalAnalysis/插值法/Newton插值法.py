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
        for i in range(m):
            total = 0
            for j in range(i+1):
                xj = xs[j]
                omega = 1
                tempXs = xs[:i+1].copy()
                tempXs.remove(xj)
                tempXs =[xj - x for x in tempXs]
                for x in tempXs:
                    omega *= x
                total += self.f(xj) / omega
            self.C.append(total)

    def getC(self, xs):
        self.DiffQuotient(xs)
        return self.C

    def getValue(self,X):
        res = 0
        m = len(self.C)
        for i in range(m):
            pro = 1
            for x in self.xs[:i]:
                pro *= (X - x)
            res += pro * self.C[i]
        return res

if __name__ == '__main__':
    xs = [0,10,20,30,40,50]
    ys = [10.6,3.810,1.492,0.629,0.2754,0.1867]
    NI = NewtonInterpolation(xs, ys)
    print(NI.getC(xs))
    print(NI.getValue(45))
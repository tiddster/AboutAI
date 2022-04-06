import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

class Interpolation():
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    # 拉格朗日插值法
    def lagrange(self, x=None):
        xs = self.xs.copy()
        ys = self.ys.copy()
        newX = np.linspace(max(xs), min(xs), 100)

        L = interpolate.lagrange(xs,ys)
        print(L)

        if x is not None:
            print(f"当x = {x}时, 预估值为:{L(x)}")

        plt.grid(True)
        plt.plot(newX,L(newX))
        plt.show()

    # 分段插值法
    def piecewiseLinear(self,x=None):
        xs = self.xs.copy()
        ys = self.ys.copy()
        newX = np.linspace(max(xs),min(xs),100)

        A = interpolate.interp1d(xs,ys,'linear')

        if x is not None:
            print(f"当x = {x}时, 预估值为:{A(x)}")

        plt.plot(xs,A(xs),'r')
        plt.plot(xs,ys,'o')
        plt.plot(newX,self.f(newX),'b')
        plt.show()

    # 三段样条插值法
    def cubicSpline(self, x=None):
        xs = self.xs.copy()
        ys = self.ys.copy()
        newX = np.linspace(max(xs), min(xs), 100)

        S = interpolate.interp1d(xs, ys, 'cubic')

        if x is not None:
            print(f"当x = {x}时, 预估值为:{S(x)}")

        plt.plot(newX, S(newX), 'r')
        plt.plot(xs, ys, 'o')
        plt.plot(newX, self.f(newX), 'b')
        plt.show()

    def f(self,x):
        return 1 / (1 + x**4)

if __name__ == '__main__':
    xs = np.linspace(0,50,6)
    ys = np.array([
        10.6, 3.810, 1.492, 0.629, 0.2754, 0.1867
    ])
    print(xs,ys)
    itpl = Interpolation(xs,ys)
    itpl.lagrange(2)
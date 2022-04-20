import sympy
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

class BSA:
    def __init__(self,X,a,b,n):
        self.X = X
        self.D = np.zeros([n,1])
        self.H = np.zeros([n,n])
        self.a = np.zeros([n,1])
        self.calD(n,a,b)
        self.calH(n)
        self.calA()

    def f(self, x):
        return (1 + x ** 2) ** (1 / 2)

    def calD(self,n,a,b):
        for i in range(n):
            f = lambda x:(x**i) * self.f(x)
            self.D[i][0] = integrate.quad(f, a, b)[0]

    def calH(self,n):
        for i in range(n):
            for j in range(n):
                self.H[i][j] = 1 / (i+j+1)

    def calA(self):
        self.a = np.linalg.inv(self.H) @ self.D

    def newF(self, n, x):
        mat = np.matrix([x ** i for i in range(n)])
        res = self.a.T @ mat
        return res.T

    def plot(self,a,b,n):
        px = np.linspace(a,b,100)
        Y = []
        for x in self.X:
            Y.append(self.f(x))

        plt.plot(self.X, Y,'.')
        plt.plot(px,self.newF(n,px))
        plt.show()

if __name__ == '__main__':
    x = np.linspace(0,1,10)
    bas = BSA(x,0,1,3)
    bas.plot(0,1,3)




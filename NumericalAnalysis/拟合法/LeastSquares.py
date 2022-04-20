import numpy as np
import matplotlib.pyplot as plt

class LS:
    def __init__(self, X, Y, n):
        self.X = X
        self.Y = Y
        self.n = n
        print(self.X)
        print(self.Y)

    def NormalEquation1(self):
        neX = np.zeros([self.n + 1, self.n + 1])
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                if i == 0 and j == 0:
                    neX[0][0] = len(self.Y)
                else:
                    neX[i][j] = np.sum(self.X ** (i + j))

        neY = np.zeros([self.n + 1, 1])
        for i in range(self.n + 1):
            neY[i][0] = np.dot((self.X ** i).T, self.Y)

        return np.dot(np.linalg.inv(neX), neY)

    def NormalEquation2(self):
        neX = np.zeros([len(self.Y), self.n + 1])
        for i in range(len(self.Y)):
            x = self.X[i]
            for j in range(self.n + 1):
                neX[i][j] = x ** j
        return np.dot(np.dot(np.linalg.inv(np.dot(neX.T, neX)), neX.T), self.Y)

    def plot2(self,a,b):
        thetas = self.NormalEquation2()
        print(thetas)

        def f(x):
            L = np.array([x ** i for i in range(len(thetas))])
            return np.dot(thetas, L.T)

        XS = np.linspace(min(self.X),max(self.Y),100)
        YS = []
        for x in XS:
            YS.append(f(x))

        plt.plot(self.X, self.Y, '.')
        plt.plot(XS, YS)
        plt.show()

if __name__ == '__main__':
    X = np.arange(1997,2010,1)
    Y = np.array([767,895,995,1117,1261,1437,1640,1957,2244,2489,2801,3096,3500])
    ls = LS(X, Y, 2)
    ls.plot2(1997,2010)

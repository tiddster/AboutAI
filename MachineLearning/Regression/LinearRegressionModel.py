import numpy as np

class LRM():
    def __init__(self, xs, ys, theta0=0, theta1=0):
        self.theta0 = theta0
        self.theta1 = theta1

        self.xs = xs
        self.ys = ys

    def h(self, x):
        return self.theta0 + self.theta1 * x

    def J(self):
        m = len(xs)
        total = 0
        for x, y in zip(self.xs, self.ys):
            total += (self.h(x)-y)**2
        return 1/(2*m) * total
    
    def GradientDescent(self,alpha=0.1, k=10e-6):
        lastCost = self.J()
        m = len(xs)
        total0 = 0
        total1 = 0
        for x,y in zip(self.xs, self.ys):
            total0 += self.h(x) - y
            total1 += (self.h(x) - y) * x
        self.theta0 -= alpha * total0 / m
        self.theta1 -= alpha * total1 / m

        nowCost = self.J()
        if abs(nowCost - lastCost) < k:
            return self.theta0, self.theta1
        else:
            return self.GradientDescent(alpha)

if __name__ == '__main__':
    xs = [5,3,0,4]
    y = [4,4,1,3]
    lrm = LRM(xs,y,0,1)
    print(lrm.J())
    print(lrm.GradientDescent())

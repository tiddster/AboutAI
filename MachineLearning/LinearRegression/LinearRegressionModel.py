import numpy as np

class LRM():
    def __init__(self, theta0 = 0, theta1 = 0):
        self.theta0 = theta0
        self.theta1 = theta1

    def h(self, x):
        return self.theta0 + self.theta1 * x

    def J(self, xs, ys):
        m = len(xs)
        total = 0
        for x, y in zip(xs, ys):
            total += (self.h(x)-y)**2
        return 1/(2*m) * total

if __name__ == '__main__':
    xs = [5,3,0,4]
    y = [4,4,1,3]
    lrm = LRM(0,1)
    print(lrm.J(xs, y))

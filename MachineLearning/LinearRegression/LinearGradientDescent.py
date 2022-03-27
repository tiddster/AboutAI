# 梯度下降函数
# 注意，所有参数(theta)需要同时更新
# 当越来越接近最小值时，迭代速度会越来越慢

class LGD():
    def __init__(self, lrm):
        self.lrm = lrm
        self.theta0 = lrm.theta0
        self.theta1 = lrm.theta1
    
    def GradientDescent(self,xs,ys, alpha = 1, k=10e-12):
        m = len(xs)
        total0 = 0
        total1 = 0
        for x,y in zip(xs, ys):
            total0 = self.lrm.h(x) - y
            total1 = (self.lrm.h(x) - y) * x
        lastTheta0 = self.theta0
        lastTheta1 = self.theta1
        self.theta0 -= alpha * total0 / m
        self.theta1 -= alpha * total1 / m
        if abs(lastTheta0 - self.theta0) < k and abs(lastTheta1 - self.theta1):
            return self.theta0, self.theta1
        else:
            return self.GradientDescent(xs, ys, alpha)
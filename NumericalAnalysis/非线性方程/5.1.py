import math
import scipy.optimize as opt
from 二分求解方程 import erFen as EF
from 牛顿迭代法 import Newton as NT


def f(x):
    return 3 * x ** 2 - math.exp(x)

def ff(x):
    return 6 * x - math.exp(x)

if __name__ == '__main__':
    print(f"二分法根与迭代次数分别为：{EF(f, -1, 0)}")
    print(f"牛顿迭代法根与迭代次数分别为：{opt.newton(f, -0.5, fprime = lambda x:ff(x))}")
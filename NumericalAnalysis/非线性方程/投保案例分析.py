import scipy.optimize as opt
from 牛顿迭代法 import Newton as nt
from 弦截法 import XianJie as XJ
from 二分求解方程 import erFen as EF

dataSet1 = [200, 2282, 420, 639]  #二十五岁投保
dataSet2 = [200, 1056, 300, 519]  #三十五岁投保
dataSet3 = [200, 420, 180, 399]

def f(p, q, N, M, x):
    """
    :param p: 每月投保金额
    :param q: 每月领取金额
    :param N: 投保月数
    :param M: 领取月数
    :return:
    """
    return p * x ** M - (q+p) * x ** (M-N) + q

def ff(p, q, N, M, x):
    return p * M * x **(M-1) - (q+p)*(M-N) * x ** (M-N-1)

def func(x):
    return f(dataSet2[0], dataSet2[1], dataSet2[2], dataSet2[3], x)

def ffunc(x):
    return ff(dataSet2[0], dataSet2[1], dataSet2[2], dataSet2[3], x)


# 导入库进行求解
def ErFen():
    solution = opt.root_scalar(func, method='bisect', bracket=(1.001,2))
    print(f"解为{solution.root}, 迭代次数为{solution.iterations}")

def XianJie():
    solution = opt.root_scalar(func, x0=2, x1=1.001, method='secant')
    print(f"解为{solution.root}, 迭代次数为{solution.iterations}")

def NewTon():
    solution = opt.root_scalar(func, x0=1.01, fprime=lambda x: ffunc(x), method='newton')
    print(f"解为{solution.root}, 迭代次数为{solution.iterations}")


if __name__ == '__main__':
    print("使用库函数：")
    ErFen()
    XianJie()
    NewTon()
    print("使用上节课的代码：")
    print(EF(func, 2, 1.001))
    print(XJ(func, 1.001, 2))
    print(nt(func, ffunc, 1.01))
    root = nt(func, ffunc, 1.01)[0]
    print(f"在25岁时投,r = {root - 1}")


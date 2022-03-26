import math
import scipy.optimize as opt


def f1(x):
    return math.exp(-math.exp(-x[0] - x[1])) / (1 + x[0] ** 2)


def f2(x):
    return (0.5 - x[1] * math.sin(x[0])) / math.cos(x[1])


def funcGroup(x):
    return [
        math.exp(-math.exp(-x[0] - x[1])) - x[1] * (1 + x[0] ** 2),
        x[0] * math.cos(x[1]) + x[1] * math.sin(x[0]) - 1 / 2
    ]


# 不懂点法
def BuDongDian(x, k=10e-12):
    newX = [0, 0]
    newX[1] = f1(x)
    newX[0] = f2(x)
    if abs(newX[0] - x[0]) < k and abs(newX[1] - x[1]) < k:
        return newX
    else:
        return BuDongDian(newX)


# OPT库中的不动点法
def BuDongDianOfOPT(funG, x):
    res = opt.root(funG, x)
    return res.x


if __name__ == '__main__':
    print(f"方程组其中一个解(x1, x2)为:{BuDongDian([0, 0])}")
    print(BuDongDianOfOPT(funcGroup, [0, 0]))

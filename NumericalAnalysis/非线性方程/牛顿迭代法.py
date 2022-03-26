import scipy.optimize as opt


# 原函数， a为系数数组
def f(x, a):
    m = len(a) - 1
    res = 0
    for i in range(len(a)):
        res += a[i] * x ** m
        m -= 1
    return res


# 求导后函数， a为系数数组
def ff(x, a):
    m = len(a) - 1
    res = 0
    for i in range(len(a)):
        if m - 1 >= 0:
            res += m * a[i] * x ** m
        m -= 1
    return res


def func(x):
    return f(x, [1, 2, 10, -20])


def ffunc(x):
    return ff(x, [1, 2, 10, -20])


# 原理
def Newton(fun, ffun, x, k=10e-12, times=0):
    times += 1
    xx = x - fun(x) / ffun(x)
    if abs(xx - x) < k:
        return xx, times
    else:
        return Newton(fun, ffun, xx, k, times)


# 库函数
def NewtonInOPT(fun, ffun, x):
    res = opt.newton(fun, x, fprime=lambda x: ffun(x))
    return res

# 通用库函数
def NewtonOfScalar(fun, ffun, x):
    res = opt.root_scalar(fun, x0=x, fprime=lambda x:ffun(x), method='newton')
    return res.root, res.iterations

if __name__ == '__main__':
    print(Newton(func, ffunc, 1.5))
    print(NewtonInOPT(func, ffunc, 1.5))
    print(NewtonOfScalar(func, ffunc, 1.5))

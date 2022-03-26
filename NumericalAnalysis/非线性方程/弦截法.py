import scipy.optimize as opt


def f(x):
    return x ** 3 + 2 * x - 6


# 原理
def XianJie(fun, x0, x1, a=10e-12, times=0):
    x = x1 - (x1 - x0) * fun(x1) / (fun(x1) - fun(x0))
    closer = x0 if abs(x - x0) < abs(x - x1) else x1
    if abs(x - closer) < a:
        return x, times
    else:
        return XianJie(fun, closer, x, times=times + 1)


# 调用库函数
def Secant(fun, x):
    res = opt.newton(fun, x)
    return res


def SecantOfScalar(fun, x0, x1):
    res = opt.root_scalar(fun, x0=x0, x1=x1, method='secant')
    return res.root, res.iterations


if __name__ == '__main__':
    print(XianJie(f, 1.5, 2))
    print(Secant(f, 1.5))
    print(SecantOfScalar(f, 1.5, 2))

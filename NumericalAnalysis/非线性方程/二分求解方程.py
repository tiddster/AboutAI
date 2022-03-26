import scipy.optimize as opt

def f(x):
    return 2 * x**3 - 5*x-1

# 二分法原理
def erFen(fun, a,b,k=10e-12,times=0, xs=[]):
    mid = (a+b)/2
    xs.append(mid)
    times += 1

    # 递归跳出条件，也就是求的了近似的解
    if fun(mid) * fun(a) == 0:
        return mid,times
    elif len(xs)>2 and abs(xs[len(xs)-1] - xs[len(xs)-2])<k:
        return mid, times
    #根据条件进行递归
    elif fun(mid) * fun(a) > 0:
        return erFen(fun, mid, b, times=times, xs=xs)
    elif fun(mid) * fun(a) < 0:
        return erFen(fun, a, mid, times=times, xs=xs)

# 直接调用库函数
def Bisect(fun, a, b):
    res = opt.bisect(fun, a, b)
    return res

def BisectOfScalar(fun, a, b):
    # 这种方法可以返回根和迭代次数
    res = opt.root_scalar(fun, method='bisect', bracket=(a,b))
    return res.root, res.iterations

if __name__ == '__main__':
    print(erFen(f,1,2))
    print(Bisect(f,1,2))
    print(BisectOfScalar(f,1,2))

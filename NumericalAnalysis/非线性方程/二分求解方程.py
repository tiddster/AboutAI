def f(x):
    return 2 * x**3 - 5*x-1

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


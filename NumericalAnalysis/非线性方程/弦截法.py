

def f(x):
    return x**3 + 2*x +6

def XianJie(fun, x0, x1, a=10e-12, times = 0):
    x = x1 - (x1 - x0) * fun(x1)/(fun(x1) - fun(x0))
    closer = x0 if abs(x-x0)<abs(x-x1) else x1
    if abs(x - closer) < a:
        return x, times
    else:
        return XianJie(fun, closer, x, times=times+1)

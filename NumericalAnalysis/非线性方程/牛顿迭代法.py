#原函数， a为系数数组
def f(x, a):
    m = len(a) - 1
    res = 0
    for i in range(len(a)):
        res += a[i] * x ** m
        m -= 1
    return res

#求导后函数， a为系数数组
def ff(x, a):
    m = len(a) - 1
    res = 0
    for i in range(len(a)):
        if m - 1 >= 0:
            res += m * a[i] * x ** m
        m -= 1
    return res

def Newton(x, k, a):
    global times
    times += 1
    xx = x - f(x, a) / ff(x, a)
    if abs(xx - x) < k:
        return xx
    else:
        return Newton(xx, k, a)

times = 0
print(f"解：{Newton(1, 0.000001, [1, 2, 10, -20])}, 迭代次数：{times}")

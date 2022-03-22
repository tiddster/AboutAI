def f(x):
    return 2 * x**3 - 5*x-1

def erFen(a,b,k,times):
    mid = (a+b)/2
    xs.append(mid)
    times += 1

    # 递归跳出条件，也就是求的了近似的解
    if f(mid) * f(a) == 0:
        return mid,times
    elif len(xs)>2 and abs(xs[len(xs)-1] - xs[len(xs)-2])<k:
        return mid, times
    #根据条件进行递归
    elif f(mid) * f(a) > 0:
        return erFen(mid, b, k, times)
    elif f(mid) * f(a) < 0:
        return erFen(a, mid, k, times)

if __name__ == '__main__':
    xs = []
    x, times = erFen(1,2,0.01,0)
    print(f"近似解为{x}, 二分次数为{times}, 带回f(x)求值{f(x)}")


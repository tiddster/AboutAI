def f(x):
    return 20 / (x ** 2 + 2 * x + 10)


def PuDong(x, k):
    global times
    times += 1
    if abs(f(x) - x) < k:
        return f(x)
    else:
        return PuDong(f(x), k)


if __name__ == '__main__':
    times = 0
    print(f"解：{PuDong(1, 0.0005)},次数： {times}")

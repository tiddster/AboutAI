import math


def f(x):
    if x == 0:
        return 1
    return math.sin(x) / x


def NC2(a, b, func):
    return (b - a) / 2 * (func(a) + func(b))


def NC3(a, b, func):
    return (b - a) / 6 * (func(a) + 4 * func((a + b) / 2) + func(b))


if __name__ == '__main__':
    print(NC2(0,1,f))
    print(NC3(0,1,f))

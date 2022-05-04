import math

import sympy


def f(x):
    if x == 0:
        return 1
    return math.sin(x) / x


def NC2(a, b, func):
    return (b - a) / 2 * (func(a) + func(b))


def NC3(a, b, func):
    return (b - a) / 6 * (func(a) + 4 * func((a + b) / 2) + func(b))


def trapezoid(a, b, f):
    return (b - a) / 2 * (f(a) + f(b))


def simpson(a, b, f):
    return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))


def cotes(a, b, f):
    h = (b - a) / 4
    return (b - a) / 90 * (7 * f(a) + 32 * f(a + h) + 12 * f(a + 2 * h) + 32 * f(a + 3 * h) + 7 * f(b))


def f(x):
    X = sympy.symbols('X')
    Y = sympy.sin(X) / X
    result = sympy.limit(Y, X, x)
    return float(result)

if __name__ == '__main__':
    print(trapezoid(0, 1, f))
    print(simpson(0, 1, f))
    print(cotes(0, 1, f))
    print(NC2(0, 1, f))
    print(NC3(0, 1, f))

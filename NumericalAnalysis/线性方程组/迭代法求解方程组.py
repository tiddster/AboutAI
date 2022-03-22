"""
原方程组为:
8*x1 + x2 - 2*x3 = 9
3*x1 - 10 * x2 + x3 = 19
5*x1 -2*x2 + 20*x3 = 72
"""


def Jacobi(x1, x2, x3, count=1):
    # Jacobi迭代格式
    y1 = (-1/2) * x2 + 0 * x3 + 1/2
    y2 = (-1/3) * x1 + 1/3 * x3 + 8/3
    y3 = 0 * x1 + 1/2 * x2 - 5/2

    # 如果 yk 与 xk 中的每个元素相差的绝对值小于0.00001，迭代结束
    if abs(y1 - x1) < 0.001 and abs(y2 - x2) < 0.001 and abs(y3 - x3) < 0.001:
        print(f"Jacobi最终的计算结果为{format(y1, '.5f')},{format(y2, '.4f')},{format(y3, '.4f')}")
    else:
        print(f"Jacobi第{count}次的迭代结果为{format(y1, '.5f')},{format(y2, '.4f')},{format(y3, '.4f')}")
        x1, x2, x3, count = y1, y2, y3, count + 1
        return Jacobi(x1, x2, x3, count)


Jacobi(0, 0, 0)


def Seidel(x1, x2, x3, count=1):
    # Seidel迭代法格式
    y1 = (-1 / 2) * x2 + 0 * x3 + 1 / 2
    y2 = (-1 / 3) * y1 + 1 / 3 * x3 + 8 / 3
    y3 = 0 * y1 + 1 / 2 * y2 - 5 / 2

    # 如果 yk 与 xk 中的每个元素相差的绝对值小于0.00001，迭代结束
    if abs(y1 - x1) < 0.00001 and abs(y2 - x2) < 0.00001 and abs(y3 - x3) < 0.00001:
        print(f"Seidel最终的计算结果为{format(y1, '.5f')},{format(y2, '.4f')},{format(y3, '.4f')}")
    else:
        print(f"Seidel第{count}次的迭代结果为{format(y1, '.5f')},{format(y2, '.4f')},{format(y3, '.4f')}")
        x1, x2, x3, count = y1, y2, y3, count + 1
        return Seidel(x1, x2, x3, count)


Seidel(0, 0, 0)


def Sor(x1, x2, x3, w, count=1):
    # SOR迭代个数, w为松弛因子
    y1 = (1-w)*x1 + w*((-1 / 2) * x2 + 0 * x3 + 1 / 2)
    y2 = (1-w)*x2 + w*((-1 / 3) * y1 + 1 / 3 * x3 + 8 / 3)
    y3 = (1-w)*x3 + w*(0 * y1 + 1 / 2 * y2 - 5 / 2)

    # 如果 yk 与 xk 中的每个元素相差的绝对值小于0.00001，迭代结束
    if abs(y1 - x1) < 0.0001 and abs(y2 - x2) < 0.0001 and abs(y3 - x3) < 0.00001:
        print(f"SOR最终的计算结果为{format(y1, '.5f')},{format(y2, '.4f')},{format(y3, '.4f')}")
    else:
        print(f"SOR第{count}次的迭代结果为{format(y1, '.5f')},{format(y2, '.4f')},{format(y3, '.4f')}")
        x1, x2, x3, count = y1, y2, y3, count + 1
        return Sor(x1, x2, x3, w, count)

Sor(0,0,0,1)
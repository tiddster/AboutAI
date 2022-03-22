import numpy as np
from numpy import float64

"""
原方程组为:
8*x1 + x2 - 2*x3 = 9
3*x1 - 10 * x2 + x3 = 19
5*x1 -2*x2 + 20*x3 = 72
"""

# 交换两行
def swapRow(a, b):
    for i in range(0, len(a)):
        a[i], b[i] = b[i], a[i]


# 交换两列
def swapColumn(A, c1, c2):
    for i in range(0, len(A)):
        A[i][c1], A[i][c2] = A[i][c2], A[i][c1]


# 列主消元法选择过程
def SelectInColumn(A, b, n):
    maxE = abs(A[0, 0])
    k = 0
    # 选出最大主元
    for i in range(1, n):
        if abs(A[i, 0]) > maxE:
            maxE = A[i, 0]
            k = i
    swapRow(A[0, :], A[k, :])
    b[0], b[k] = b[k], b[0]
    return 0


# 全主消元法选择主元过程
def SelectInAll(A, b, n):
    maxE = abs(A[0, 0])
    r, c = 0, 0
    for i in range(0, n):
        for j in range(0, n):
            if abs(A[i, j]) > maxE:
                maxE = A[i, j]
                r = i
                c = j
    swapRow(A[0, :], A[r, :])
    b[0], b[r] = b[r], b[0]
    swapColumn(A, 0, c)
    return c


def GaussElimin(A, b):
    n = len(b)
    """
    rec: 若用到了全主元消去法，则用于记录交换列序号
    n: b的长度
    mainE: 每次消元时最左上角的元素（未进行优化），也是回代时最右下角的元素
    subE: 消元/回代时主元素所对应列的元素
    eachE: 消元/回代时，主元素所对应行的元素
    所以 消元公式为: newE = oldE - eachE * subE/mainE
    回代公式: newE = oldE - eachE * subE/mainE
    所以回代公式和消元公式是同样的，只是顺序,方向不同而已
    """

    record = SelectInColumn(A, b, n)

    # 变为上三角矩阵->消元
    for k in range(0, n - 1):
        # 判断每一次消元的主元是否为0
        mainE = A[k][k]
        if mainE == 0:
            isZAll = True
            for j in range(k+1, n):
                if A[j][j] != 0:
                    swapRow(A[k, :], A[j, :])
                    b[k], b[j] = b[j], b[k]
                    isZAll = False
                break
            if isZAll:
                continue
        for i in range(k + 1, n):
            subE = A[i][k]
            if subE != 0:
                l = subE / mainE
                for j in range(k, n):
                    eachE = A[k][j]
                    A[i][j] -= l * eachE
                b[i] -= b[k] * l

    print(f"消元结果为{A, b}")

    # 回代
    count = 0  # 回代正向次数
    for k in range(n - 1, -1, -1):
        count += 1
        mainE = A[k][k]
        for i in range(k - 1, -1, -1):
            subE = A[i][k]
            if subE != 0:
                l = subE / mainE
                for j in range(k, n - count - 1, -1):  # 因为在n-count-1之后全是0, 所以没有遍历的必要了
                    eachE = A[k][j]
                    A[i][j] -= l * eachE
                b[i] -= b[k] * l

    print(f"回代结果为{A, b}")

    return record


def Calculate_Final_Answer(A, b):
    res = []
    for i in range(0, len(b)):
        res.append(format(b[i] / A[i, i], '.4f'))
    res[0], res[rec] = res[rec], res[0]
    resX = []
    for i in range(0, len(res)):
        resX.append("x"+str(i+1)+"="+str(res[i])+" ")
    return resX


if __name__ == '__main__':

    # 若用到了全主元消去法，则用于记录交换列序号
    rec = 0

    A = np.array([[-3.01, 0.921, 2.16],
            [0.210, -4.27, 1.73],
            [2.25, .982, -5.87]], dtype=float64)

    B = [4.01, 7.32, 6.28]

    row = len(A)
    column = len(A[0])

    rec = GaussElimin(A, B)
    print(rec)
    print(f"最终答案为：{Calculate_Final_Answer(A, B)}")

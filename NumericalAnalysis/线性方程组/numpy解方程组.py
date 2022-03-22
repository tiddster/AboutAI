import numpy.linalg
from numpy import *
"""
原方程组为:
8*x1 + x2 - 2*x3 = 9
3*x1 - 10 * x2 + x3 = 19
5*x1 -2*x2 + 20*x3 = 72
"""

A = numpy.array([
    [2, 1, 0, 0],
    [1, 3, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 2, 1]
],dtype=float64)

B = ([1, 2, 2, 0])

res = numpy.linalg.solve(A,B)

print(res)
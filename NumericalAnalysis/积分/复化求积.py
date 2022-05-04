import math


def complex_quad(a,b,n,f):
    h = (b-a)/n
    x = a
    S = 0
    for i in range(n):
        x += h
        S += f(x)
    return (h/2) * (2*S + f(a) + f(b)), h

def simpson_quad(a,b,n,f):
    h = (b-a)/n
    x = a
    S = 0
    for i in range(n):
        x1 = x + i*h
        x2 = x + (2*i - h)/2*h
        S += 4*f(x1)
        S += 2*f(x2)
    return (S + f(a) + f(b)) * h/6,h


f = lambda x: math.e ** x
i = 0
k = 0.00005
while True:
    i += 1
    c,h = complex_quad(0,1,i,f)
    ori = f(1)-f(0)
    print(f"复化求积公式结果{c},步长为{h},误差为{abs(c - ori)}")
    if abs(c - ori) < k:
        print(f"当步长为{h}时,误差小于0.5*10e-4")
        break


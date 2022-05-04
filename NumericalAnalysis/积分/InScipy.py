import numpy as np
from scipy import integrate

def cotes(a,b,n):
    f = lambda x: np.e**x
    x = np.linspace(a,b,n)
    cn = integrate.newton_cotes(n-1, 1)[0] / (n-1)
    quad = (b-a) * np.sum(cn*f(x))
    print(cn)
    print(quad)

def gauss(a,b,n):
    f = lambda x:np.e**x
    res = integrate.fixed_quad(f,a,b,n=n)[0]
    print(res)

def quad(a,b):
    f = lambda x:np.e**x
    res = integrate.quad(f,a,b)[0]
    print(res)

cotes(0,1,3)
gauss(0,1,3)
quad(0,1)
import numpy as np
from matplotlib import pyplot as plt

# 利用sin函数生成一段类似于二次函数的数据
x = np.linspace(-3/2*np.pi, 1/2*np.pi, 100)
y = np.sin(x) + np.random.rand(100)/5

# 二次多项式的函数形式
def f(x, thetas):
    return [thetas[0] + thetas[1] * i + thetas[2] * i**2 + thetas[3] * i**3 + thetas[4] * i**4 for i in x]

# 将x的值转换为二次函数的形式（常数项，一次项，二次项）的形式，并将其结合成一个矩阵，为最小二乘法的正规方程做准备
x = x.reshape((-1,1))
ones = np.ones(x.shape)
X = np.hstack((ones, x, x**2, x**3, x**4))

# 正规方程
thetas = np.linalg.inv(X.T @ X) @ X.T @ y

print(thetas)

plt.plot(x,f(x, thetas))
plt.plot(x, y, '.')
plt.show()
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

def f_(x, y):
    return y - 2 * x / y if y!=0 else 1

solve = solve_ivp(f_, [0,1], [1], t_eval=[0, 0.2, 0.4, 0.6, 0.8, 1])
print(solve.t)
print(solve.y)
plt.plot(solve.t, solve.y[0])
#plt.plot(xs, ys, '.')
plt.show()
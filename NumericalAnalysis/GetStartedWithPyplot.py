import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

x = np.linspace(-np.pi, np.pi, 100)

plt.plot(x,f(x),'m')
plt.show()
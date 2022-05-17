import numpy as np
from matplotlib import pyplot as plt

from MachineLearning.BiasAndVariance.CostFunction import reg_costFunc


def computeCost(thetas, oneX, y, oneX_cv, y_cv):
    m = oneX.shape[0]
    train_cost = []
    cv_cost = []

    for i in range(1, m+1):
        print(oneX[:i], y[:i])
        train_cost.append(reg_costFunc(thetas, oneX[:i], y[:i]))
        cv_cost.append(reg_costFunc(thetas, oneX_cv[:i], y_cv[:i]))

    plt.plot(np.arange(1, m + 1), train_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)
    plt.show()
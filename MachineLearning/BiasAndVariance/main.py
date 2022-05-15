from MachineLearning.BiasAndVariance.CostFunction import costFunc, reg_costFunc
from MachineLearning.BiasAndVariance.DataUtils import DataUtils
from MachineLearning.BiasAndVariance.Gradient import gradient, reg_gradient, findBest

du = DataUtils()
print(costFunc(du.theta, du.oneX, du.y))
print(reg_costFunc(du.theta, du.oneX, du.y))
print(gradient(du.theta, du.oneX, du.y).shape)
print(reg_gradient(du.theta, du.oneX, du.y).shape)
print(findBest(du.oneX, du.y))
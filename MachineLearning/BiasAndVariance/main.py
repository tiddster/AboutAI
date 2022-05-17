from MachineLearning.BiasAndVariance.ComputeCost import computeCost
from MachineLearning.BiasAndVariance.CostFunction import costFunc, reg_costFunc
from MachineLearning.BiasAndVariance.DataUtils import DataUtils
from MachineLearning.BiasAndVariance.Gradient import gradient, reg_gradient, findBest

du = DataUtils()

thetas = findBest(du.oneX, du.y).x
print(thetas)

computeCost(thetas, du.oneX, du.y, du.oneX_test, du.y_test)


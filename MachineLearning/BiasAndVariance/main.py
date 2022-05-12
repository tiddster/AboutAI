from MachineLearning.BiasAndVariance.CostFunction import cost_function
from MachineLearning.BiasAndVariance.DataUtils import DataUtils

du = DataUtils()
print(cost_function(du.theta, du.X, du.y))
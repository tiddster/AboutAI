import numpy as np

from MachineLearning.NeuralNetwork_BP import DataUtils as DU, config
from MachineLearning.NeuralNetwork_BP.BackPropagate import back_prop, reg_back_prop
from MachineLearning.NeuralNetwork_BP.CostFunction import cost_function, reg_cost_function
from MachineLearning.NeuralNetwork_BP.DataUtils import plot_images, saveWeight, readWeight
from MachineLearning.NeuralNetwork_BP.Gradient import gradient
from MachineLearning.NeuralNetwork_BP.Predict import predict, getAccuracy

du = DU.DataUtils()
thetas = du.thetas
Y = du.Y
Y_OH = du.Y_OH
X = du.X

pickXs, pickYs = plot_images(X, Y)

thetas = readWeight(DU.save_path)

y_pred = predict(thetas,pickXs)

print(y_pred)
print(getAccuracy(y_pred, pickYs))
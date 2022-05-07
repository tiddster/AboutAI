import numpy as np

from MachineLearning.NeuralNetwork_BP import config
from scipy.optimize import minimize

from MachineLearning.NeuralNetwork_BP.BackPropagate import back_prop, reg_back_prop
from MachineLearning.NeuralNetwork_BP.CostFunction import reg_cost_function

input_size = config.inputSize
hidden_size = config.hiddenSize
num_labels = config.numLabel


def gradient(thetas, X, Y_oh, l=0.3):
    f_min = minimize(fun=reg_cost_function, x0=thetas, args=(input_size, hidden_size, num_labels, X, Y_oh, l), jac=reg_back_prop, options={'maxiter':10})
    return f_min
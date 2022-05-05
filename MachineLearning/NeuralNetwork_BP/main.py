from MachineLearning.NeuralNetwork_BP import DataUtils as DU, config
from MachineLearning.NeuralNetwork_BP.BackPropagate import back_prop, reg_back_prop
from MachineLearning.NeuralNetwork_BP.CostFunction import cost_function, reg_cost_function

du = DU.DataUtils()
thetas = du.thetas
Y_OH = du.Y_OH
X = du.X

J, grad = reg_back_prop(thetas, config.inputSize, config.hiddenSize, config.outputSize, X, Y_OH)

print(J, grad.shape)
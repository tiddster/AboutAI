from MachineLearning.NeuralNetwork_BP import DataUtils as DU, config
from MachineLearning.NeuralNetwork_BP.CostFunction import costFunction

du = DU.DataUtils()
thetas = du.thetas
Y_OH = du.Y_OH
X = du.X

print(costFunction(thetas, config.inputSize, config.hiddenSize, config.outputSize, X, Y_OH))
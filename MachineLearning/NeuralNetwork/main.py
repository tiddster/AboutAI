import OneVsAll as OVA
import MachineLearning.NeuralNetwork.DataProcess as DP
import PredictOVA as POVA

if __name__ == '__main__':
    dp = DP.DP('dataset\ex3data1.mat')
    labels = OVA.getAllLabels(dp.Y)
    thetas = OVA.oneVsAll(dp.X, dp.Y, len(labels), 1)

    print(POVA.getAccuracy(dp.X, dp.Y, thetas))


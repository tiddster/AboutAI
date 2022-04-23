import numpy as np

import OneVsAll as OVA
import MachineLearning.NeuralNetwork.DataProcess as DP
import PredictOVA as POVA

if __name__ == '__main__':
    dp = DP.DP('dataset\ex3data1.mat')
    labels = OVA.getAllLabels(dp.Y)
    thetas = OVA.oneVsAll(dp.oneX, dp.Y, len(labels), 1)
    pickXs, pickYs = DP.plot_images(dp.X, dp.Y)

    print(POVA.getAccuracy(pickXs.reshape((-1,400)), pickYs, thetas))


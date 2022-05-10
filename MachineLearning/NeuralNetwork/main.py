import numpy as np

import OneVsAllGradient as OVA
import MachineLearning.NeuralNetwork.DataProcess as DP
import PredictOVA as POVA
import NN


if __name__ == '__main__':
    dp = DP.DP('dataset\ex3data1.mat')
    dpT = DP.DPTheta('dataset\ex3weights.mat')

    pickX, pickY = DP.plot_images(dp.X, dp.Y)

    pros = NN.Layer2(pickX, dpT.theta1, dpT.theta2)
    print(POVA.getAccuracy(pickX, pickY, pros))


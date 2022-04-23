import numpy as np

from DataProcess import DataProcess as DP
import Gradient
import Predict
import LogisiticRegression

if __name__ == "__main__":
    dp = DP('dataset\\ex2data2.txt', 10)
    res = Gradient.findBest(len(dp.theta), dp.X, dp.Y, 2)
    theta = res.x
    dp.plotBound(theta)

    print(Predict.getAccuracy(theta, dp.X, dp.Y))

    Y = np.array(dp.Y)
    model = LogisiticRegression.getModel(dp.X, Y)
    print(model.score(dp.X, Y))


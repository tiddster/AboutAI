import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

from MachineLearning.NeuralNetwork_BP import config

train_path = 'dataset\\ex4data1.mat'
weights_path = 'dataset\\ex4weights.mat'


def textRead(file):
    data = loadmat(file)
    return data


def plot_an_image(X, y):
    """
    随机打印一个数字
    :param X: 不带常量1的X
    """
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.matshow(image.reshape((20, 20)).T, cmap='gray_r')
    plt.xticks([])  # 去除刻度，美观
    plt.yticks([])
    plt.show()
    print('this should be {}'.format(y[pick_one]))
    return image


def plot_images(X, y):
    """
    随机打印一个数字
    :param X: 不带常量1的X
    """
    picks = [np.random.randint(0, X.shape[0]) for i in range(100)]
    pickXs = X[picks, :]

    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(12, 12))
    for r in range(10):
        for c in range(10):
            ax[r, c].matshow(pickXs[r * 10 + c].reshape((20, 20)).T, cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
    plt.show()
    pickYs = y[picks]

    print('this should be {}'.format(pickYs.ravel()))
    return pickXs, pickYs


class DataUtils:
    def __init__(self):
        data = textRead(train_path)
        self.X = data['X']
        self.Y = data['y']

        ones = np.ones(self.Y.shape)
        self.oneX = np.hstack((ones, self.X))

        self.Y_OH = self.oneHotY()

        self.thetas = self.initTheta()

    def oneHotY(self):
        encoder = OneHotEncoder(sparse=False)
        Y_OH = encoder.fit_transform(self.Y)
        return Y_OH

    # 初始化thetas全部为数值较小的随机数，否则BP算法算出的权值全都一样
    def initTheta(self):
        hiddenSize = config.hiddenSize
        inputSize = config.inputSize
        outputSize = config.outputSize
        thetas = (np.random.random(size=hiddenSize * (inputSize + 1) + outputSize * (hiddenSize + 1)) - 0.5) * 0.25
        return thetas
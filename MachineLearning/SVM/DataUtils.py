import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat

path = 'dataset\\ex6data1.mat'


def loadData(filePath):
    data = loadmat(filePath)
    return data


class DataUtils:
    def __init__(self):
        data = loadData(path)
        self.x = data['X']
        self.y = data['y']
        self.pre_process(data)

    def pre_process(self, data):
        tempData = pd.DataFrame(data['X'], columns=['X1', 'X2'])

        positive = data[data['y'].isin([1])]
        negative = data[data['y'].isin([0])]

        plt.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
        plt.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')

        plt.show()

du = DataUtils()
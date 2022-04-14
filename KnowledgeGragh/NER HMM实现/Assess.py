import numpy as np

class Assess():
    # 利用混淆矩阵CM来求，准确率和召回率
    # 其中CM[i][j] 为 标记为i类，结果被预测为j类的数量
    def __init__(self, originLabels, predictLabels, hmm=None):
        self.originLabels = originLabels
        self.predictLabels = predictLabels

        length = len(originLabels)
        self.label_to_index = hmm.label_to_index

        self.CM = np.zeros([len(self.label_to_index),len(self.label_to_index)])

        self.assessModel()

    def assessModel(self):
        for OList, PList in zip(self.originLabels, self.predictLabels):
            for OLabel, PLabel in zip(OList, PList):
                OIndex = self.label_to_index[OLabel]
                PIndex = self.label_to_index[PLabel]
                self.CM[OIndex][PIndex] += 1

    def getRecall(self, label):
        index = self.label_to_index[label]
        fenzi = self.CM[index][index]
        fenmu = np.sum(self.CM[index])
        return fenzi/fenmu

    def getPrecision(self, label):
        index = self.label_to_index[label]
        fenzi = self.CM[index][index]
        fenmu = 0
        for PList in self.CM:
            fenmu += PList[index]
        return fenzi/fenmu

    def F1_score(self, label):
        recall = self.getRecall(label)
        precision = self.getPrecision(label)
        return 2 * precision * recall / (recall + precision)

'''
if __name__ == '__main__':
    OLabels = [['O','B','B','B','O','O']]
    PLabels = [['O','B','B','O','O','B']]
    assess = Assess(OLabels, PLabels)
    assess.assessModel()
    print(assess.CM)
    print(assess.getPrecision('O'))
    print(assess.getPrecision('B'))
'''



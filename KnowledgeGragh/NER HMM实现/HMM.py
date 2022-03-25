import os
import pickle

import numpy as np
import  TextRead


class HMM():
    def __init__(self, wordList, labelList):
        self.wordList = wordList
        self.labelList = labelList

        self.label_to_index = {"O": 0, "B-KNOW": 1, "I-KNOW": 2, "B-PRIN": 3,"I-PRIN":4}
        self.index_to_label = ["O", "B-KNOW", "I-KNOW", "B-PRIN","I-PRIN"]

        length = len(self.index_to_label)
        self.initMatrix = np.zeros(length)
        self.initDict = {"sumMat":self.initMatrix, "normalMat":self.initMatrix}

        self.transferMatrix = np.zeros((length,length))
        self.transferDict = {"sumMat":self.transferMatrix, "normalMat":self.transferMatrix}

        self.emitMatrix = {
            "O": {"total": 0},
            "B-KNOW": {"total": 0},
            "I-KNOW": {"total": 0},
            "B-PRIN": {"total": 0},
            "I-PRIN": {"total": 0}
        }
        self.emitDict = {"sumMat":self.emitMatrix, "normalMat":self.emitMatrix}

    def setInitMatrix(self):
        labelList = self.labelList
        for labels in labelList:
            if labels:
                index = self.label_to_index[labels[0]]
                self.initMatrix[index] += 1
                self.initDict.update({"sumMat": self.initMatrix})

    # 计算 训练集 中 标签->下一个标签 的转换数
    # 例如B->I->I，则[B,I]转换矩阵中对应的值+1, [I,I] +1
    def setTransferMatrix(self):
        labelList = self.labelList
        for labels in labelList:
            for i in range(len(labels)-1):
                nowLabel = labels[i]
                nextLabel = labels[i+1]
                nowIndex = self.label_to_index[nowLabel]
                nextIndex = self.label_to_index[nextLabel]
                self.transferMatrix[nowIndex][nextIndex] += 1
        self.transferDict.update({"sumMat": self.transferMatrix})

    # 发射矩阵,记录每一个[标签，字]出现的次数
    def setEmitMatrix(self):
        wordList = self.wordList
        labelList = self.labelList
        for words, labels in zip(wordList, labelList):
            for w, l in zip(words, labels):
                self.emitMatrix[l][w] = self.emitMatrix[l].get(w,0)+1
                self.emitMatrix[l]["total"] += 1
        self.emitDict.update({"sumMat": self.emitMatrix})

    # 标准化
    def normalize(self):
        self.initMatrix = self.initMatrix / np.sum(self.initMatrix)
        self.transferMatrix = self.transferMatrix / np.sum(self.transferMatrix, axis=1)

        for label, dictionary in self.emitMatrix.items():  # 二级字典，这里每次遍历拿到的分别是："O" 和 {"total": 2, "word1":1, "word2":1} 等
            for word, times in dictionary.items():     # 对上面拿到的字典再次遍历，现在就可以拿到 "word1" 和 1 等
                if word != "total":
                    self.emitMatrix[label][word] = times/self.emitMatrix[label]["total"] * 100

        self.initDict.update({"normalMat": self.initMatrix})
        self.transferDict.update({"normalMat": self.transferMatrix})
        self.emitDict.update({"normalMat": self.emitMatrix})

    def getMatrix(self, modelFileName):
        if os.path.exists(modelFileName):
            f = open(modelFileName, "rb")
            self.initDict, self.transferDict, self.emitDict = pickle.load(f)
            self.initMatrix = self.initDict.get("normalMat", self.initMatrix)
            self.transferMatrix = self.transferDict.get("normalMat", self.transferMatrix)
            self.emitMatrix = self.emitDict.get("normalMat", self.emitMatrix)
            f.close()
        else:
            self.train(modelFileName)

    def train(self, modelFileName):
        if os.path.exists(modelFileName):
            f = open(modelFileName, "rb")
            self.initDict, self.transferDict, self.emitDict = pickle.load(f)
            self.initMatrix = self.initDict.get("sumMat", self.initMatrix)
            self.transferMatrix = self.transferDict.get("sumMat", self.transferMatrix)
            self.emitMatrix = self.emitDict.get("sumMat", self.emitMatrix)

        self.setInitMatrix()
        self.setTransferMatrix()
        self.setEmitMatrix()

        self.normalize()

        print(self.initDict)
        print(self.transferDict)
        print(self.emitDict)

        f = open(modelFileName, "wb")
        pickle.dump([self.initDict, self.transferDict, self.emitDict], f)
        f.close()

    def clearModel(self, modelFile):
        if os.path.exists(modelFile):
            f = open(modelFile, "w")
            f.truncate()
            f.close()

if __name__ == '__main__':
    wordList, labelList = TextRead.TextRead.readFile("test.txt")
    hmm = HMM(wordList, labelList)
    hmm.train("model.txt")




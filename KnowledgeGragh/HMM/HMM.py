import os
import pickle
import re

import numpy as np

from KnowledgeGragh.HMM.HMM分词 import textRead


class HMM:
    def __init__(self, textFile, labelFile):
        self.labels = open(labelFile, "r", encoding="utf-8").read().split("\n")
        self.text = open(textFile, "r", encoding="utf-8").read().split("\n")

        self.labels_to_index = {"B": 0, "M": 1, "S": 2, "E": 3}
        self.index_to_labels = ["B", "M", "S", "E"]  # 这里为什么没有写成上面的形式，因为这里是用列表存储的，列表本身就有 下表和值 对应关系

        self.labelsLength = len(self.index_to_labels)
        self.initMatrix = np.zeros(self.labelsLength)
        self.transferMatrix = np.zeros((self.labelsLength, self.labelsLength))

        self.emitMatrix = {
            "B": {"total": 0},
            "M": {"total": 0},
            "S": {"total": 0},
            "E": {"total": 0}
        }

    def setInitMatrix(self, labels):
        # BMSE 四种状态, 对应状态出现 1次 就 +1
        index = self.labels_to_index[labels[0][0]]
        self.initMatrix[index] += 1

    def setTransferMatrix(self, labels):
        # 将labels连接起来
        labelsLinked = "".join(labels)
        str1 = labelsLinked[:-1]
        str2 = labelsLinked[1:]

        # 同时遍历str1 与 str2
        for s1, s2 in zip(str1, str2):
            index1 = self.labels_to_index[s1]
            index2 = self.labels_to_index[s2]
            self.transferMatrix[index1, index2] += 1

    def setEmitMatrix(self, words, labels):
        wordsLinked = "".join(words)
        labelsLinked = "".join(labels)
        for word, label in zip(wordsLinked, labelsLinked):
            # 字典的get(key, default)方法：获取key对应的值，若不存在；则添加 key：default 键对
            self.emitMatrix[label][word] = self.emitMatrix[label].get(word, 0) + 1
            self.emitMatrix[label]["total"] += 1

    def normalize(self):
        self.initMatrix = self.initMatrix / np.sum(self.initMatrix)
        # np.sum(a, axis=1,keepdims=True)表示对a按照行求和
        # 例如a=[[1,2],[3,4]]，返回的是[3,7]
        self.transferMatrix = self.transferMatrix / np.sum(self.transferMatrix, axis=1)

        for label, dictionary in self.emitMatrix.items():  # 二级字典，这里每次遍历拿到的分别是："B" 和 {"total": 2, "word1":1, "word2":1} 等
            for word, times in dictionary.items():  # 对上面拿到的字典再次遍历，现在就可以拿到 "word1" 和 1 等
                if word != "total":
                    self.emitMatrix[label][word] = times / self.emitMatrix[label]["total"] * 1000

    def train(self):
        modelFile = 'model.txt'
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
        if os.path.exists(modelFile):  # 如果此前已经训练好模型，直接调用就行
            self.initMatrix, self.transferMatrix, self.emitMatrix = pickle.load(open(modelFile, "rb"))
            print(self.initMatrix, self.transferMatrix, self.emitMatrix)
            return

        for words, labels in zip(self.text, self.labels):
            words = re.split(pattern, words)
            labels = re.split(" ", labels)
            self.setInitMatrix(labels)
            self.setTransferMatrix(labels)
            self.setEmitMatrix(words, labels)

        self.normalize()
        print(self.initMatrix, self.transferMatrix, self.emitMatrix)
        pickle.dump([self.initMatrix, self.transferMatrix, self.emitMatrix], open(modelFile, "wb"))

if __name__ == '__main__':
    print("开始读取数据......")
    textRead('a.txt', 'g.txt')
    print("数据处理完毕!")
    hmm = HMM('a.txt', 'g.txt')
    print("开始训练HMM模型......")
    hmm.train()
    print("模型训练结束!\n")


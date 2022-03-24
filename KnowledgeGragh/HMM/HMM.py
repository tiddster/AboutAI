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

        # TODO：
        # 这三个dict是按照发射矩阵的思路储存计算总值，以实现重复更新模板，可长期使用
        # sum: 统计过的词语个数， sumMat: 未标准化的矩阵，也就是setInitMatrix之后的原矩阵  normalMat：已经标准化后的矩阵
        # 统计时更新前两个， 标准化时更新后一个
        self.initMatrix = np.zeros(self.labelsLength)
        self.initDict = {"sum": 0, "sumMat": self.initMatrix, "normalMat": self.initMatrix}

        self.transferMatrix = np.zeros((self.labelsLength, self.labelsLength))
        self.transferDict = {"sum": [[0], [0], [0], [0]], "sumMat": self.transferMatrix,
                             "normalMat": self.transferMatrix}

        self.emitMatrix = {
            "B": {"total": 0},
            "M": {"total": 0},
            "S": {"total": 0},
            "E": {"total": 0}
        }
        self.emitDict = {"sumMat": self.emitMatrix, "normalMat": self.emitMatrix}

    # 计算 每一个词语 第一个字的标签
    def setInitMatrix(self, label):
        # BMSE 四种状态, 对应状态出现 1次 就 +1
        # 统计每一个词语最开头的标签（其实思考得知也只有B和S会纳入统计）
        if label:
            index = self.labels_to_index[label[0]]
            self.initMatrix[index] += 1
        self.initDict.update({"sum": np.sum(self.initMatrix), "sumMat": self.initMatrix})

    # 计算 训练集 中 标签->下一个标签 的转换数
    # 例如B->M->E，则[B,M]转换矩阵中对应的值+1, [M,E] +1
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

        self.transferDict.update({"sum": np.sum(self.transferMatrix, axis=1), "sumMat": self.transferMatrix})

    # 计算 [字, 标签]的次数， 例如：{B: {一: 1}}代表 以一作为B出现过一次
    def setEmitMatrix(self, words, labels):
        wordsLinked = "".join(words)
        labelsLinked = "".join(labels)
        for word, label in zip(wordsLinked, labelsLinked):
            # 字典的get(key, default)方法：获取key对应的值，若不存在；则添加 key：default 键对
            self.emitMatrix[label][word] = self.emitMatrix[label].get(word, 0) + 1
            self.emitMatrix[label]["total"] += 1
        self.emitDict.update({"sumMat": self.emitMatrix})

    # 标准化算出权重
    def normalize(self):
        self.initMatrix = self.initMatrix / np.sum(self.initMatrix)
        # np.sum(a, axis=1,keepdims=True)表示对a按照行求和
        # 例如a=[[1,2],[3,4]]，返回的是[3,7]
        self.transferMatrix = self.transferMatrix / np.sum(self.transferMatrix, axis=1)

        for label, dictionary in self.emitMatrix.items():  # 二级字典，这里每次遍历拿到的分别是："B" 和 {"total": 2, "word1":1, "word2":1} 等
            for word, times in dictionary.items():  # 对上面拿到的字典再次遍历，现在就可以拿到 "word1" 和 1 等
                if word != "total":
                    self.emitMatrix[label][word] = times / self.emitMatrix[label]["total"] * 100
        self.initDict.update({"normalMat": self.initMatrix})
        self.transferDict.update({"normalMat": self.transferMatrix})
        self.emitDict.update({"normalMat": self.emitMatrix})

    # 获取三个矩阵, 若存在直接读取数据，不存在则训练
    def getMatrix(self, modelFile):
        if os.path.exists(modelFile):
            f = open(modelFile, "rb")
            self.initDict, self.transferDict, self.emitDict = pickle.load(f)
            self.initMatrix = self.initDict.get("normalMat", self.initMatrix)
            self.transferMatrix = self.transferDict.get("normalMat", self.transferMatrix)
            self.emitMatrix = self.emitDict.get("normalMat", self.emitMatrix)
            f.close()
        else:
            self.train(modelFile)

    def train(self, modelFile):
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'

        # TODO：拿到DICT，进行重新
        if os.path.exists(modelFile):  # 如果此前已经训练好模型，直接调用就行
            f = open(modelFile, "rb")
            self.initDict, self.transferDict, self.emitDict = pickle.load(f)
            self.initMatrix = self.initDict.get("sumMat", self.initMatrix)
            self.transferMatrix = self.transferDict.get("sumMat", self.transferMatrix)
            self.emitMatrix = self.emitDict.get("sumMat", self.emitMatrix)
            f.close()
            # self.initMatrix, self.transferMatrix, self.emitMatrix = pickle.load(open(modelFile, "rb"))
            # print(self.initMatrix, self.transferMatrix, self.emitMatrix)
            # return

        for words, labels in zip(self.text, self.labels):
            # 像步骤一一样将词分开，并将对应的标签分开
            words = re.split(pattern, words)
            labels = re.split(" ", labels)
            # 上面将labels分开之后labels就是列表，所以要遍历每一个元素（也就是标签组中的每一个标签），作为参数去初始化 初始矩阵
            for l in labels:
                self.setInitMatrix(l)
            self.setTransferMatrix(labels)
            self.setEmitMatrix(words, labels)

        self.normalize()
        # print(self.initMatrix, self.transferMatrix, self.emitMatrix)
        # TODO： 储存Dict的值，方便更新数据以及取出以及训练好的数据
        f = open(modelFile, "wb")
        pickle.dump([self.initDict, self.transferDict, self.emitDict], f)
        f.close()

    def clearModel(self, modelFile):
        if os.path.exists(modelFile):
            f = open(modelFile, "w")
            f.truncate()
            f.close()

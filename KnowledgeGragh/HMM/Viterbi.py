class Viterbi():
    def __init__(self, hmm):
        self.index_to_Label = hmm.index_to_labels
        self.labels_to_index = hmm.labels_to_index

        self.initMatrix = hmm.initMatrix
        self.transferMatrix = hmm.transferMatrix
        self.emitMatrix = hmm.transferMatrix

        self.allPath = []

    def getPath(self, text):
        for label in self.index_to_Label:
            index = self.labels_to_index[label]
            # 应该是计算text的首字符所 对应的标签概率
            probability = self.initMatrix[index] * self.emitMatrix[label].get(text[0], 0)
            path = [label]
            # 以有权图的形式记录 首字符 进入某一个标签 的概率  allPath = [概率,[路径]]
            self.allPath.append((probability, path))

        # 循环逻辑：针对每一个字而言(一重循环)
        # 去寻找在训练集中该字的标签，并获取(字，标签)的概率(二重循环)
        # 在(字，标签)的基础上，遍历路径，获取路径末尾的标签，计算他们连接起来的概率，更新路径
        # 计算概率公式实际上是条件概率（？）：如何走到该末尾标签的概率 * (字，标签)的概率 * 转换矩阵(末尾标签，字标签)的概率
        for word in text[1:]:
            tmpAllPath = []

            # 判断该词是否在发射矩阵中出现过
            isNeverSeen = word not in self.emitMatrix['S'].keys() and \
                          word not in self.emitMatrix['M'].keys() and \
                          word not in self.emitMatrix['E'].keys() and \
                          word not in self.emitMatrix['B'].keys()

            for label in self.index_to_Label:
                # 获取(字，标签)的概率
                emitP = self.emitMatrix[label].get(word, 0) if not isNeverSeen else 1.0

                # 以该该label结尾的路径
                newPath = []

                for nodes in self.allPath:
                    indexEnd = self.labels_to_index[nodes[1][-1]]  # 获取最后路径的最终标签对应的索引值
                    indexNow = self.labels_to_index[label]
                    # 这一句是计算并记录从 末尾标签 到 现在标签 的概率，并连接该路径
                    newPath.append((nodes[0] * emitP * self.transferMatrix[indexEnd][indexNow], nodes[1] + [label]))
                # 从新添加的所有路径中选择概率最大的
                tmpAllPath.append((max(newPath)))
            self.allPath = tmpAllPath

            self.getRes(text)

    def getRes(self, text):
        res = ""
        # max(self.allPath)[1]: 选概率最大的路径集合，allPath中有两个元素0对应概率， 1对应路径
        for t, p in zip(text, max(self.allPath)[1]):
            res += t
            if p == "S" or p == "E":
                res += " "
        return res, max(self.allPath)[1]
